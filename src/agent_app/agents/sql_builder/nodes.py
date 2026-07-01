from typing import Callable, Dict, Any, Optional, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from pydantic import BaseModel
from collections import defaultdict
from agent_app.agents.response_struct.agent import ResponseStructAgent
from agent_app.agents.sql_builder.model import AgentState, SchemaRetriever, SchemaRetrieverResult, ExtractIntentResult, IntentExtractionError, TableRelevanceAssessment
from agent_app.domain.entities.database_schema import TableSchema
from agent_app.shared.database.helper.database_helper import DatabaseHelper
from agent_app.shared.model.response import Result
from agent_app.shared.utils.content import remove_think_blocks
import asyncio


class QueryBuilderAgentNodes:
    def __init__(self, response_struct_agent: ResponseStructAgent):
        self._response_struct_agent = response_struct_agent
        pass

    def extract_intent_node(self, state: AgentState) -> dict:
        """
        LangGraph Node: Takes raw user query and transforms it into 
        vector-search-optimized strings.
        """
        user_raw_input = state.get("user_query", "")
        print(f"\n--- [NODE] Extracting Intent for: '{user_raw_input}' ---")

        # System prompt explicitly guiding the LLM on how to think for a metadata search
        system_instruction = """
        You are an expert database architect helping retrieve database schemas.

        Your job is to convert a user's natural language question into semantic search
        keywords for retrieving relevant database tables and columns from a vector index.

        Generate keywords from multiple perspectives:

        1. Business entities
        - student
        - teacher
        - customer
        - invoice
        - product

        2. Business processes
        - enrollment
        - payment
        - attendance
        - purchase
        - shipment

        3. Possible table names
        - students
        - student
        - student_profile
        - student_registration

        4. Possible column names
        - student_id
        - user_id
        - created_at
        - status
        - total_amount

        5. Synonyms
        - employee ↔ staff ↔ worker
        - class ↔ course
        - purchase ↔ order
        - client ↔ customer

        6. Database terminology
        - mapping
        - association
        - relationship
        - transaction
        - history
        - detail
        - summary

        Rules:

        - Return 5–10 high-quality keywords.
        - Include singular and plural forms when useful.
        - Expand abbreviations.
        - Infer related business concepts.
        - Prefer nouns over verbs.
        - Do not invent unrelated concepts.
        - Remove duplicates.
        """

        try:

            extraction_result = self._response_struct_agent.invoke(
                schema_cls=ExtractIntentResult,
                llm=state["llm"],
                prompts=[
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_raw_input),
                ],
                max_retry=3
            )

            if extraction_result.error:
                raise IntentExtractionError(f"Intent extraction failed: {extraction_result.error.message}")
            
            # Combine the optimized query and keywords to give our retrieval engine maximum context
            # combined_search_terms = state.get("search_keywords", []) + extraction_result.data.search_keywords
            
            print(f"🎯 Optimized Search Terms Generated: {",".join(extraction_result.data.build_retrieval_queries())}")
            
            # Return updates to the graph state fields
            return {
                "extract_intent_result": extraction_result.data,
                "retry_count": 0, # Initialize tracking
                "execution_error": None # Reset previous errors
            }
            
        except Exception as e:
            print(f"❌ Intent Extraction Node Failed: {e}")
            # Fallback strategy: if LLM fails, pass raw input to keep graph running safely
            raise IntentExtractionError(f"Intent extraction failed: {e}") from e

    def _build_referenced_by_summary(self, table: TableSchema) -> str:
        """
        Build a concise summary of tables that reference this table.
        Suitable for providing as LLM context.
        """
        if not table.referenced_by:
            return (
                f"The `{table.name}` table is not referenced by any other tables."
            )

        lines: List[str] = [
            f"The `{table.name}` table is referenced by the following tables:"
        ]

        for ref in table.referenced_by:
            relationships = ", ".join(
                f"`{src}` → `{table.name}.{dst}`"
                for src, dst in zip(ref.columns, ref.referred_columns)
            )

            lines.append(
                f"- `{ref.reference_table}` references `{table.name}` via {relationships}."
            )

        return "\n".join(lines)


    def _build_candidate_summary(
        self, tables: list[SchemaRetrieverResult], user_query: str
    ) -> str:
        blocks = []
        for idx, entry in enumerate(tables, 1):
            schema = entry.table_schema
            fk_lines = (
                "\n".join(
                    f"  - ({', '.join(fk.constrained_columns)}) "
                    f"→ {fk.referred_table}({', '.join(fk.referred_columns)})"
                    for fk in schema.foreign_keys
                )
                or "  None"
            )
            blocks.append(
                f"Candidate {idx}: {schema.name}\n"
                f"Columns: {', '.join(c.name for c in schema.columns)}\n"
                f"Foreign Keys:\n{fk_lines}\n"
                f"Referenced By:\n  {self._build_referenced_by_summary(schema)}\n"
            )

        return (
            f"User Question:\n{user_query}\n\n"
            "Candidate Tables:\n\n"
            + "\n---\n".join(blocks)
        )

    async def relevant_schema_retriever_node(self, state: AgentState) -> dict:
        """
        Pure retrieval node: batch vector search + FK-graph one-hop expansion.
        On subsequent passes (from validate_schema → loop), merges new results
        with previously retrieved tables so the candidate pool grows.
        """

        intent: ExtractIntentResult = state["extract_intent_result"]
        retriever: SchemaRetriever = state["schema_retriever"]
        additional_kw = state.get("additional_search_keywords", [])

        queries = list(intent.build_retrieval_queries())
        if additional_kw:
            deduped = list(dict.fromkeys(additional_kw))
            print(f"🔄 Additional retrieval pass with keywords: {deduped}")
            queries.extend(deduped)

        seen = set()
        queries = [q for q in queries if q and not (q in seen or seen.add(q))]

        try:
            hit_counts: dict[str, int] = defaultdict(int)
            score_sums: dict[str, float] = defaultdict(float)
            candidate_map: dict[str, SchemaRetrieverResult] = {}

            semaphore = asyncio.Semaphore(3)

            async def retrieve_one(q: str) -> None:
                async with semaphore:
                    print(f"🔍 Retrieving: '{q}'")
                    results = await asyncio.to_thread(retriever, q)
                    for res in results:
                        name = res.table_schema.name
                        if name not in candidate_map or res.score > candidate_map[name].score:
                            candidate_map[name] = res
                        hit_counts[name] += 1
                        score_sums[name] += res.score

            async def retrieve_batch(ql: list[str]) -> None:
                await asyncio.gather(*[retrieve_one(q) for q in ql])

            async def expand_fk_neighbors() -> None:
                neighbor_queries = []
                for entry in candidate_map.values():
                    for fk in entry.table_schema.foreign_keys:
                        if fk.referred_table not in candidate_map:
                            neighbor_queries.append(fk.referred_table)
                    for ref in entry.table_schema.referenced_by:
                        if ref.reference_table not in candidate_map:
                            neighbor_queries.append(ref.reference_table)
                if neighbor_queries:
                    unique = list(dict.fromkeys(neighbor_queries))
                    print(f"🔗 FK-graph expansion — fetching neighbors: {unique}")
                    await retrieve_batch(unique)

            # Retrieve for all queries (intent + accumulated keywords)
            await retrieve_batch(queries)

            # FK-graph one-hop expansion
            if candidate_map:
                await expand_fk_neighbors()

            # Merge with any previously retrieved tables from state
            existing = state.get("retrieved_tables", [])
            for entry in existing:
                name = entry.table_schema.name
                if name not in candidate_map:
                    candidate_map[name] = entry
                    hit_counts[name] = 1
                    score_sums[name] = entry.score

            if not candidate_map:
                return {
                    "retrieved_tables": [],
                    "execution_error": "No candidate tables found in the vector store.",
                }

            def composite_score(name: str) -> float:
                cnt = hit_counts.get(name, 1)
                avg = score_sums.get(name, 0.0) / cnt if cnt else 0.0
                return cnt * avg

            sorted_tables = sorted(
                candidate_map.values(),
                key=lambda x: composite_score(x.table_schema.name),
                reverse=True,
            )

            print(
                "📦 Retrieved tables:",
                [t.table_schema.name for t in sorted_tables],
            )

            return {"retrieved_tables": sorted_tables, "execution_error": None}

        except Exception as ex:
            print(f"❌ Retrieval Failed: {ex}")
            return {"retrieved_tables": [], "execution_error": str(ex)}

    def schema_validator_node(self, state: AgentState) -> dict:
        """
        LangGraph Node: LLM validates whether currently retrieved tables
        are sufficient. If not, returns search keywords so the graph loops
        back to retrieve_schema for another pass.
        """
        retrieved_tables = state.get("retrieved_tables", [])
        user_query = state["user_query"]

        if not retrieved_tables:
            return {"execution_error": "No tables to validate.", "schema_validated": False}

        print(f"\n--- [NODE] Validating {len(retrieved_tables)} retrieved tables ---")

        candidate_summary = self._build_candidate_summary(retrieved_tables, user_query)

        system_prompt = (
            "You are an expert database engineer.\n"
            "Given a user question and candidate database tables, "
            "determine if these tables ARE SUFFICIENT to answer the question.\n\n"
            "1. In `relevant_table_names`, list the names of tables (from the candidates) "
            "that are strictly required. Return an empty list if ALL candidates are needed.\n"
            "2. If ALL necessary tables are among the candidates, "
            "return an EMPTY list for `additional_search_relevant_table_keywords`.\n"
            "3. If an important table is MISSING, add short noun-phrase search keywords "
            "(2–6 words) to `additional_search_relevant_table_keywords` so the system "
            "can retrieve it.\n\n"
            "Consider:\n"
            "- Do you have all tables for SELECT columns?\n"
            "- Do you have all tables for JOIN paths (incl. bridge/junction tables)?\n"
            "- Do you have all tables for WHERE / GROUP BY / HAVING filters?\n"
            "- Follow foreign key chains — include referenced and referencing tables."
        )

        assessment = self._response_struct_agent.invoke(
            schema_cls=TableRelevanceAssessment,
            llm=state["llm"],
            prompts=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=candidate_summary),
            ],
            max_retry=2,
        )

        if assessment.is_failure:
            return {"execution_error": assessment.error.message, "schema_validated": False}

        result = assessment.data
        needs_more = bool(result.additional_search_relevant_table_keywords)

        if needs_more:
            print(
                "🔄 Tables insufficient, requesting more keywords:",
                result.additional_search_relevant_table_keywords,
            )
            return {
                "schema_validated": False,
                "additional_search_keywords": result.additional_search_relevant_table_keywords,
                "schema_repair_retries": state.get("schema_repair_retries", 0) + 1,
                "execution_error": None,
            }

        # Confident — filter to relevant tables + auto-include FK bridge tables
        print("✅ Tables validated as sufficient.")
        selected_names = (
            list(dict.fromkeys(result.relevant_table_names))
            if result.relevant_table_names
            else [t.table_schema.name for t in retrieved_tables]
        )

        candidate_map = {t.table_schema.name: t for t in retrieved_tables}
        bridge_additions = []
        for name in list(selected_names):
            schema = candidate_map.get(name)
            if schema is None:
                continue
            for fk in schema.table_schema.foreign_keys:
                target = fk.referred_table
                if target in candidate_map and target not in selected_names:
                    bridge_additions.append(target)

        if bridge_additions:
            unique_bridges = list(dict.fromkeys(bridge_additions))
            print(f"🌉 Auto-adding FK bridge tables: {unique_bridges}")
            selected_names.extend(unique_bridges)

        final_tables = [candidate_map[n] for n in selected_names if n in candidate_map]

        print("✅ Final validated tables:", [t.table_schema.name for t in final_tables])

        return {
            "retrieved_tables": final_tables,
            "schema_validated": True,
            "additional_search_keywords": [],
            "execution_error": None,
        }

    def enhance_retrieved_schema_node(self, state: AgentState) -> dict:
        """
        LangGraph Node: Zero live inspection overhead. Parses the pre-cached 
        TableSchema definitions directly into a dense semantic markdown layout for the LLM.
        """
        # Note: state["retrieved_tables"] now contains list[TableSchema] instead of pure strings
        schemas: list[SchemaRetrieverResult] = state.get("retrieved_tables", [])
        db: DatabaseHelper = state["db_connection"]
        db_type = state.get("database_type", "generic").lower()
        
        print(f"\n--- [NODE] Serializing Cached Meta-Schema to Prompt Context ({db_type.upper()}) ---")
        
        if not schemas:
            return {"schema_context": "# Zero verified schema matches found in registry."}
            
        compiled_schema_context = []
        
        try:
            for get_schema_result in schemas:
                schema = get_schema_result.table_schema
                # 1. Structure structural attributes directly out of the TableSchema object
                schema_str = f"\n\nTable: {schema.name}\n"
                
                # Contextual description documentation if present
                if schema.prose:
                    schema_str += f"Description Context: {' '.join(schema.prose)}\n"
                    
                schema_str += "Columns:\n"
                for col in schema.columns:
                    pk_marker = " [PRIMARY KEY]" if col.name in schema.primary_keys else ""
                    nullable_str = " NULL" if col.nullable else " NOT NULL"
                    schema_str += f"  - {col.name} ({col.type}){nullable_str}{pk_marker}\n"
                
                if schema.foreign_keys:
                    schema_str += "Foreign Key Constraints:\n"
                    for fk in schema.foreign_keys:
                        schema_str += f"  - {fk.constrained_columns} references {fk.referred_table}({fk.referred_columns})\n"

                schema_str += f"Referenced by : {self._build_referenced_by_summary(table=schema)} \n"
                
                # 2. Extract quick live sample rows safely using dialect-specific boundaries
                sample_rows_str = "[]"
                try:
                    limit_clause = f"SELECT TOP 3 * FROM {schema.name}" if "mssql" in db_type or "sqlserver" in db_type else f"SELECT * FROM {schema.name} LIMIT 3"
                    cursor_result = db.execute(limit_clause)
                    sample_rows = [dict(row) for row in cursor_result.mappings().all()]
                    sample_rows_str = str(sample_rows)
                except Exception as row_err:
                    # Non-blocking logging fallback if data is empty or locked
                    sample_rows_str = f"Could not pull row instances safely: {row_err}"
                
                table_context = (
                    f"{schema_str}"
                    f"Real Sample Data Rows Examples:\n{sample_rows_str}\n"
                    f"{'-'*40}\n"
                )
                compiled_schema_context.append(table_context)
                
            full_context = "\n".join(compiled_schema_context)
            return {"schema_context": full_context}
            
        except Exception as e:
            print(f"❌ Schema Enhancement Serialization Error: {e}")
            return {"schema_context": f"Failed to assemble structural meta strings: {e}"}

    def sql_planner_node(self, state: AgentState) -> dict:
        """
        LangGraph Node: Reasons strategy constraints using target flavor architecture rules.
        """
        db_type = state.get("database_type", "generic")
        print(f"\n--- [NODE] Generating SQL Strategy Blueprint for Dialect: {db_type.upper()} ---")
        
        system_prompt = (
            f"You are a Senior Data Architect specializing in {db_type} database systems. "
            "Write a logical step-by-step query execution plan to answer the user's inquiry. "
            f"Account for specific {db_type} syntax behaviors, structural constraints, and functionalities. "
            "DO NOT write the raw code block yet."
        )
        user_prompt = (
            f"User Query: {state['user_query']}\n\n"
            f"Database Schema:\n{state['schema_context']}\n\n"
            f"Prior Dialect Errors (if any):\n{state.get('error_history', [])}"
        )
        
        response = state["llm"].invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return {"messages": [AIMessage(content=f"SQL Plan: {remove_think_blocks(response.content)}")]}

    def sql_generator_node(self, state: AgentState) -> dict:
        """
        LangGraph Node: Writes explicit engine code utilizing the exact target engine schema guidelines.
        Enforces a strict Closed-World constraint to eliminate table hallucinations.
        """
        db_type = state.get("database_type", "generic")
        print(f"\n--- [NODE] Compiling Query directly to {db_type.upper()} Code Syntax ---")
        
        # 1. Dynamically extract the exact whitelist of verified table names from the state
        verified_schemas = state.get("retrieved_tables", [])
        allowed_table_names = [schema.table_schema.name for schema in verified_schemas]
        allowed_tables_str = ", ".join([f"'{name}'" for name in allowed_table_names])
        
        last_plan = state["messages"][-1].content if state.get("messages") else ""
        
        # 2. Inject ironclad constraints into the system prompt
        system_prompt = (
            f"You are an expert developer building high-performance {db_type} commands.\n\n"
            
            "CRITICAL SECURITY & VALIDITY CONSTRAINTS:\n"
            f"1. You are strictly allowed to use ONLY these tables: [{allowed_tables_str}].\n"
            "2. DO NOT hallucinate, invent, or assume any other tables exist, even if you think a query requires them.\n"
            "3. Only use columns explicitly defined for each table in the provided Schema Context.\n"
            "4. If the user question cannot be answered using the allowed tables, write a query that safely returns no data or an empty result, but NEVER use an undeclared table.\n\n"
            
            f"Output ONLY valid, executable {db_type} statement scripts syntax. "
            "Ensure identifier quoting conventions match requirements perfectly (e.g., use backticks for MySQL, double-quotes for PostgreSQL).\n"
            "Wrap your output strictly inside a single ```sql ... ``` markdown container block."
        )
        
        # 3. Present the user prompt with clear separation of what is verified
        user_prompt = (
            f"User Question: {state['user_query']}\n\n"
            f"Dialect Infrastructure Target: {db_type}\n\n"
            f"STRICTLY ALLOWED TABLES LIST:\n{allowed_table_names}\n\n"
            f"Table Layout Schema Context:\n{state['schema_context']}\n\n"
            f"Strategy Plan Guidance:\n{last_plan}"
        )
        
        response = state["llm"].invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw_content = remove_think_blocks(response.content)
        
        generated_sql = raw_content
        if "```sql" in raw_content:
            generated_sql = raw_content.split("```sql")[1].split("```")[0].strip()
        elif "```" in raw_content:
            generated_sql = raw_content.split("```")[1].split("```")[0].strip()
            
        print(f"💻 Generated Code:\n{generated_sql}")
        return {"generated_sql": generated_sql}

    def sql_static_validator_node(self, state: AgentState) -> dict:
        sql = state.get("generated_sql", "")
        db_type = state.get("database_type", "generic").lower()
        print(f"\n--- [NODE] Running Static Analysis on {db_type.upper()} command structure ---")
        
        if not sql:
            return {"execution_error": "Static analysis caught empty code output."}
            
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        for word in forbidden_keywords:
            if f" {word} " in f" {sql.upper()} ":
                return {"execution_error": f"Security Rule Violation: Mutating operator '{word}' forbidden."}
                
        # Dialect quote balancing verification wrapper
        quote_char = "`" if "mysql" in db_type else "'"
        if sql.count(quote_char) % 2 != 0 or sql.count("(") != sql.count(")"):
            return {"execution_error": f"Dialect Syntax Violation: Unbalanced quotes ({quote_char}) or parameters context encountered."}
            
        return {"execution_error": None}

    def sql_repair_node(self, state: AgentState) -> dict:
        db_type = state.get("database_type", "generic")
        print(f"\n--- [NODE] Repairing Static Dialect Breakers for {db_type.upper()} ---")
        
        system_prompt = (
            f"You are an automated code fixing program for {db_type}. Look at the script "
            "and structural linter complaint, then output a completely cleaned expression block "
            "inside a ```sql block container code format."
        )
        user_prompt = (
            f"Original SQL Input Script:\n{state.get('generated_sql')}\n\n"
            f"Linter Feedback Error Trace:\n{state.get('execution_error')}"
        )
        
        response = state["llm"].invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw_content = remove_think_blocks(response.content)
        repaired_sql = raw_content.split("```sql")[1].split("```")[0].strip() if "```sql" in raw_content else raw_content.strip()
        
        return {
            "generated_sql": repaired_sql,
            "static_repair_retries": state.get("static_repair_retries", 0) + 1,
            "retry_count": state.get("static_repair_retries", 0) + 1,  # keep legacy in sync
            "error_history": state.get("error_history", []) + [state.get("execution_error", "")]
        }

    def database_validation_node(self, state: AgentState) -> dict:
        """
        LangGraph Node: Performs a safe pre-compile dry-run validation using
        dialect-appropriate syntax structure.
        """
        sql = state.get("generated_sql", "")
        db: DatabaseHelper = state["db_connection"]
        db_type = state.get("database_type", "generic").lower()
        print(f"\n--- [NODE] Performing Engine Parsing Check on {db_type.upper()} ---")
        
        try:
            # Avoid using EXPLAIN if targeted to MSSQL/SQL Server since syntax differs
            if "mssql" in db_type or "sqlserver" in db_type:
                # Use SET SHOWPLAN_ALL to test parsing without payload execution on MSSQL
                db.execute("SET SHOWPLAN_ALL ON")
                try:
                    db.execute(sql)
                finally:
                    db.execute("SET SHOWPLAN_ALL OFF")
            else:
                db.execute(f"EXPLAIN {sql}")
                
            print(f"✅ Live Server Pre-compile Validation Success.")
            return {"execution_error": None}
        except Exception as engine_exception:
            print(f"⚠️  Live Server Parsing Block Failure: {engine_exception}")
            return {"execution_error": str(engine_exception)}

    def db_error_repair_node(self, state: AgentState) -> dict:
        db_type = state.get("database_type", "generic")
        print(f"\n--- [NODE] Self-Healing Dialect Error Logic ({db_type.upper()}) ---")
        
        system_prompt = (
            f"You are an elite database diagnostic technician specialized in {db_type}. "
            "The statement failed execution on the live server environment instance. "
            f"Fix the command block leveraging deep structural understanding of {db_type} system engine logic. "
            "Provide your final executable solution output in a standard markdown ```sql code block wrapper."
        )
        user_prompt = (
            f"Target System Engine: {db_type}\n\n"
            f"Database Columns Structure:\n{state['schema_context']}\n\n"
            f"Faulty Executed Script Query Line:\n{state['generated_sql']}\n\n"
            f"Live Server Exception Log Trace:\n{state['execution_error']}"
        )
        
        response = state["llm"].invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw_content = remove_think_blocks(response.content)
        repaired_sql = raw_content.split("```sql")[1].split("```")[0].strip() if "```sql" in raw_content else raw_content.strip()
        
        return {
            "generated_sql": repaired_sql,
            "db_repair_retries": state.get("db_repair_retries", 0) + 1,
            "execute_repair_retries": state.get("execute_repair_retries", 0) + 1,
            "retry_count": state.get("db_repair_retries", 0) + 1,  # keep legacy in sync
            "error_history": state.get("error_history", []) + [state.get("execution_error", "")]
        }

    def query_optimizer_node(self, state: AgentState) -> dict:
        return {}

    def execute_sql_node(self, state: AgentState) -> dict:
        sql = state.get("generated_sql", "")
        db: DatabaseHelper = state["db_connection"]
        db_type = state.get("database_type", "generic").lower()
        print(f"\n--- [NODE] Executing Production Payload against {db_type.upper()} ---")
        
        try:
            safe_query = sql.rstrip("; ")
            # Ensure safe execution limits if the user query forgot it
            if "LIMIT" not in safe_query.upper() and "TOP" not in safe_query.upper():
                if "mssql" in db_type or "sqlserver" in db_type:
                    # If it starts with SELECT, append TOP cleanly if not already parsed
                    if safe_query.upper().startswith("SELECT"):
                        safe_query = safe_query.replace("SELECT", "SELECT TOP 100", 1)
                else:
                    safe_query = f"{safe_query} LIMIT 100"
                    
            cursor_result = db.execute(safe_query)
            results = [dict(row) for row in cursor_result.mappings().all()]
            return {"messages": [AIMessage(content=f"Execution Result: {str(results)}")]}
        except Exception as e:
            print(f"❌ Execution Critical Crash: {e}")
            return {"execution_error": str(e)}

    def result_formatter_node(self, state: AgentState) -> dict:
        print(f"\n--- [NODE] Formatting Output presentation results UI ---")
        raw_data = state["messages"][-1].content if state.get("messages") else "No data returned."
        
        system_prompt = (
            "You are a clean business intelligence presenter. Summarize the raw relational database "
            "records into an easy-to-read, conversational response answering the user's explicit question. "
            "Present tabular metrics using markdown data tables when appropriate."
        )
        user_prompt = (
            f"Original Query Intent: {state['user_query']}\n\n"
            f"Executed Query Script Code:\n```sql\n{state.get('generated_sql')}\n```\n\n"
            f"Raw Database Engine Records JSON Data Structure:\n{raw_data}"
        )
        
        response = state["llm"].invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return {"messages": [AIMessage(content=remove_think_blocks(response.content))]}