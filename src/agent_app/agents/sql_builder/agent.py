from typing import Literal
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent_app.agents.response_struct.agent import ResponseStructAgent
from agent_app.agents.sql_builder.model import AgentState, SchemaRetriever
from agent_app.agents.sql_builder.nodes import QueryBuilderAgentNodes
from agent_app.shared.database.helper.database_helper import DatabaseHelper

# ---------------------------------------------------------------------------
# Retry budget constants — change here to tune all loops at once
# ---------------------------------------------------------------------------
_MAX_SCHEMA_REPAIRS = 3   # validate_schema ↔ retrieve_schema
_MAX_STATIC_REPAIRS = 3   # sql_repair     ↔ static_validator
_MAX_DB_REPAIRS = 3       # db_error_repair ↔ database_validation
_MAX_EXEC_REPAIRS = 2     # db_error_repair ↔ execute_sql (execution-time errors)


class SqlBuilderAgent:
    def __init__(self, response_struct_agent: ResponseStructAgent):
        self._agent = _build_graph(
            QueryBuilderAgentNodes(response_struct_agent=response_struct_agent)
        )

    async def ainvoke(
        self,
        llm: ChatOpenAI,
        db_helper: DatabaseHelper,
        vector_store_retriever_function: SchemaRetriever,
        user_input: str,
        database_type: str,
    ) -> AgentState:
        initial_state = {
            "llm": llm,
            "schema_retriever": vector_store_retriever_function,
            "db_connection": db_helper,
            "database_type": database_type,
            "user_query": user_input,
            # counters
            "retry_count": 0,
            "schema_repair_retries": 0,
            "static_repair_retries": 0,
            "db_repair_retries": 0,
            "execute_repair_retries": 0,
            "error_history": [],
            "messages": [],
            "schema_validated": False,
            "additional_search_keywords": [],
        }
        result = await self._agent.ainvoke(initial_state)
        return AgentState(**result)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(nodes: QueryBuilderAgentNodes):
    """
    Compile the full Text-to-SQL LangGraph state machine.

    Flow overview
    ─────────────
    START
      → understand_intent          (LLM: extract search keywords)
          ↓ error → abort_with_error → END
      → retrieve_schema            (vector search + FK expansion, accumulates tables)
          ↓ no tables → abort_with_error → END
      → validate_schema            (LLM: are these tables sufficient?)
          ↓ confident             → enhance_schema
          ↓ not confident + retry  → retrieve_schema  (loop, up to _MAX_SCHEMA_REPAIRS)
          ↓ budget exhausted      → abort_with_error → END
      → enhance_schema             (build DDL + sample rows context string)
      → sql_planner                (LLM: step-by-step query plan)
      → sql_generator              (LLM: emit raw SQL)
      → static_validator           (security + syntax checks)
          ↓ error  → sql_repair → static_validator  (up to _MAX_STATIC_REPAIRS)
          ↓ budget → abort_with_error → END
      → database_validation        (EXPLAIN / SHOWPLAN dry-run)
          ↓ error  → db_error_repair → database_validation  (up to _MAX_DB_REPAIRS)
          ↓ budget → abort_with_error → END
      → execute_sql
          ↓ error  → db_error_repair → execute_sql  (up to _MAX_EXEC_REPAIRS)
          ↓ budget → abort_with_error → END
      → result_formatter
      → END
    """
    workflow = StateGraph(AgentState)

    # -----------------------------------------------------------------------
    # Inline abort node — writes a human-readable error into messages so the
    # caller always gets a well-formed final state even on failure.
    # -----------------------------------------------------------------------
    def abort_with_error_node(state: AgentState) -> dict:
        error = state.get("execution_error", "An unknown error occurred.")
        print(f"\n🚨 [ABORT] Pipeline halted: {error}")
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"I was unable to generate a valid SQL query for your request.\n\n"
                        f"**Reason:** {error}\n\n"
                        f"**Error history:**\n"
                        + "\n".join(
                            f"- {e}"
                            for e in state.get("error_history", [])
                            if e
                        )
                    )
                )
            ]
        }

    # -----------------------------------------------------------------------
    # Register nodes
    # -----------------------------------------------------------------------
    workflow.add_node("understand_intent",   nodes.extract_intent_node)
    workflow.add_node("retrieve_schema",     nodes.relevant_schema_retriever_node)
    workflow.add_node("validate_schema",     nodes.schema_validator_node)
    workflow.add_node("enhance_schema",      nodes.enhance_retrieved_schema_node)
    workflow.add_node("sql_planner",         nodes.sql_planner_node)
    workflow.add_node("sql_generator",       nodes.sql_generator_node)
    workflow.add_node("static_validator",    nodes.sql_static_validator_node)
    workflow.add_node("sql_repair",          nodes.sql_repair_node)
    workflow.add_node("database_validation", nodes.database_validation_node)
    workflow.add_node("db_error_repair",     nodes.db_error_repair_node)
    workflow.add_node("execute_sql",         nodes.execute_sql_node)
    workflow.add_node("result_formatter",    nodes.result_formatter_node)
    workflow.add_node("abort_with_error",    abort_with_error_node)

    # -----------------------------------------------------------------------
    # Linear backbone edges
    # -----------------------------------------------------------------------
    workflow.add_edge("enhance_schema",   "sql_planner")
    workflow.add_edge("sql_planner",      "sql_generator")
    workflow.add_edge("sql_generator",    "static_validator")
    workflow.add_edge("sql_repair",       "static_validator")   # repair feeds back
    workflow.add_edge("db_error_repair",  "database_validation") # repair feeds back
    workflow.add_edge("result_formatter", END)
    workflow.add_edge("abort_with_error", END)

    # -----------------------------------------------------------------------
    # Conditional: understand_intent → retrieve_schema | abort
    # -----------------------------------------------------------------------
    def route_after_intent(
        state: AgentState,
    ) -> Literal["retrieve_schema", "abort_with_error"]:
        if state.get("execution_error"):
            return "abort_with_error"
        intent = state.get("extract_intent_result")
        if intent is None or not intent.build_retrieval_queries():
            return "abort_with_error"
        return "retrieve_schema"

    workflow.add_conditional_edges(
        START,
        # understand_intent raises on hard failure, but we also need to handle
        # the case where it returned gracefully with an error flag.
        lambda s: "understand_intent",
        {"understand_intent": "understand_intent"},
    )
    # understand_intent itself; route its output
    workflow.add_conditional_edges(
        "understand_intent",
        route_after_intent,
        {
            "retrieve_schema":   "retrieve_schema",
            "abort_with_error":  "abort_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Conditional: retrieve_schema → validate_schema | abort
    # -----------------------------------------------------------------------
    def route_after_retrieval(
        state: AgentState,
    ) -> Literal["validate_schema", "abort_with_error"]:
        tables = state.get("retrieved_tables", [])
        if not tables or state.get("execution_error"):
            return "abort_with_error"
        return "validate_schema"

    workflow.add_conditional_edges(
        "retrieve_schema",
        route_after_retrieval,
        {
            "validate_schema":  "validate_schema",
            "abort_with_error": "abort_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Conditional: validate_schema → enhance_schema | retrieve_schema | abort
    # -----------------------------------------------------------------------
    def route_after_schema_validation(
        state: AgentState,
    ) -> Literal["enhance_schema", "retrieve_schema", "abort_with_error"]:
        if state.get("execution_error"):
            return "abort_with_error"
        if state.get("schema_validated", False):
            return "enhance_schema"
        retries = state.get("schema_repair_retries", 0)
        if retries >= _MAX_SCHEMA_REPAIRS:
            print(
                f"🚨 Schema validation budget exhausted after {retries} attempts."
            )
            return "abort_with_error"
        print(
            f"🔄 Schema validation #{retries}/{_MAX_SCHEMA_REPAIRS} — re-retrieving with new keywords."
        )
        return "retrieve_schema"

    workflow.add_conditional_edges(
        "validate_schema",
        route_after_schema_validation,
        {
            "enhance_schema":   "enhance_schema",
            "retrieve_schema":  "retrieve_schema",
            "abort_with_error": "abort_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Conditional: static_validator → database_validation | sql_repair | abort
    # -----------------------------------------------------------------------
    def route_static_validation(
        state: AgentState,
    ) -> Literal["database_validation", "sql_repair", "abort_with_error"]:
        error = state.get("execution_error")
        retries = state.get("static_repair_retries", 0)

        if error:
            if retries < _MAX_STATIC_REPAIRS:
                print(
                    f"🔄 Static repair #{retries + 1}/{_MAX_STATIC_REPAIRS}: {error}"
                )
                return "sql_repair"
            print(
                f"🚨 Static repair budget exhausted after {retries} attempts."
            )
            return "abort_with_error"

        return "database_validation"

    workflow.add_conditional_edges(
        "static_validator",
        route_static_validation,
        {
            "sql_repair":        "sql_repair",
            "database_validation": "database_validation",
            "abort_with_error":  "abort_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Conditional: database_validation → execute_sql | db_error_repair | abort
    # -----------------------------------------------------------------------
    def route_database_validation(
        state: AgentState,
    ) -> Literal["execute_sql", "db_error_repair", "abort_with_error"]:
        error = state.get("execution_error")
        retries = state.get("db_repair_retries", 0)

        if error:
            if retries < _MAX_DB_REPAIRS:
                print(
                    f"🔄 DB repair #{retries + 1}/{_MAX_DB_REPAIRS}: {error}"
                )
                return "db_error_repair"
            print(
                f"🚨 DB repair budget exhausted after {retries} attempts."
            )
            return "abort_with_error"

        return "execute_sql"

    workflow.add_conditional_edges(
        "database_validation",
        route_database_validation,
        {
            "db_error_repair":  "db_error_repair",
            "execute_sql":      "execute_sql",
            "abort_with_error": "abort_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Conditional: execute_sql → result_formatter | db_error_repair | abort
    # Execution-time errors (e.g. runtime type mismatch) deserve their own
    # repair budget separate from the EXPLAIN-phase budget.
    # -----------------------------------------------------------------------
    def route_execute_sql(
        state: AgentState,
    ) -> Literal["result_formatter", "db_error_repair", "abort_with_error"]:
        error = state.get("execution_error")
        retries = state.get("execute_repair_retries", 0)

        if error:
            if retries < _MAX_EXEC_REPAIRS:
                print(
                    f"🔄 Execution repair #{retries + 1}/{_MAX_EXEC_REPAIRS}: {error}"
                )
                return "db_error_repair"
            print(
                f"🚨 Execution repair budget exhausted after {retries} attempts."
            )
            return "abort_with_error"

        return "result_formatter"

    workflow.add_conditional_edges(
        "execute_sql",
        route_execute_sql,
        {
            "db_error_repair":  "db_error_repair",
            "result_formatter": "result_formatter",
            "abort_with_error": "abort_with_error",
        },
    )
    
    agent = workflow.compile()
    # png = agent.get_graph().draw_mermaid_png()

    # with open("sql_builder_architecture.png", "wb") as f:
        # f.write(png)

    return agent
