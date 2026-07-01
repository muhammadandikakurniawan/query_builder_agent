from typing import Callable, List, Dict, Any, Optional, Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from itertools import product
from agent_app.domain.entities.database_schema import TableSchema
from agent_app.shared.database.helper.database_helper import DatabaseHelper

class IntentExtractionError(Exception):
    pass
class SchemaRetrieverResult(BaseModel):
    table_schema: TableSchema
    score: float

class ExtractIntentResult(BaseModel):
    primary_entities: list[str] = []
    related_entities: list[str] = []
    relationship_keywords: list[str] = []
    filter_attributes: list[str] = []

    def build_retrieval_queries(self) -> list[str]:
        queries: list[str] = []

        # 1. Individual entity lookups
        queries.extend(self.primary_entities)
        queries.extend(self.related_entities)

        # 2. Relationship lookups
        queries.extend(self.relationship_keywords)

        # 3. Entity + filter attribute
        for entity, attr in product(
            self.primary_entities + self.related_entities,
            self.filter_attributes,
        ):
            queries.append(f"{entity} {attr}")

        # 4. Entity relationships
        for entity, related in product(
            self.primary_entities,
            self.related_entities,
        ):
            queries.append(f"{entity} {related}")

        # 5. Relationship + filter
        for relation, attr in product(
            self.relationship_keywords,
            self.filter_attributes,
        ):
            queries.append(f"{relation} {attr}")

        # Remove duplicates while preserving order
        seen = set()
        return [
            q for q in queries
            if q and not (q in seen or seen.add(q))
        ]

SchemaRetriever = Callable[[str], list[SchemaRetrieverResult]]

class TableRelevanceAssessment(BaseModel):
    relevant_table_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names of additional tables that are likely needed to fully answer the user's query. "
            "Use the table names referenced by foreign key relationships or other related tables. "
            "Return an empty list if the current table alone is sufficient. "
            "Only include existing table names, not search phrases or explanations."
        )
    )

    additional_search_relevant_table_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "A list of additional search keywords to retrieve other potentially "
            "relevant tables if the current table is insufficient. "
            "Return an empty list if no further search is needed. "
            "Each keyword should be a short noun phrase (2-6 words), not a full sentence."
        ),
        examples=[
            ["customer orders", "sales transactions", "invoice history"],
            [],
        ],
    )

class AgentState(TypedDict):
    """
    The canonical state schema passing data context between LangGraph nodes.
    """
    llm: ChatOpenAI
    schema_retriever: SchemaRetriever
    db_connection: DatabaseHelper
    database_type: str

    # 1. User Ingress Inputs
    user_query: str

    # 2. RAG & Retrieval Engine State Variables
    extract_intent_result: ExtractIntentResult
    retrieved_tables: list[SchemaRetrieverResult]
    schema_context: str  # Compiled DDL / semantic text given to the Text-to-SQL LLM

    # 3. Code Generation Outputs
    generated_sql: Optional[str]

    # 4. Introspection & Self-Correction Variables
    execution_error: Optional[str]   # Active error propagated between repair nodes
    # Separate counters so repair retries don't consume each other's budgets
    schema_repair_retries: int       # Tracks validate_schema ↔ retrieve_schema loop
    static_repair_retries: int       # Tracks sql_repair ↔ static_validator loop
    db_repair_retries: int           # Tracks db_error_repair ↔ database_validation loop
    execute_repair_retries: int      # Tracks db_error_repair ↔ execute_sql loop
    error_history: list[str]         # Full trace of all prior errors for LLM context
    schema_validated: bool           # Whether LLM confirmed tables are sufficient
    additional_search_keywords: list[str]  # Keywords accumulated across retrieval passes

    # Legacy alias kept for backward-compat; prefer the specific counters above
    retry_count: int

    exception: Optional[Exception]

    # 5. Native LangGraph Message State Tracker
    messages: Annotated[list[BaseMessage], add_messages]
