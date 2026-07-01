from typing import Optional

from pydantic import BaseModel, Field


class DatabaseConnection(BaseModel):
    id: Optional[str] = None
    database_type: str = Field(..., max_length=50)
    host: str
    port: int
    db_name: str
    username: str
    password: str

class SearchTableSchemaRequest(BaseModel):
    database_id: str
    query: str
    limit: int


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str

class BuildQueryRequest(BaseModel):
    database_id: str
    user_input: str
    query_builder_llm: LLMConfig

class BuildQueryResponse(BaseModel):
    sample_data: str
    generated_query: str