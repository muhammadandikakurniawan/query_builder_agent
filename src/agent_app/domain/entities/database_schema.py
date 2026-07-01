from dataclasses import dataclass, field
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from agent_app.shared.database.helper.database_helper import BaseTable

class DatabaseConnectionEntity(BaseTable):
    __tablename__ = "database_connections"  # custom table name

    id: Mapped[str] = mapped_column(String, primary_key=True)
    database_type: Mapped[str] = mapped_column(String(50))
    host: Mapped[str] = mapped_column(String)
    port: Mapped[int] = mapped_column(Integer)
    db_name: Mapped[str] = mapped_column(String)
    username: Mapped[str] = mapped_column(String)
    password: Mapped[str] = mapped_column(String)

    def generate_vector_collection_name(self) -> str:
        return f"db_schema_{self.id or ""}_{self.db_name}"


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool

@dataclass
class ForeignKeyInfo:
    constrained_columns: List[str]
    referred_table: str
    referred_columns: List[str]


class TableReference(BaseModel):
    """Represents another table referencing this table."""

    reference_table: str
    reference_table_columns: List[ColumnInfo]           # e.g. school_reference
    columns: List[str]             # e.g. ["user_id"]
    referred_columns: List[str]    # e.g. ["id"]
    constraint_name: Optional[str] = None

# table representation
class TableSchema(BaseModel):
    id: Optional[str] = None
    database_connection_id: str
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    # Other tables -> this table
    referenced_by: List[TableReference] = Field(default_factory=list)
    prose: List[str] = field(default_factory=list) #for semantic search rag

