from abc import ABC, abstractmethod
from typing import Optional

from agent_app.domain.entities.database_schema import DatabaseConnectionEntity, TableSchema
from agent_app.shared.model.response import Result as ResponseResult

class IDatabaseSchemaRepository(ABC):
    """Abstract interface for Database Connection operations."""

    @abstractmethod
    def save(self, entity: DatabaseConnectionEntity) -> None:
        """Inserts or updates a database connection."""
        pass

    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[DatabaseConnectionEntity]:
        """Retrieves a database connection by ID."""
        pass

    @abstractmethod
    def get_all(self) -> list[DatabaseConnectionEntity]:
        """Retrieves all database connections."""
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> None:
        """Deletes a database connection."""
        pass

    @abstractmethod
    def generate_database_schema(self, db_conn: DatabaseConnectionEntity) -> ResponseResult[list[TableSchema]]:
        """Deletes a database connection."""
        pass

    @abstractmethod
    def sync_schemas_to_vectordb(self,  collection_name: str, db_conn_data: DatabaseConnectionEntity, schemas: list[TableSchema] ) -> None:
        pass

    @abstractmethod
    def retrieve_relevant_tables(self, db: DatabaseConnectionEntity, user_query: str,limit: int = 3) -> list[tuple[dict, float]]:
        pass

    @abstractmethod
    async def generate_database_schema_async(self, db_conn: DatabaseConnectionEntity) -> ResponseResult[list[TableSchema]]:
        pass