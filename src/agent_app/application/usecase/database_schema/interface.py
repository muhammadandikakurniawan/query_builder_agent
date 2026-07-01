from abc import ABC, abstractmethod

from agent_app.application.dto.database_schema import BuildQueryRequest, BuildQueryResponse, DatabaseConnection, SearchTableSchemaRequest
from agent_app.domain.entities.database_schema import TableSchema
from agent_app.shared.model.response import Result

class IDatabaseSchemaUsecase(ABC):

    @abstractmethod
    async def sync_schema(self, request: DatabaseConnection) -> Result[DatabaseConnection]:
        pass

    @abstractmethod
    def search_table(self, request: SearchTableSchemaRequest) -> Result[list[TableSchema]]:
        pass

    @abstractmethod
    async def build_query(self, request: BuildQueryRequest) -> Result[BuildQueryResponse]:
        pass