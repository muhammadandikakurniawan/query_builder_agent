import asyncio
import uuid

from langchain_openai import ChatOpenAI

from agent_app.agents.sql_builder.agent import SqlBuilderAgent
from agent_app.agents.sql_builder.model import SchemaRetrieverResult
from agent_app.application.dto.database_schema import BuildQueryRequest, BuildQueryResponse, DatabaseConnection, SearchTableSchemaRequest
from agent_app.application.external.repositories.database_schema import IDatabaseSchemaRepository
from agent_app.application.usecase.database_schema.interface import IDatabaseSchemaUsecase
from agent_app.domain.entities.database_schema import DatabaseConnectionEntity, TableSchema
from agent_app.shared.database.connection.create_connection import create_connection, check_database
from agent_app.shared.database.connection.model import DRIVER_MAP, DatabaseConnectionConfig
from agent_app.shared.database.helper.database_helper import DatabaseHelper
from agent_app.shared.logging.logger import get_logger
from agent_app.shared.model.response import Result


class DatabaseSchemaUsecase(IDatabaseSchemaUsecase):

    def __init__(
        self,
        database_schema_repo: IDatabaseSchemaRepository,
        sql_builder_agent: SqlBuilderAgent
    ):
        self._sql_builder_agent = sql_builder_agent
        self._database_schema_repo = database_schema_repo


    async def sync_schema(self, request: DatabaseConnection) -> Result[DatabaseConnection]:
        try:
            id = request.id or str(uuid.uuid4())
            new_database_connection = DatabaseConnectionEntity(
                id = id,
                database_type=request.database_type,
                host=request.host,
                port=request.port,
                username=request.username,
                password=request.password,
                db_name=request.db_name
            )
            driver_prefix = DRIVER_MAP.get(request.database_type)
            if driver_prefix is None:
                return Result.fail(code="DRIVER_NOTFOUND", message="invalid database type")

            connection = create_connection(config=DatabaseConnectionConfig(
                driver=driver_prefix,
                host=request.host,
                port=request.port,
                username=request.username,
                password=request.password,
                database=request.db_name
            ))
            if not check_database(connection):
                return Result.fail(code="CONNECTING_FAILED", message="failed to connect")
            connection.dispose(close=True)
            
            self._database_schema_repo.save(new_database_connection)

            generate_database_schema_res = await self._database_schema_repo.generate_database_schema_async(new_database_connection)
            if generate_database_schema_res.is_failure:
                return Result.fail(code=generate_database_schema_res.error.code, message=generate_database_schema_res.error.message)
            
            database_schema = generate_database_schema_res.data or []
            
            self._database_schema_repo.sync_schemas_to_vectordb(
                collection_name=new_database_connection.generate_vector_collection_name(),
                db_conn_data=new_database_connection,
                schemas=database_schema
            )
            
            return Result.ok(new_database_connection)
        except Exception as ex:
            get_logger(__name__).exception(str(ex))
            return Result.fail(code="INTERNAL_SERVER_ERROR", message=str(ex))


    def search_table(self, request: SearchTableSchemaRequest) -> Result[list[TableSchema]]:
        database = self._database_schema_repo.get_by_id(request.database_id)
        if database is None:
            return Result.fail(code="DATABASE_NOTFOUND", message="invalid database")

        tables = self._database_schema_repo.retrieve_relevant_tables(db=database, user_query=request.query, limit=request.limit)
        return Result.ok([table for table, score in tables])

    async def build_query(self, request: BuildQueryRequest) -> Result[BuildQueryResponse]:

        database = self._database_schema_repo.get_by_id(request.database_id)
        if database is None:
            return Result.fail(code="DATABASE_NOTFOUND", message="invalid database")

        def vector_store_retriever_function(query: str) -> list[SchemaRetrieverResult]:
            list_document = self._database_schema_repo.retrieve_relevant_tables(db=database, user_query=query, limit=10)
            result = [SchemaRetrieverResult(table_schema=TableSchema.model_validate(table), score=score) for table, score in list_document]
            return result


        llm = ChatOpenAI(
            model=request.query_builder_llm.model,
            api_key=request.query_builder_llm.api_key,
            base_url=request.query_builder_llm.base_url,
            temperature=0,
            max_tokens=1000
        )

        db_driver = DRIVER_MAP[database.database_type]
        database_helper_instance = DatabaseHelper(config=DatabaseConnectionConfig(
            driver=db_driver,
            host=database.host,
            port=database.port,
            username=database.username,
            password=database.password,
            database=database.db_name
        ))

        final_output_state = await self._sql_builder_agent.ainvoke(
            llm=llm,
            db_helper=database_helper_instance,
            vector_store_retriever_function=vector_store_retriever_function,
            user_input=request.user_input,
            database_type=database.database_type
        )
        
        return Result.ok(BuildQueryResponse(
            sample_data=final_output_state["messages"][-1].content,
            generated_query=final_output_state.get("generated_sql","")
        ))

        
