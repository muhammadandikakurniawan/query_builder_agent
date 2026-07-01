import functools
import json
import re
from typing import List, Optional
import uuid
from concurrent.futures import ThreadPoolExecutor
import uuid
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sqlalchemy import Inspector, inspect

from agent_app.application.external.repositories.database_schema import (
    IDatabaseSchemaRepository,
)
from agent_app.domain.entities.database_schema import ColumnInfo, DatabaseConnectionEntity, ForeignKeyInfo, TableReference, TableSchema
from agent_app.infrastructure.repository.db_manager import DatabaseManager
from agent_app.shared.database.connection import create_connection
from agent_app.shared.database.connection.model import DRIVER_MAP, DatabaseConnectionConfig, PoolConfig, RetryConfig
from agent_app.shared.database.helper.database_helper import DatabaseHelper
from agent_app.shared.database.helper.vectordb import BaseVectorDBHelper
from agent_app.shared.logging.logger import get_logger
from agent_app.shared.model.response import Result as ResponseResult
import asyncio
class ChromaIngestionPayload(BaseModel):
    documents: List[Document] = Field(description="List of LangChain Document objects ready for Chroma insertion.")
    ids: List[str|int] = Field(description="Unique vector identifier hashes matching each document respectively.")

class DatabaseSchemaRepository(IDatabaseSchemaRepository):
    """Concrete implementation using the Connection Helper."""

    def __init__(self, db_manager: DatabaseManager, vectordb: BaseVectorDBHelper):
        self._db_manager = db_manager
        self._vectordb = vectordb

    def save(self, entity: DatabaseConnectionEntity) -> None:
        query = """
            INSERT INTO database_connections (
                id,
                database_type,
                host,
                port,
                db_name,
                username,
                password
            )
            VALUES (
                :id,
                :database_type,
                :host,
                :port,
                :db_name,
                :username,
                :password
            )
            ON CONFLICT(id) DO UPDATE SET
                database_type = excluded.database_type,
                host = excluded.host,
                port = excluded.port,
                db_name = excluded.db_name,
                username = excluded.username,
                password = excluded.password;
        """

        params = {
            "id": entity.id,
            "database_type": entity.database_type,
            "host": entity.host,
            "port": entity.port,
            "db_name": entity.db_name,
            "username": entity.username,
            "password": entity.password,
        }

        self._db_manager.execute_master(sql=query, params=params)

    def get_by_id(self, entity_id: str) -> Optional[DatabaseConnectionEntity]:
        query = """
            SELECT
                id,
                database_type,
                host,
                port,
                db_name,
                username,
                password
            FROM database_connections
            WHERE id = :id;
        """

        cursor = self._db_manager.execute_slave(
            sql=query,
            params={"id": entity_id},
        )

        row = cursor.mappings().fetchone()

        if not row:
            return None

        return DatabaseConnectionEntity(
            id=str(row["id"]),
            database_type=row["database_type"],
            host=row["host"],
            port=row["port"],
            db_name=row["db_name"],
            username=row["username"],
            password=row["password"],
        )

    def get_all(self) -> list[DatabaseConnectionEntity]:
        query = """
            SELECT
                id,
                database_type,
                host,
                port,
                db_name,
                username,
                password
            FROM database_connections;
        """

        cursor = self._db_manager.execute_slave(sql=query)
        rows = cursor.mappings().all()

        return [
            DatabaseConnectionEntity(
                id=row["id"],
                database_type=row["database_type"],
                host=row["host"],
                port=row["port"],
                db_name=row["db_name"],
                username=row["username"],
                password=row["password"],
            )
            for row in rows
        ]

    def delete(self, entity_id: str) -> None:
        query = """
            DELETE FROM database_connections
            WHERE id = :id;
        """

        self._db_manager.execute_master(
            sql=query,
            params={"id": entity_id},
        )

    def _populate_references(self, tables: list[TableSchema]) -> None:
        """Populate referenced_by for every table."""

        table_map = {table.name: table for table in tables}

        for source_table in tables:
            for fk in source_table.foreign_keys:
                target = table_map.get(fk.referred_table)
                if target is None:
                    continue

                target.referenced_by.append(
                    TableReference(
                        reference_table=source_table.name,
                        reference_table_columns=source_table.columns,
                        columns=fk.constrained_columns,
                        referred_columns=fk.referred_columns,
                    )
                )

    def _build_table_schema(self, inspector: Inspector, table_name: str, db_conn) -> TableSchema:
        pk_constraint = inspector.get_pk_constraint(table_name)
        primary_keys = pk_constraint.get("constrained_columns", [])

        prose = f"Table Name: {table_name}\n"
        prose += (
            f"Description: Contains structural database info regarding entity "
            f"attributes for {table_name}.\n"
        )

        table_obj = TableSchema(
            id=str(uuid.uuid4()),
            name=table_name,
            primary_keys=primary_keys,
            database_connection_id=db_conn.id,
        )

        prose += (
            f"Primary Keys: {', '.join(primary_keys) if primary_keys else 'None'}\n"
        )

        prose += "Columns:\n"

        for column in inspector.get_columns(table_name):
            col_name = column["name"]

            table_obj.columns.append(
                ColumnInfo(
                    name=col_name,
                    type=str(column["type"]),
                    nullable=column["nullable"],
                )
            )

            nullable = "nullable" if column["nullable"] else "not nullable"

            prose += (
                f"  - Column '{col_name}' "
                f"is type {column['type']} and is {nullable}.\n"
            )

        prose += "Relationships / Foreign Keys:\n"

        for fk in inspector.get_foreign_keys(table_name):
            fk_obj = ForeignKeyInfo(
                constrained_columns=fk["constrained_columns"],
                referred_table=fk["referred_table"],
                referred_columns=fk["referred_columns"],
            )

            table_obj.foreign_keys.append(fk_obj)

            prose += (
                f"  - Connects via columns {fk_obj.constrained_columns} "
                f"to table '{fk_obj.referred_table}' "
                f"on columns {fk_obj.referred_columns}.\n"
            )

        table_obj.prose.append(prose)

        return table_obj

    async def generate_database_schema_async(
        self,
        db_conn: DatabaseConnectionEntity
    ) -> ResponseResult[list[TableSchema]]:

        try:
            driver_prefix = DRIVER_MAP.get(db_conn.database_type)
            if not driver_prefix:
                supported_types = list(DRIVER_MAP.keys()) + ["sqlite"]
                raise ValueError(
                    f"Unsupported database type: '{db_conn.database_type}'. "
                    f"Supported types: {supported_types}"
                )

            # Get table names first (single connection)
            db_helper = DatabaseHelper(
                config=DatabaseConnectionConfig(
                    driver=driver_prefix,
                    host=db_conn.host,
                    port=db_conn.port,
                    username=db_conn.username,
                    password=db_conn.password,
                    database=db_conn.db_name,
                    pool=PoolConfig(),
                    retry=RetryConfig(),
                )
            )

            try:
                inspector = db_helper.inspect()
                table_names = inspector.get_table_names()
            finally:
                db_helper.dispose()

            loop = asyncio.get_running_loop()

            with ThreadPoolExecutor(max_workers=10) as executor:
                tasks = [
                    loop.run_in_executor(
                        executor,
                        self._build_table_schema,
                        inspector,
                        table_name,
                        db_conn,
                    )
                    for table_name in table_names
                ]

                result: list[TableSchema] = await asyncio.gather(*tasks)

            self._populate_references(result)
            return ResponseResult.ok(result)

        except Exception as ex:
            get_logger(__name__).exception(str(ex))
            return ResponseResult.fail(
                code="FAILED_GET_DB_SCHEMA",
                message=str(ex),
            )

    def generate_database_schema(self, db_conn: DatabaseConnectionEntity) -> ResponseResult[list[TableSchema]]:
        db_helper: DatabaseHelper = None
        response = ResponseResult.fail(code="UNPROCESS_REQUEST", message="UNPROCESS_REQUEST")
        try:
            driver_prefix = DRIVER_MAP.get(db_conn.database_type)
            if not driver_prefix:
                supported_types = list(DRIVER_MAP.keys()) + ["sqlite"]
                raise ValueError(f"Unsupported database type: '{db_conn.database_type}'. Supported types: {supported_types}")

            db_helper = DatabaseHelper(config = DatabaseConnectionConfig(
                driver=driver_prefix,
                host=db_conn.host,
                port=db_conn.port,
                username=db_conn.username,
                password=db_conn.password,
                database=db_conn.db_name,
                pool=PoolConfig(),
                retry=RetryConfig()
            ))

            inspector = db_helper.inspect()
            result: list[TableSchema] = []
            for table_name in inspector.get_table_names():
                pk_constraint = inspector.get_pk_constraint(table_name)
                primary_keys = pk_constraint.get("constrained_columns", [])

                prose = f"Table Name: {table_name}\n"
                prose += f"Description: Contains structural database info regarding entity attributes for {table_name}.\n"
                

                # Create Table instance
                table_obj = TableSchema(
                    id=str(uuid.uuid4()),
                    name=table_name,
                    primary_keys=primary_keys,
                    database_connection_id=db_conn.id
                )
                prose += f"Primary Keys: {', '.join(primary_keys) if primary_keys else 'None'}\n"
                
                # Populate Columns
                prose += "Columns:\n"
                for column in inspector.get_columns(table_name):
                    col_name = column["name"]
                    table_obj.columns.append(ColumnInfo(
                        name=col_name,
                        type=str(column["type"]),
                        nullable=column["nullable"]
                    ))
                    nullable_str = "nullable" if column["nullable"] else "not nullable"
                    prose += f"  - Column '{col_name}' is type {column["type"]} and is {nullable_str}.\n"
                    
                # Populate Foreign Keys
                prose += "Relationships / Foreign Keys:\n"
                for fk in inspector.get_foreign_keys(table_name):
                    fk_obj = ForeignKeyInfo(
                        constrained_columns=fk["constrained_columns"],
                        referred_table=fk["referred_table"],
                        referred_columns=fk["referred_columns"]
                    )
                    prose += f"  - Connects via columns {fk_obj.constrained_columns} to table '{fk_obj.referred_table}' on columns {fk_obj.referred_columns}.\n"
                    
                    table_obj.foreign_keys.append(fk_obj)
                
                table_obj.prose.append(prose)
                result.append(table_obj)
            
            response = ResponseResult.ok(result)
        except Exception as ex:
            get_logger(__name__).exception(str(ex))
            response = ResponseResult.fail(code="FAILED_GET_DB_SCHEMA", message=str(ex))
        
        finally:
            if db_helper:
                db_helper.dispose()

            return response

    def _tokenize_name(self, name: str) -> str:
        """Splits snake_case, camelCase, or kebab-case into space-separated words."""
        # Handle camelCase / PascalCase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
        # Handle snake_case / dashes / prefixes like tbl_, mst_
        clean = re.sub(r'^(tbl|mst|ref|fact|dim)[_\-]', '', s2.lower())
        return re.sub(r'[_\-]+', ' ', clean).strip()


    def generate_universal_table_prose(self, db_name: str, database_type: str, schema: TableSchema) -> str:
        """
        Generates a highly adaptive, domain-agnostic semantic summary for any database table.
        Translates raw infrastructure architectures into explicit intent patterns for AI Agents.
        """
        # 1. Semantic Tokenization (Crucial for any Vector Search Engine)
        human_table_name = self._tokenize_name(schema.name)
        
        # Extract keywords from column names to build a natural language semantic cloud
        column_keywords = []
        column_descriptions = []
        
        for col in schema.columns:
            human_col_name = self._tokenize_name(col.name)
            column_keywords.append(human_col_name)
            
            nullable_str = "optional (nullable)" if col.nullable else "required (not null)"
            column_descriptions.append(
                f"  - Field '{col.name}' represents the entity's '{human_col_name}' attribute. "
                f"Type: {col.type}, Constraints: {nullable_str}."
            )
        
        # 2. Compile Structural Metadata Strings
        pk_str = ", ".join(schema.primary_keys) if schema.primary_keys else "None (No primary key defined)"
        cols_str = "\n".join(column_descriptions) if column_descriptions else "  - No structured columns found."
        
        # 3. Dynamic Structural Join Map
        fk_descriptions = []
        connected_tables = []
        for fk in schema.foreign_keys:
            referred_human = self._tokenize_name(fk.referred_table)
            connected_tables.append(referred_human)
            fk_descriptions.append(
                f"  - Can join with target table '{fk.referred_table}' ('{referred_human}') "
                f"by matching local columns {fk.constrained_columns} to foreign columns {fk.referred_columns}."
            )
        fks_str = "\n".join(fk_descriptions) if fk_descriptions else "  - No explicitly defined outbound relationships."

        # 4. Synthesize Semantic Context Cloud
        # This automatically weaves keywords into searchable sentences, adapting to whatever schema is passed
        keyword_cloud = ", ".join(sorted(list(set(column_keywords))))
        relational_context = f" It correlates directly with context matching: {', '.join(connected_tables)}." if connected_tables else ""

        # 5. Build Final LLM-Optimized Textual Block
        universal_prose = (
            f"=== COGNITIVE DATA NODE ===\n"
            f"SOURCE ID: {db_name} | DIALECT: {database_type}\n"
            f"TARGET TABLE: {schema.name}\n\n"
            
            f"CONCEPTUAL SUMMARY:\n"
            f"This structural component manages business rules, entity models, and records tied to '{human_table_name}'.{relational_context}\n"
            f"It contains descriptive properties involving: {keyword_cloud}.\n"
            f"Select this table if the intent requires looking up, filtering, or analyzing records by any of these attributes.\n\n"
            
            f"DATA FIELDS & SCHEMATIC ATTRIBUTES:\n"
            f"Primary Unique Identifier(s): [{pk_str}]\n"
            f"Available structural field specifications:\n{cols_str}\n\n"
            
            f"RELATIONAL GRAPH & SQL JOIN PATHS:\n"
            f"To cross-reference data or perform relational queries, use these mapping paths:\n{fks_str}\n"
            f"=== END OF NODE ==="
        )

        return universal_prose

    def prepare_vector_ingestion_payload(self, table_schemas: List[TableSchema]) -> ChromaIngestionPayload:
        """
        Transforms a list of TableSchema objects into a clean, highly indexable 
        LangChain Document collection paired with unique identifiers.
        """
        documents = []
        ids = []
        
        for table_no, table in enumerate(table_schemas):
            # 1. Generate a predictable, unique deterministic ID
            # unique_id = f"{table.database_connection_id}_{table.name}"
            # unique_id = table.id or str(uuid.uuid4())
            # unique_id = table.name
            unique_id  = table_no
            
            # 2. Compile structural metadata elements into searchable text
            columns_text = "\n".join([
                f"  - Column '{col.name}' is type {col.type} (Nullable: {col.nullable})"
                for col in table.columns
            ])
            
            pks_text = ", ".join(table.primary_keys) if table.primary_keys else "None"
            
            fks_text = ""
            for fk in table.foreign_keys:
                fks_text += f"  - Relational link: {fk.constrained_columns} connects to target table '{fk.referred_table}' on {fk.referred_columns}\n"
            if not fks_text:
                fks_text = "  - No relationships defined."

            # 3. Incorporate your pre-generated semantic 'prose' blocks
            prose_compiled = "\n".join([f"- {p}" for p in table.prose]) if table.prose else "- No conceptual metadata provided."

            # 4. Construct the definitive semantic document body
            document_content = f"""
    Table Name: {table.name}
    ==================================================
    Business Context & Meanings:
    {prose_compiled}

    Database Structural Layout:
    * Primary Keys: {pks_text}
    * Table Attributes / Columns:
    {columns_text}

    Relational Connections / Foreign Keys:
    {fks_text}
    """.strip()
            metadata = table.model_dump(mode="json")
            
            # 6. Instantiate LangChain container
            doc = Document(page_content=document_content, metadata=metadata)
            
            documents.append(doc)
            ids.append(unique_id)
            
        return ChromaIngestionPayload(documents=documents, ids=ids)

    def sync_schemas_to_vectordb(self,  collection_name: str, db_conn_data: DatabaseConnectionEntity, schemas: list[TableSchema] ) -> None:
        self._vectordb.delete_collection(collection_name=collection_name)
        payload = self.prepare_vector_ingestion_payload(schemas)
        self._vectordb.upsert_documents(
            collection_name=collection_name,
            documents=payload.documents,
            ids=payload.ids
        )

    def retrieve_relevant_tables(self, db: DatabaseConnectionEntity, user_query: str, limit: int = 3) -> list[tuple[dict, float]]:
        """
        Finds the exact structural table schemas matching a natural language user request.
        Restricts search scopes explicitly to the provided database connection identifier.
        """

        collection_name = db.generate_vector_collection_name()
        docs = self._vectordb.similarity_search(collection_name=collection_name, query_text=user_query,limit=limit) or []

        result = [(TableSchema.model_validate(doc.metadata), score) for doc, score in docs]
        return result
        