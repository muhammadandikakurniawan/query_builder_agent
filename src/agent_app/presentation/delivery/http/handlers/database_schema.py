from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException, status
from pathlib import Path
import tempfile
import os
import uuid
from agent_app.application.dto.database_schema import BuildQueryRequest, DatabaseConnection, SearchTableSchemaRequest
from agent_app.application.usecase.database_schema.interface import IDatabaseSchemaUsecase
from fastapi.concurrency import run_in_threadpool

from agent_app.shared.logging.logger import get_logger
from agent_app.shared.model.response import Result


def register_database_schema_routes(
    router: APIRouter,
    database_schema_usecase: IDatabaseSchemaUsecase,
):

    @router.post(
        "/v1/database-schema/schema-sync",
        tags=["Database Schema"],
        summary="Synchronize database schema",
        description="Synchronize the database schema with the vector database. If the id is empty, a new vector database entry is created. If the id is provided, the existing entry is updated."
    )
    async def sync_schema(
        payload: DatabaseConnection
    ):
        try:
            # Process file
            result = await database_schema_usecase.sync_schema(request=payload)
            if result.is_failure:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": result.error.code or "400",
                        "message": result.error.message or "",
                    },
                )
            return result
        
        except HTTPException:
            raise

        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": str(ex),
                },
            )


    @router.post(
        "/v1/database-schema/build-query",
        tags=["Database Schema"],
        summary="Build query using llm",
    )
    async def build_query(
        payload: BuildQueryRequest
    ):
        try:
            # Process file
            result = await database_schema_usecase.build_query(request=payload)
            if result.is_failure:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": result.error.code or "400",
                        "message": result.error.message or "",
                    },
                )
            return result
        
        except HTTPException as ex:
            get_logger(__name__).exception(str(ex))
            raise

        except Exception as ex:
            get_logger(__name__).exception(str(ex))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": str(ex),
                },
            )

    @router.get(
        "/v1/database-schema/table-schema",
        tags=["Database Schema"],
        summary="Search table schema",
    )
    async def search_table(
        query: SearchTableSchemaRequest = Depends()
    ):
        try:
            # Process file
            result = database_schema_usecase.search_table(request=query)

            return result
        
        except Exception as ex:
            return Result.fail(code="INTERNAL_SERVER_ERROR", message=str(ex))