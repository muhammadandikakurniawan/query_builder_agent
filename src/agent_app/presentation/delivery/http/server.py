# infrastructure/http/server.py

import asyncio

import uvicorn

from  fastapi import APIRouter
from  fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference
from agent_app.presentation.delivery.http.handlers.database_schema import register_database_schema_routes
from agent_app.presentation.middleware.timeout_middleware import TimeoutMiddleware
from container.dependency_injection import Container
from shared.model.lifecycle import Lifecycle


class HTTPServer(Lifecycle):

    def __init__(
        self,
        container: Container,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._container = container

        app = FastAPI(
            title="Agent API",
            version="1.0.0",
        )

        cfg : Config = self._container.config()
        app.add_middleware(TimeoutMiddleware, timeout=cfg.app.http_server.api_timeout)  # No timeout

        router = APIRouter()

        # register_chat_routes(
        #     router=router,
        #     usecase=container.chat_usecase(),
        # )

        # register_file_processor_routes(
        #     router=router,
        #     file_processor_usecase=container.file_processor_usecase(),
        # )

        # register_document_schema_routes(
        #     router=router,
        #     document_schema_usecase=container.document_schema_usecase(),
        # )

        # register_sismed_routes(
        #     router=router,
        #     sismed_usecase=container.sismed_usecase(),
        # )

        register_database_schema_routes(
            router=router,
            database_schema_usecase=container.database_schema_usecase(),
        )

     

        @app.get("/api-doc", include_in_schema=False)
        async def scalar_html():
            return get_scalar_api_reference(
                openapi_url=app.openapi_url,
                title="Agent API",
            )

        app.include_router(router)

        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                loop="asyncio",
            )
        )

        self._task = None

    async def start(self):

        self._task = asyncio.create_task(
            self._server.serve(),
        )

    async def stop(self):

        self._server.should_exit = True

        if self._task:
            await self._task
