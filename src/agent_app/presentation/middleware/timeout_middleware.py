from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from fastapi import FastAPI
from starlette.responses import JSONResponse

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout: int | None = None):
        super().__init__(app)
        self._timeout = timeout

    async def dispatch(self, request, call_next):
        try:
            if self._timeout is None or self._timeout == 0:
                return await call_next(request)

            return await asyncio.wait_for(
                call_next(request),
                timeout=self._timeout,
            )

        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=408,
                content={"error": "Request timed out"},
            )
