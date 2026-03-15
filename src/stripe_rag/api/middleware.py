"""Request ID injection and structured JSON request logging."""
from __future__ import annotations

import json
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that attaches a unique ``X-Request-ID`` header and emits structured logs.

    For every request it generates a UUID, injects it into both the request context and
    the response headers, times the downstream call, then prints a JSON log line with
    method, path, status code, latency, and request ID.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Per-request handler: generate UUID, time the call, log on completion."""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        response: Response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        log_record = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "request_id": request_id,
        }
        print(json.dumps(log_record), flush=True)

        return response
