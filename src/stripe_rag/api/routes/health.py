"""Health and readiness endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from qdrant_client import AsyncQdrantClient

from stripe_rag.api.schemas import HealthResponse
from stripe_rag.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """``GET /health`` — liveness probe.

    Always returns ``{"status": "ok"}`` plus the app version and whether Qdrant is
    reachable (with a 5 s timeout).  Never returns a non-200 status — use ``/ready``
    for hard dependency checks.
    """
    settings = get_settings()
    qdrant_connected = False
    try:
        client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5
        )
        await client.get_collections()
        qdrant_connected = True
    except Exception:
        pass

    from stripe_rag import __version__

    return HealthResponse(
        status="ok",
        qdrant_connected=qdrant_connected,
        version=__version__,
    )


@router.get("/ready")
async def ready() -> dict:
    """``GET /ready`` — readiness probe.

    Returns 200 ``{"status": "ready"}`` when Qdrant is reachable, or HTTP 503 if the
    ping fails.  Used by load-balancers and container orchestrators to gate traffic.
    """
    settings = get_settings()
    try:
        client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5
        )
        await client.get_collections()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {exc}") from exc
