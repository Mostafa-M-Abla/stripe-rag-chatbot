"""FastAPI request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for ``POST /chat`` and ``POST /chat/stream``."""

    question: str = Field(..., min_length=1, max_length=2000)
    section_filter: str | None = Field(
        default=None,
        description="Limit retrieval to a section: payments | billing | connect | api | webhooks",
    )
    session_id: str | None = Field(default=None, description="Omit to start a new session.")


class SourceRef(BaseModel):
    """A single cited source included in the chat response."""

    url: str
    title: str
    heading_path: str
    score: float


class ChatResponse(BaseModel):
    """Response body for the non-streaming ``POST /chat`` endpoint."""

    answer: str
    sources: list[SourceRef]
    latency_ms: float
    model: str
    session_id: str  # always returned so client knows which session to reuse


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    qdrant_connected: bool
    version: str
