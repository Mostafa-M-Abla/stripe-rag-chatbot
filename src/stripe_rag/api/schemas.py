"""FastAPI request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    section_filter: str | None = Field(
        default=None,
        description="Limit retrieval to a section: payments | billing | connect | api | webhooks",
    )


class SourceRef(BaseModel):
    url: str
    title: str
    heading_path: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    version: str
