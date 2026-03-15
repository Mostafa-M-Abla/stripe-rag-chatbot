"""Reranking: CohereReranker (production) and NoOpReranker (fallback)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from langsmith import traceable

from stripe_rag.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for reranking a list of ``RetrievedChunk`` objects."""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Reorder (and optionally filter) ``chunks`` by relevance to ``query``.

        Args:
            query: Original user question.
            chunks: Candidates from the retrieval stage (RRF-fused).
            top_n: Maximum number of chunks to return.

        Returns:
            Reordered list of at most ``top_n`` chunks with updated scores.
        """
        ...


class CohereReranker(BaseReranker):
    """Rerank using Cohere Rerank v3 API."""

    def __init__(self, api_key: str) -> None:
        """Initialise the async Cohere client.

        Args:
            api_key: Cohere API key (set via ``COHERE_API_KEY`` in ``.env``).
        """
        import cohere  # type: ignore[import-untyped]

        self._client = cohere.AsyncClient(api_key=api_key)

    @traceable(name="cohere_rerank")
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Call Cohere rerank API and replace RRF scores with Cohere relevance scores (0–1)."""
        if not chunks:
            return []

        response = await self._client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[c.content for c in chunks],
            top_n=top_n,
        )

        return [
            RetrievedChunk(
                chunk_id=chunks[r.index].chunk_id,
                source_url=chunks[r.index].source_url,
                page_title=chunks[r.index].page_title,
                section_prefix=chunks[r.index].section_prefix,
                heading_path=chunks[r.index].heading_path,
                content=chunks[r.index].content,
                token_count=chunks[r.index].token_count,
                score=r.relevance_score,
            )
            for r in response.results
        ]


class NoOpReranker(BaseReranker):
    """Pass-through reranker: sorts by existing score, truncates to top_n."""

    @traceable(name="noop_rerank")
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Identity rerank: preserves RRF order, returning at most ``top_n`` chunks.

        Used as a zero-dependency fallback when ``COHERE_API_KEY`` is absent.
        """
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]


def get_reranker(cohere_api_key: str | None, cohere_rerank_top_n: int = 5) -> BaseReranker:
    """Return CohereReranker if API key is set, otherwise NoOpReranker."""
    if cohere_api_key:
        logger.info("Using CohereReranker (rerank-english-v3.0)")
        return CohereReranker(api_key=cohere_api_key)
    logger.info("COHERE_API_KEY not set — using NoOpReranker fallback")
    return NoOpReranker()
