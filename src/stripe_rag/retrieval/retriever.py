"""Hybrid retriever: dense + sparse Qdrant prefetch with RRF fusion."""
from __future__ import annotations

import logging

from langsmith import traceable
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder
from stripe_rag.retrieval.models import RetrievedChunk

try:
    from qdrant_client.models import Fusion
except ImportError:
    # Older client compat
    from qdrant_client.http.models import Fusion  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str | None,
        collection_name: str,
        dense_embedder: OpenAIEmbedder,
        sparse_embedder: SparseEmbedder,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> None:
        self._client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
        self._collection_name = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k

    @traceable(name="hybrid_retrieve")
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        section_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Embed query (dense + sparse), issue Qdrant prefetch query with RRF fusion,
        return top_k RetrievedChunk objects sorted by RRF score.
        """
        import asyncio

        dense_task = asyncio.create_task(self._dense.embed_one(query))
        sparse_vec = self._sparse.embed_one(query)
        dense_vec = await dense_task

        query_filter: Filter | None = None
        if section_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="section_prefix",
                        match=MatchValue(value=section_filter),
                    )
                ]
            )

        results = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=self._dense_top_k,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values()),
                    ),
                    using="sparse",
                    limit=self._sparse_top_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        chunks: list[RetrievedChunk] = []
        for point in results.points:
            p = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(p.get("chunk_id", point.id)),
                    source_url=p.get("source_url", ""),
                    page_title=p.get("page_title", ""),
                    section_prefix=p.get("section_prefix", ""),
                    heading_path=p.get("heading_path", ""),
                    content=p.get("content", ""),
                    token_count=int(p.get("token_count", 0)),
                    score=float(point.score),
                )
            )
        return chunks
