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
    """Hybrid retriever that combines dense and sparse search over Qdrant using RRF fusion.

    Search flow:
      1. Embed the query into a dense vector (OpenAI text-embedding-3-large) and a sparse
         vector (BM42) concurrently.
      2. Issue two Qdrant prefetch branches — one per vector type — each retrieving their
         own top-k candidates.
      3. Qdrant merges the two candidate sets on the server via Reciprocal Rank Fusion (RRF),
         then returns the top-k results after fusion.

    RRF scores (~0.25–0.50) are rank-based, not cosine similarity values.
    """

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
        """Initialise the retriever and create a persistent async Qdrant client.

        Args:
            qdrant_url: URL of the Qdrant cluster (e.g. Qdrant Cloud endpoint).
            qdrant_api_key: API key for authenticated Qdrant Cloud clusters; None for local.
            collection_name: Name of the Qdrant collection to search.
            dense_embedder: OpenAI embedder that produces 3072-dim float vectors.
            sparse_embedder: BM42 embedder that produces token-weight sparse vectors.
            dense_top_k: Number of candidates to retrieve in the dense prefetch branch.
            sparse_top_k: Number of candidates to retrieve in the sparse prefetch branch.
        """
        # timeout=30s guards against slow Qdrant Cloud cold-start responses
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
        """Run a hybrid search and return the top-k chunks after RRF fusion.

        Steps:
          1. Embed the query into dense and sparse vectors concurrently.
          2. Optionally build a Qdrant payload filter to restrict results to a single
             Stripe doc section (e.g. "payments").
          3. Send a prefetch query with two branches (dense + sparse); Qdrant performs
             server-side RRF fusion and returns the top `top_k` results.
          4. Unpack each result point's payload into a RetrievedChunk dataclass.

        Args:
            query: Natural-language question to search for.
            top_k: Final number of chunks to return after RRF fusion. This is the hard
                   limit applied by Qdrant after merging the two prefetch sets — it is
                   NOT the size of either prefetch branch.
            section_filter: If set, restricts search to chunks whose `section_prefix`
                            payload field exactly matches this value.

        Returns:
            List of RetrievedChunk objects sorted by descending RRF score.
        """
        import asyncio

        # Step 1: embed query concurrently — dense is async (OpenAI API call),
        # sparse is sync (local BM42 model), so we overlap them with a task.
        dense_task = asyncio.create_task(self._dense.embed_one(query))
        sparse_vec = self._sparse.embed_one(query)
        dense_vec = await dense_task

        # Step 2: build an optional payload filter to scope results to one doc section.
        # section_prefix is stored as a KEYWORD index in Qdrant for fast exact matching.
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

        # Step 3: issue the hybrid query.
        # Two prefetch branches run independently on Qdrant; the fusion query then
        # merges their ranked result sets using Reciprocal Rank Fusion (RRF) and
        # returns only the top `top_k` points from the combined pool.
        results = await self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                # Dense branch: cosine similarity over OpenAI 3072-dim vectors
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=self._dense_top_k,
                ),
                # Sparse branch: BM42 keyword-weighted vectors (term-frequency style)
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values()),
                    ),
                    using="sparse",
                    limit=self._sparse_top_k,
                ),
            ],
            # RRF fusion: score = Σ 1/(60 + rank) across both branches; no reranking here
            query=FusionQuery(fusion=Fusion.RRF),
            # Final cut: keep only the top_k points from the fused result set
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        # Step 4: unpack Qdrant point payloads into typed RetrievedChunk dataclasses.
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
                    score=float(point.score),  # RRF score (~0.25–0.50)
                )
            )
        return chunks
