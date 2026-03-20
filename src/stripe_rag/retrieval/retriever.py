"""Hybrid retriever: dense + sparse Qdrant prefetch with RRF fusion."""
from __future__ import annotations

import logging
import time

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
from tenacity import retry, stop_after_attempt, wait_exponential

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
        dense_top_k: int,
        sparse_top_k: int,
        mmr_enabled: bool,
        mmr_lambda: float,
        mmr_top_k: int = 10,
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
            mmr_enabled: Whether to apply MMR diversity selection after RRF fusion.
            mmr_lambda: MMR trade-off weight (0 = max diversity, 1 = max relevance).
            mmr_top_k: Number of diverse chunks to select via MMR before reranking.
        """
        # timeout=30s guards against slow Qdrant Cloud cold-start responses
        self._client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
        self._collection_name = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._mmr_enabled = mmr_enabled
        self._mmr_lambda = mmr_lambda
        self._mmr_top_k = mmr_top_k

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
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
            with_vectors=["dense"] if self._mmr_enabled else False,
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

        # Step 5: optionally apply MMR to promote diversity across the candidate set.
        if self._mmr_enabled:
            vectors: dict[str, list[float]] = {
                str(p.get("chunk_id", point.id)): point.vector["dense"]  # type: ignore[index,misc]
                for point in results.points
                if isinstance(point.vector, dict) and "dense" in point.vector
                for p in [point.payload or {}]
            }
            t0 = time.perf_counter()
            chunks = self._apply_mmr(dense_vec, chunks, vectors, self._mmr_lambda, self._mmr_top_k)
            logger.debug("MMR selected %d chunks in %.2f ms", len(chunks), (time.perf_counter() - t0) * 1000)

        return chunks

    @staticmethod
    def _apply_mmr(
        query_vec: list[float],
        chunks: list[RetrievedChunk],
        vectors: dict[str, list[float]],
        lam: float,
        k: int,
    ) -> list[RetrievedChunk]:
        """Select ``k`` diverse chunks using Maximal Marginal Relevance.

        Iteratively selects the chunk that maximises:
            λ × cosine_sim(chunk, query) − (1−λ) × max_sim(chunk, already_selected)

        Args:
            query_vec: Dense query embedding (used to compute relevance scores).
            chunks: Candidates to select from (must all have entries in ``vectors``).
            vectors: Map from chunk_id → dense vector for each candidate.
            lam: Trade-off weight (0 = max diversity, 1 = max relevance).
            k: Number of chunks to select.

        Returns:
            List of ``k`` (or fewer, if candidates < k) selected chunks in MMR order.
        """
        import numpy as np

        if not chunks:
            return chunks

        q = np.array(query_vec, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-10

        vecs = np.array([vectors[c.chunk_id] for c in chunks], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms + 1e-10

        relevance = vecs @ q  # cosine similarity to query (shape: n,)
        selected: list[int] = []
        remaining = list(range(len(chunks)))

        while remaining:
            if not selected:
                # First pick: pure relevance
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_vecs = vecs[selected]  # (k, d)

                def mmr_score(i: int, _sv: np.ndarray = sel_vecs) -> float:
                    max_sim = float((vecs[i] @ _sv.T).max())
                    return lam * float(relevance[i]) - (1 - lam) * max_sim

                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
            if len(selected) == k:
                break

        return [chunks[i] for i in selected]
