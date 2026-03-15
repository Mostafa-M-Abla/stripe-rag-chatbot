"""Qdrant collection creation and batch upsert."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 100


class QdrantIndexer:
    """Manages the Qdrant collection lifecycle and bulk upsert of embedded chunks.

    Responsible for: creating / dropping the collection (HNSW config is set once at
    creation), and upserting batches of ``PointStruct`` objects that carry both a
    3072-dim dense vector and a BM42 sparse vector.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        dense_embedder: OpenAIEmbedder,
        sparse_embedder: SparseEmbedder,
        upsert_batch: int = _UPSERT_BATCH,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        hnsw_full_scan_threshold: int = 10_000,
    ) -> None:
        """Initialise the async Qdrant client and attach embedders.

        Args:
            url: Qdrant cluster URL (e.g. ``https://xxx.qdrant.io:6333``).
            api_key: Qdrant API key; ``None`` for unauthenticated local instances.
            collection_name: Name of the collection to create / upsert into.
            dense_embedder: ``OpenAIEmbedder`` instance for dense vector generation.
            sparse_embedder: ``SparseEmbedder`` instance for BM42 sparse vectors.
            upsert_batch: Number of points per upsert call (default ``_UPSERT_BATCH``).
            hnsw_m: HNSW ``m`` parameter controlling graph connectivity.
            hnsw_ef_construct: HNSW ``ef_construct`` parameter for index build quality.
            hnsw_full_scan_threshold: Point count below which Qdrant uses brute-force search.
        """
        self._collection_name = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=60)
        self._upsert_batch = upsert_batch
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._hnsw_full_scan_threshold = hnsw_full_scan_threshold

    async def ensure_collection(self) -> None:
        """Create collection with HNSW + payload indexes if it does not already exist."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}

        if self._collection_name in existing:
            logger.info(f"Collection '{self._collection_name}' already exists — skipping")
            return

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
            hnsw_config=HnswConfigDiff(
                m=self._hnsw_m,
                ef_construct=self._hnsw_ef_construct,
                full_scan_threshold=self._hnsw_full_scan_threshold,
            ),
        )
        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="section_prefix",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._collection_name,
            field_name="source_url",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created collection '{self._collection_name}'")

    async def drop_collection(self) -> None:
        """Delete the collection if it exists."""
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection_name in existing:
            await self._client.delete_collection(self._collection_name)
            logger.info(f"Dropped collection '{self._collection_name}'")
        else:
            logger.info(f"Collection '{self._collection_name}' does not exist — nothing to drop")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    async def _upsert_with_retry(self, points: list[PointStruct]) -> None:
        """Upsert a batch of points with up to 5 retries (4–60 s exponential back-off).

        Guards against Qdrant Cloud cold-start timeouts that would otherwise fail the
        entire indexing run on the first few batches.
        """
        await self._client.upsert(collection_name=self._collection_name, points=points)

    async def upsert_chunks(self, chunks_jsonl: Path) -> int:
        """Read chunks from JSONL, generate embeddings, upsert to Qdrant. Returns point count."""
        chunks: list[dict] = []
        with chunks_jsonl.open(encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        if not chunks:
            logger.warning("No chunks found — nothing to upsert")
            return 0

        logger.info(f"Generating embeddings for {len(chunks)} chunks…")
        contents = [c["content"] for c in chunks]
        dense_vecs = await self._dense.embed_all(contents)
        sparse_vecs = self._sparse.embed_all(contents)

        total = 0
        n_batches = (len(chunks) + self._upsert_batch - 1) // self._upsert_batch
        for i in range(0, len(chunks), self._upsert_batch):
            batch_chunks = chunks[i : i + self._upsert_batch]
            batch_dense = dense_vecs[i : i + self._upsert_batch]
            batch_sparse = sparse_vecs[i : i + self._upsert_batch]

            points = [
                PointStruct(
                    id=c["chunk_id"],
                    vector={
                        "dense": d,
                        "sparse": SparseVector(
                            indices=list(s.keys()),
                            values=list(s.values()),
                        ),
                    },
                    payload={
                        k: c[k]
                        for k in (
                            "chunk_id",
                            "source_url",
                            "page_title",
                            "section_prefix",
                            "heading_path",
                            "content",
                            "token_count",
                            "chunk_index",
                            "total_chunks",
                        )
                    },
                )
                for c, d, s in zip(batch_chunks, batch_dense, batch_sparse, strict=False)
            ]
            await self._upsert_with_retry(points)
            total += len(points)
            batch_num = i // self._upsert_batch + 1
            logger.info(f"Upserted batch {batch_num}/{n_batches}: {total}/{len(chunks)} points")

        return total
