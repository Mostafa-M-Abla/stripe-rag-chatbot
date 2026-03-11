"""Qdrant collection creation and batch upsert."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

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

from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 100


class QdrantIndexer:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        dense_embedder: OpenAIEmbedder,
        sparse_embedder: SparseEmbedder,
    ) -> None:
        self._collection_name = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=60)

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
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200, full_scan_threshold=10000),
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
        n_batches = (len(chunks) + _UPSERT_BATCH - 1) // _UPSERT_BATCH
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch_chunks = chunks[i : i + _UPSERT_BATCH]
            batch_dense = dense_vecs[i : i + _UPSERT_BATCH]
            batch_sparse = sparse_vecs[i : i + _UPSERT_BATCH]

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
            batch_num = i // _UPSERT_BATCH + 1
            logger.info(f"Upserted batch {batch_num}/{n_batches}: {total}/{len(chunks)} points")

        return total
