#!/usr/bin/env python3
"""
Phase 4/5 smoke test.
Verifies the Qdrant collection is healthy, then runs one hybrid retrieval
query end-to-end and prints the top results.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stripe_rag.config import get_settings
from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder
from stripe_rag.retrieval.retriever import HybridRetriever


async def main() -> None:
    """Run two checks: (1) Qdrant collection health and (2) a sample hybrid retrieval query.

    Exits with a non-zero status code if either check fails.
    """
    settings = get_settings()

    # ── 1. Collection health check ─────────────────────────────────────────
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    info = await client.get_collection(settings.qdrant_collection_name)

    print("=" * 60)
    print("QDRANT COLLECTION INFO")
    print("=" * 60)
    print(f"Collection  : {settings.qdrant_collection_name}")
    print(f"Points      : {info.points_count}")
    print(f"Status      : {info.status}")

    try:
        dense_cfg = info.config.params.vectors.get("dense")  # type: ignore[union-attr]
        if dense_cfg:
            print(f"Dense dim   : {dense_cfg.size}")
            print(f"Distance    : {dense_cfg.distance}")
    except Exception:
        pass

    assert info.points_count and info.points_count > 0, "❌ No points in collection!"
    print("\n✓ Collection looks healthy")

    # ── 2. Hybrid retrieval smoke test ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL SMOKE TEST")
    print("=" * 60)

    query = "How do I create a PaymentIntent?"
    print(f"Query: {query!r}\n")

    dense = OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
    sparse = SparseEmbedder()
    retriever = HybridRetriever(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        dense_embedder=dense,
        sparse_embedder=sparse,
        dense_top_k=settings.retrieval_dense_top_k,
        sparse_top_k=settings.retrieval_sparse_top_k,
    )

    chunks = await retriever.retrieve(query, top_k=5)

    assert chunks, "❌ Retrieval returned no results!"

    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c.score:.4f}")
        print(f"     {c.page_title} > {c.heading_path}")
        print(f"     {c.source_url}")
        print(f"     {c.content[:120].strip()}…")
        print()

    print("✓ Retrieval working — Phase 4 & 5 verified")


if __name__ == "__main__":
    asyncio.run(main())
