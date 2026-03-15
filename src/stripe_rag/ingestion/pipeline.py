"""Ingestion pipeline: documents.jsonl → chunks.jsonl → Qdrant."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from stripe_rag.ingestion.chunker import HeadingAwareChunker
from stripe_rag.ingestion.cleaner import clean_markdown

logger = logging.getLogger(__name__)


def chunk_documents(input_jsonl: Path, output_jsonl: Path, settings=None) -> int:  # type: ignore[no-untyped-def]
    """
    Read documents from input_jsonl, apply markdown cleaning, chunk each page,
    write to output_jsonl. Returns total chunk count.
    """
    if settings is None:
        from stripe_rag.config import get_settings
        settings = get_settings()
    chunker = HeadingAwareChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        min_chunk_tokens=settings.chunk_min_tokens,
    )
    total_chunks = 0
    total_pages = 0

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_jsonl.open(encoding="utf-8") as fin,
        output_jsonl.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            doc = json.loads(line)
            markdown = clean_markdown(doc["markdown"])
            chunks = chunker.chunk_page(
                markdown=markdown,
                source_url=doc["url"],
                page_title=doc["title"],
                section_prefix=doc["section_prefix"],
            )
            for chunk in chunks:
                fout.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                total_chunks += 1
            total_pages += 1

            if total_pages % 100 == 0:
                logger.info(f"Chunked {total_pages} pages → {total_chunks} chunks so far")

    logger.info(f"Chunking complete: {total_pages} pages → {total_chunks} chunks")
    return total_chunks


async def embed_and_index(
    chunks_jsonl: Path, settings, recreate: bool = False  # type: ignore[no-untyped-def]
) -> int:
    """Embed chunks and upsert to Qdrant. Returns point count."""
    from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder
    from stripe_rag.ingestion.indexer import QdrantIndexer

    dense = OpenAIEmbedder(
        settings.openai_api_key,
        settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    sparse = SparseEmbedder()
    indexer = QdrantIndexer(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        dense_embedder=dense,
        sparse_embedder=sparse,
        upsert_batch=settings.indexing_upsert_batch,
        hnsw_m=settings.qdrant_hnsw_m,
        hnsw_ef_construct=settings.qdrant_hnsw_ef_construct,
        hnsw_full_scan_threshold=settings.qdrant_hnsw_full_scan_threshold,
    )
    if recreate:
        await indexer.drop_collection()
    await indexer.ensure_collection()
    return await indexer.upsert_chunks(chunks_jsonl)
