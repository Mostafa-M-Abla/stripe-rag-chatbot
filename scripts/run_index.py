#!/usr/bin/env python3
"""CLI: data/ → Qdrant  (stages: chunk | embed | index | all)"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stripe_rag.config import get_settings
from stripe_rag.ingestion.pipeline import chunk_documents, embed_and_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: ``--stage``, ``--dry-run``, ``--no-recreate``."""
    p = argparse.ArgumentParser(description="Index Stripe docs into Qdrant")
    p.add_argument(
        "--stage",
        choices=["chunk", "embed", "index", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    p.add_argument(
        "--no-recreate",
        action="store_true",
        help="Keep existing Qdrant collection instead of dropping and recreating it",
    )
    return p.parse_args()


def run_chunk(settings, dry_run: bool) -> None:
    """Load ``documents.jsonl``, chunk all pages, and write ``chunks.jsonl``."""
    input_jsonl = settings.documents_jsonl
    output_jsonl = settings.chunks_jsonl

    if not input_jsonl.exists():
        logger.error(f"Input not found: {input_jsonl}  (run scripts/run_crawl.py first)")
        sys.exit(1)

    logger.info(f"Chunking: {input_jsonl} → {output_jsonl}")
    if dry_run:
        logger.info("Dry run — skipping.")
        return

    total = chunk_documents(input_jsonl, output_jsonl)
    logger.info(f"Done: {total} chunks written to {output_jsonl}")


async def run_embed_index(settings, dry_run: bool, recreate: bool) -> None:
    """Load ``chunks.jsonl``, embed all chunks, and upsert into Qdrant.

    If ``recreate`` is True (the default), the collection is dropped and recreated first
    to avoid stale duplicate points from previous runs.
    """
    if not settings.chunks_jsonl.exists():
        logger.error(f"Chunks file not found: {settings.chunks_jsonl}  (run --stage chunk first)")
        sys.exit(1)

    action = "Dropping + recreating" if recreate else "Upserting into existing"
    logger.info(f"{action} collection '{settings.qdrant_collection_name}'")
    logger.info(f"Embedding + indexing: {settings.chunks_jsonl}")
    if dry_run:
        logger.info("Dry run — skipping.")
        return

    total = await embed_and_index(settings.chunks_jsonl, settings, recreate=recreate)
    logger.info(f"Done: {total} points indexed in collection '{settings.qdrant_collection_name}'")


def main() -> None:
    """Orchestrate chunk → embed → index stages, honouring ``--stage`` and ``--no-recreate``."""
    args = parse_args()
    settings = get_settings()
    recreate = not args.no_recreate  # default: True (always start clean)

    if args.stage in ("chunk", "all"):
        run_chunk(settings, args.dry_run)

    if args.stage in ("embed", "index", "all"):
        asyncio.run(run_embed_index(settings, args.dry_run, recreate=recreate))


if __name__ == "__main__":
    main()
