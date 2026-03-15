#!/usr/bin/env python3
"""CLI: crawl Stripe docs → data/"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make src importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stripe_rag.config import get_settings
from stripe_rag.crawler.crawler import StripeCrawler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: ``--max-pages``, ``--concurrency``, ``--dry-run``."""
    p = argparse.ArgumentParser(description="Crawl Stripe documentation")
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print seed URLs and exit")
    p.add_argument("--concurrency", type=int, default=None)
    return p.parse_args()


async def main() -> None:
    """Entry point: build settings, instantiate ``StripeCrawler``, run, and log summary."""
    args = parse_args()
    settings = get_settings()

    max_pages = args.max_pages or settings.crawler_max_pages
    concurrency = args.concurrency or settings.crawler_concurrency

    logger.info(f"Seed URLs: {settings.crawler_seed_urls}")
    logger.info(f"Max pages: {max_pages}, concurrency: {concurrency}")

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    crawler = StripeCrawler(
        seed_urls=settings.crawler_seed_urls,
        raw_html_dir=settings.raw_html_dir,
        markdown_dir=settings.markdown_dir,
        documents_jsonl=settings.documents_jsonl,
        max_pages=max_pages,
        concurrency=concurrency,
        delay_seconds=settings.crawler_delay_seconds,
    )

    pages = await crawler.run()
    logger.info(f"Done. Crawled {len(pages)} pages.")


if __name__ == "__main__":
    asyncio.run(main())
