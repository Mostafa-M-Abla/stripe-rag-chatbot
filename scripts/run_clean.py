#!/usr/bin/env python3
"""
Clean crawled markdown files → data/markdown_clean/
Also produces data/documents_clean.jsonl with updated markdown fields.

Original data/markdown/ and data/documents.jsonl are NOT modified.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stripe_rag.config import get_settings
from stripe_rag.ingestion.cleaner import clean_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()

    clean_dir = settings.data_dir / "markdown_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean_jsonl = settings.data_dir / "documents_clean.jsonl"

    md_files = sorted(settings.markdown_dir.glob("*.md"))
    if not md_files:
        logger.error(f"No .md files found in {settings.markdown_dir}")
        sys.exit(1)

    logger.info(f"Cleaning {len(md_files)} markdown files → {clean_dir}")

    # ── Pass 1: clean individual .md files ───────────────────────────────────
    stats = {"total": 0, "chars_before": 0, "chars_after": 0}

    for md_path in md_files:
        raw = md_path.read_text(encoding="utf-8")
        cleaned = clean_markdown(raw)

        out_path = clean_dir / md_path.name
        out_path.write_text(cleaned, encoding="utf-8")

        stats["total"] += 1
        stats["chars_before"] += len(raw)
        stats["chars_after"] += len(cleaned)

    reduction = 1 - stats["chars_after"] / max(stats["chars_before"], 1)
    logger.info(
        f"Done: {stats['total']} files | "
        f"{stats['chars_before']:,} → {stats['chars_after']:,} chars "
        f"({reduction:.1%} reduction)"
    )

    # ── Pass 2: rebuild documents_clean.jsonl ────────────────────────────────
    if not settings.documents_jsonl.exists():
        logger.warning("documents.jsonl not found — skipping JSONL update")
        return

    written = 0
    with (
        settings.documents_jsonl.open(encoding="utf-8") as fin,
        clean_jsonl.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            doc = json.loads(line)
            slug = (
                doc["url"]
                .replace("https://", "")
                .replace("http://", "")
            )
            import re
            slug = re.sub(r"[/.]", "_", slug)[:200]
            clean_md_path = clean_dir / f"{slug}.md"
            if clean_md_path.exists():
                doc["markdown"] = clean_md_path.read_text(encoding="utf-8")
                doc["word_count"] = len(doc["markdown"].split())
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    logger.info(f"Wrote {written} records → {clean_jsonl}")


if __name__ == "__main__":
    main()