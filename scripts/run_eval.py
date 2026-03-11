#!/usr/bin/env python3
"""CLI: run evaluation suite against Qdrant + OpenAI."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stripe_rag.config import get_settings
from stripe_rag.evaluation.runner import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    summary = await run_evaluation(settings)

    # Print final summary table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total questions  : {summary.total}")
    print(f"source_hit_rate  : {summary.source_hit_rate:.2%}")
    print(f"keyword_hit_rate : {summary.keyword_hit_rate:.2%}")
    print(f"avg_latency_ms   : {summary.avg_latency_ms:.0f}ms")
    print("=" * 60)

    targets_met = (
        summary.source_hit_rate >= 0.80 and summary.keyword_hit_rate >= 0.75
    )
    if targets_met:
        print("✓ All targets met (source ≥ 80%, keyword ≥ 75%)")
    else:
        print("✗ Targets not yet met")
        if summary.source_hit_rate < 0.80:
            print(f"  source_hit_rate {summary.source_hit_rate:.2%} < 80% target")
        if summary.keyword_hit_rate < 0.75:
            print(f"  keyword_hit_rate {summary.keyword_hit_rate:.2%} < 75% target")

    sys.exit(0 if targets_met else 1)


if __name__ == "__main__":
    asyncio.run(main())
