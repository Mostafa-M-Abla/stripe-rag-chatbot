#!/usr/bin/env python3
"""CLI: run evaluation suite against Qdrant + OpenAI (custom LLM-as-judge)."""
from __future__ import annotations

import asyncio
import logging
import os
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
    """Bridge pydantic Settings → ``os.environ`` for the LangSmith SDK, run evaluation, and print results.

    The LangSmith SDK reads credentials from ``os.environ`` directly (not from pydantic
    settings), so this function calls ``os.environ.setdefault()`` before invoking
    ``run_evaluation()``.  Exits with code 1 if any metric target is not met.
    """
    settings = get_settings()

    # Ensure LangSmith SDK can find credentials from Settings/.env
    if settings.langsmith_api_key and settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    summary = await run_evaluation(settings)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Custom LLM-as-Judge)")
    print("=" * 60)
    print(f"Total questions      : {summary.total}")
    print(f"faithfulness         : {summary.avg_faithfulness:.2f}")
    print(f"response_relevancy   : {summary.avg_relevancy:.2f}")
    print(f"context_recall       : {summary.avg_context_recall:.2f}")
    print(f"factual_correctness  : {summary.avg_factual_correctness:.2f}")
    print(f"avg_latency_ms       : {summary.avg_latency_ms:.0f}ms")
    print("=" * 60)


    targets_met = (
        summary.avg_faithfulness >= 0.80
        and summary.avg_relevancy >= 0.75
        and summary.avg_context_recall >= 0.75
        and summary.avg_factual_correctness >= 0.75
    )
    if targets_met:
        print("✓ All targets met")
    else:
        print("✗ Targets not yet met")
        if summary.avg_faithfulness < 0.80:
            print(f"  faithfulness {summary.avg_faithfulness:.2f} < 0.80 target")
        if summary.avg_relevancy < 0.75:
            print(f"  response_relevancy {summary.avg_relevancy:.2f} < 0.75 target")
        if summary.avg_context_recall < 0.75:
            print(f"  context_recall {summary.avg_context_recall:.2f} < 0.75 target")
        if summary.avg_factual_correctness < 0.75:
            print(f"  factual_correctness {summary.avg_factual_correctness:.2f} < 0.75 target")

    if settings.langsmith_api_key and settings.langsmith_tracing:
        print(f"\nLangSmith project  : {settings.langsmith_project}")
        if summary.experiment_name:
            print(f"Experiment name    : {summary.experiment_name}")
        print("Datasets & Exps    : https://smith.langchain.com (Datasets tab)")

    sys.exit(0 if targets_met else 1)


if __name__ == "__main__":
    asyncio.run(main())
