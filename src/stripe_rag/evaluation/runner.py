"""Evaluation runner: measures source_hit_rate and keyword_hit_rate."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langsmith import traceable

from stripe_rag.config import Settings
from stripe_rag.evaluation.eval_set import EVAL_SET, EvalPair
from stripe_rag.generation.generator import AnswerGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    question: str
    answer: str
    source_urls: list[str]
    source_hit: bool   # expected_url_prefix found in any source URL
    keyword_hit: bool  # all keywords found in answer (case-insensitive)
    latency_ms: float


@dataclass
class EvalSummary:
    total: int
    source_hit_rate: float
    keyword_hit_rate: float
    avg_latency_ms: float
    results: list[EvalResult]


def _check_source_hit(expected_prefix: str, source_urls: list[str]) -> bool:
    return any(url.startswith(expected_prefix) for url in source_urls)


def _check_keyword_hit(keywords: list[str], answer: str) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


async def _evaluate_pair(generator: AnswerGenerator, pair: EvalPair) -> EvalResult:
    start = time.perf_counter()
    answer_text, chunks = await generator.answer(pair.question)
    latency_ms = (time.perf_counter() - start) * 1000

    source_urls = [c.source_url for c in chunks]
    return EvalResult(
        question=pair.question,
        answer=answer_text,
        source_urls=source_urls,
        source_hit=_check_source_hit(pair.expected_url_prefix, source_urls),
        keyword_hit=_check_keyword_hit(pair.keywords, answer_text),
        latency_ms=latency_ms,
    )


@traceable(name="run_evaluation")
async def run_evaluation(settings: Settings) -> EvalSummary:
    """Run the full evaluation suite and return aggregated metrics."""
    generator = AnswerGenerator(settings)

    logger.info(f"Running evaluation on {len(EVAL_SET)} questions…")
    results: list[EvalResult] = []

    for i, pair in enumerate(EVAL_SET, start=1):
        logger.info(f"[{i}/{len(EVAL_SET)}] {pair.question[:80]}")
        result = await _evaluate_pair(generator, pair)
        results.append(result)

        status = "✓" if result.source_hit and result.keyword_hit else "✗"
        logger.info(
            f"  {status} source_hit={result.source_hit} keyword_hit={result.keyword_hit} "
            f"latency={result.latency_ms:.0f}ms"
        )

    source_hit_rate = sum(r.source_hit for r in results) / len(results)
    keyword_hit_rate = sum(r.keyword_hit for r in results) / len(results)
    avg_latency_ms = sum(r.latency_ms for r in results) / len(results)

    summary = EvalSummary(
        total=len(results),
        source_hit_rate=source_hit_rate,
        keyword_hit_rate=keyword_hit_rate,
        avg_latency_ms=avg_latency_ms,
        results=results,
    )

    logger.info(
        f"\nEvaluation complete:\n"
        f"  source_hit_rate : {source_hit_rate:.2%}\n"
        f"  keyword_hit_rate: {keyword_hit_rate:.2%}\n"
        f"  avg_latency_ms  : {avg_latency_ms:.0f}ms"
    )
    return summary
