"""Evaluation runner: Custom LLM-as-judge metrics via LangSmith Datasets & Experiments."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from langsmith import Client as LangSmithClient
from langsmith import aevaluate
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from openai import AsyncOpenAI

from stripe_rag.config import Settings
from stripe_rag.evaluation.eval_set import load_eval_set
from stripe_rag.generation.generator import AnswerGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = (
    "You are evaluating a RAG system. Given the CONTEXT (retrieved documents) and the ANSWER "
    "(generated response), list each distinct factual claim made in the ANSWER, then determine "
    "whether each claim is explicitly or implicitly supported by the CONTEXT. "
    "Score = (supported claims) / (total claims). If the answer has no factual claims, score 1.0. "
    'Return JSON only: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
)

_RELEVANCY_SYSTEM = (
    "You are evaluating a RAG system. Given the QUESTION and ANSWER, score how directly and "
    "completely the answer addresses what was asked. Penalize answers that are off-topic, vague, "
    "or answer a different question. Do not penalize for missing information not asked about. "
    'Return JSON only: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
)

_CONTEXT_RECALL_SYSTEM = (
    "You are evaluating a RAG system. Given the REFERENCE ANSWER and the RETRIEVED CONTEXT, "
    "determine what proportion of the key facts and concepts from the reference answer are "
    "present in the context. Score 1.0 if all key facts are in context, 0.0 if none are. "
    'Return JSON only: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
)

_FACTUAL_CORRECTNESS_SYSTEM = (
    "You are evaluating a RAG system. Given the QUESTION, REFERENCE ANSWER, and GENERATED ANSWER, "
    "score the factual accuracy of the generated answer compared to the reference. "
    "Check: are facts, numbers, API names, parameters, and processes correct? "
    "A correct but incomplete answer should still score high. Only penalize for wrong facts. "
    'Return JSON only: {"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}'
)


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

async def _call_judge(client: AsyncOpenAI, model: str, system: str, user: str) -> float | None:
    """Call the LLM judge and return the parsed score, or None on failure."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_completion_tokens=256,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        logger.exception("Judge call failed.")
        return None


async def _judge_faithfulness(
    client: AsyncOpenAI, model: str, answer: str, contexts: list[str]
) -> float | None:
    """Score how well every factual claim in the answer is supported by the retrieved context.

    Score = (supported claims) / (total claims).  Claims not grounded in context
    lower the score; a groundless answer scores 0.
    """
    context_text = "\n\n---\n\n".join(contexts) if contexts else "(none)"
    user = f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}"
    return await _call_judge(client, model, _FAITHFULNESS_SYSTEM, user)


async def _judge_answer_relevancy(
    client: AsyncOpenAI, model: str, question: str, answer: str
) -> float | None:
    """Score how directly and completely the answer addresses the question.

    Penalises off-topic or vague answers; does not penalise for facts not asked about.
    """
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return await _call_judge(client, model, _RELEVANCY_SYSTEM, user)


async def _judge_context_recall(
    client: AsyncOpenAI, model: str, reference_answer: str, contexts: list[str]
) -> float | None:
    """Score the proportion of key facts from the reference answer present in the retrieved context.

    Measures retrieval quality: a score of 1.0 means all reference facts were retrieved.
    """
    context_text = "\n\n---\n\n".join(contexts) if contexts else "(none)"
    user = f"REFERENCE ANSWER:\n{reference_answer}\n\nRETRIEVED CONTEXT:\n{context_text}"
    return await _call_judge(client, model, _CONTEXT_RECALL_SYSTEM, user)


async def _judge_factual_correctness(
    client: AsyncOpenAI, model: str, question: str, reference_answer: str, answer: str
) -> float | None:
    """Score factual accuracy of the generated answer against the reference.

    Checks API names, parameter names, numbers, and processes.  A correct-but-incomplete
    answer still scores high; only wrong facts lower the score.
    """
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"GENERATED ANSWER:\n{answer}"
    )
    return await _call_judge(client, model, _FACTUAL_CORRECTNESS_SYSTEM, user)


# ---------------------------------------------------------------------------
# LangSmith evaluator factories
# ---------------------------------------------------------------------------

def _make_judge_evaluator(metric_name: str, judge_coro):
    """Wrap a judge coroutine into the LangSmith per-item evaluator signature.

    LangSmith ``aevaluate`` expects callables of the form
    ``async (run, example) -> EvaluationResult``.  This factory adapts any judge
    coroutine (which receives ``run`` and ``example``) into that shape and tags the
    result with ``metric_name`` so it appears as a named column in the Experiments UI.
    """

    async def evaluator(run: Run, example: Example) -> EvaluationResult:
        score = await judge_coro(run, example)
        return EvaluationResult(key=metric_name, score=score)

    return evaluator


def _faithfulness_evaluator(oai_client: AsyncOpenAI, model: str):
    async def _judge(run: Run, example: Example) -> float | None:
        outputs = run.outputs or {}
        return await _judge_faithfulness(
            oai_client, model,
            outputs.get("answer", ""),
            outputs.get("retrieved_contexts", []),
        )
    return _make_judge_evaluator("faithfulness", _judge)


def _relevancy_evaluator(oai_client: AsyncOpenAI, model: str):
    async def _judge(run: Run, example: Example) -> float | None:
        outputs = run.outputs or {}
        question = (example.inputs or {}).get("question", "")
        return await _judge_answer_relevancy(oai_client, model, question, outputs.get("answer", ""))
    return _make_judge_evaluator("answer_relevancy", _judge)


def _context_recall_evaluator(oai_client: AsyncOpenAI, model: str):
    async def _judge(run: Run, example: Example) -> float | None:
        outputs = run.outputs or {}
        reference = (example.outputs or {}).get("reference_answer", "")
        return await _judge_context_recall(
            oai_client, model, reference, outputs.get("retrieved_contexts", [])
        )
    return _make_judge_evaluator("context_recall", _judge)


def _factual_correctness_evaluator(oai_client: AsyncOpenAI, model: str):
    async def _judge(run: Run, example: Example) -> float | None:
        outputs = run.outputs or {}
        question = (example.inputs or {}).get("question", "")
        reference = (example.outputs or {}).get("reference_answer", "")
        return await _judge_factual_correctness(
            oai_client, model, question, reference, outputs.get("answer", "")
        )
    return _make_judge_evaluator("factual_correctness", _judge)


async def _latency_evaluator(run: Run, example: Example) -> EvaluationResult:
    return EvaluationResult(
        key="latency_ms",
        score=(run.outputs or {}).get("latency_ms", 0.0),
    )


# ---------------------------------------------------------------------------
# Dataset sync
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    """Aggregated per-metric averages for one evaluation run.

    Fields:
        total: Number of questions evaluated.
        avg_faithfulness: Mean faithfulness score across all questions.
        avg_relevancy: Mean answer-relevancy score.
        avg_context_recall: Mean context-recall score.
        avg_factual_correctness: Mean factual-correctness score.
        avg_latency_ms: Mean end-to-end latency in milliseconds.
        experiment_name: LangSmith experiment name assigned by ``aevaluate``.
    """

    total: int
    avg_faithfulness: float
    avg_relevancy: float
    avg_context_recall: float
    avg_factual_correctness: float
    avg_latency_ms: float
    experiment_name: str = field(default="")


def _sync_dataset(client: LangSmithClient, eval_set: list, dataset_name: str) -> None:
    """Create or update the LangSmith dataset with eval examples."""
    existing_questions: set[str] = set()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        for example in client.list_examples(dataset_id=dataset.id):
            q = (example.inputs or {}).get("question", "")
            if q:
                existing_questions.add(q)
        logger.info("Dataset '%s' exists with %d examples.", dataset_name, len(existing_questions))
    except Exception:
        client.create_dataset(dataset_name=dataset_name, description="Stripe RAG eval set")
        logger.info("Created dataset '%s'.", dataset_name)

    new_items = [item for item in eval_set if item.question not in existing_questions]
    if new_items:
        client.create_examples(
            dataset_name=dataset_name,
            inputs=[{"question": item.question, "section_prefix": item.section_prefix} for item in new_items],
            outputs=[{"reference_answer": item.reference_answer, "source_url": item.source_url} for item in new_items],
        )
        logger.info("Added %d new examples to dataset '%s'.", len(new_items), dataset_name)
    else:
        logger.info("Dataset '%s' is up-to-date.", dataset_name)


def _make_target(generator: AnswerGenerator):
    """Return the async target callable that LangSmith will invoke for each dataset example.

    The target receives the example ``inputs`` dict (``question``, ``section_prefix``) and
    returns an ``outputs`` dict (``answer``, ``retrieved_contexts``, ``source_urls``,
    ``latency_ms``) that the evaluators then score.
    """

    async def target(inputs: dict) -> dict:
        start = time.perf_counter()
        answer_text, chunks = await generator.answer(
            inputs["question"],
            section_filter=inputs.get("section_prefix") or None,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": answer_text,
            "retrieved_contexts": [c.content for c in chunks],
            "source_urls": [c.source_url for c in chunks],
            "latency_ms": latency_ms,
        }

    return target


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_evaluation(settings: Settings) -> EvalSummary:
    """Run the full evaluation suite with custom LLM-as-judge metrics."""
    eval_set = load_eval_set()
    generator = AnswerGenerator(settings)

    logger.info("Running evaluation on %d questions…", len(eval_set))

    oai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    judge_model = settings.eval_llm_model

    ls_client = LangSmithClient(api_key=settings.langsmith_api_key)
    _sync_dataset(ls_client, eval_set, "stripe-rag-eval")

    experiment_results = await aevaluate(
        _make_target(generator),
        data="stripe-rag-eval",
        evaluators=[
            _faithfulness_evaluator(oai_client, judge_model),
            _relevancy_evaluator(oai_client, judge_model),
            _context_recall_evaluator(oai_client, judge_model),
            _factual_correctness_evaluator(oai_client, judge_model),
            _latency_evaluator,
        ],
        experiment_prefix="stripe-rag",
        max_concurrency=1,
        client=ls_client,
        metadata={
            "llm_model": settings.llm_model,
            "embedding_model": settings.openai_embedding_model,
            "retrieval_top_k": settings.retrieval_final_top_k,
        },
    )

    # Aggregate scores
    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "factual_correctness": [],
        "latency_ms": [],
    }
    experiment_name = getattr(experiment_results, "experiment_name", "")
    async for row in experiment_results:
        eval_results = row["evaluation_results"]
        result_list = eval_results["results"] if isinstance(eval_results, dict) else eval_results.results
        for er in result_list:
            if er.key in scores and er.score is not None:
                scores[er.key].append(float(er.score))

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    summary = EvalSummary(
        total=len(eval_set),
        avg_faithfulness=_mean(scores["faithfulness"]),
        avg_relevancy=_mean(scores["answer_relevancy"]),
        avg_context_recall=_mean(scores["context_recall"]),
        avg_factual_correctness=_mean(scores["factual_correctness"]),
        avg_latency_ms=_mean(scores["latency_ms"]),
        experiment_name=experiment_name,
    )

    logger.info(
        "\nEvaluation complete:\n"
        "  faithfulness         : %.2f\n"
        "  answer_relevancy     : %.2f\n"
        "  context_recall       : %.2f\n"
        "  factual_correctness  : %.2f\n"
        "  avg_latency_ms       : %.0fms",
        summary.avg_faithfulness,
        summary.avg_relevancy,
        summary.avg_context_recall,
        summary.avg_factual_correctness,
        summary.avg_latency_ms,
    )
    return summary
