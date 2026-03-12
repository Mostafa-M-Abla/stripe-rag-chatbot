"""Evaluation runner: RAGAS LLM-as-judge metrics via LangSmith Datasets & Experiments."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langsmith import Client as LangSmithClient
from langsmith import aevaluate
from langsmith.schemas import Example, Run
from langsmith.evaluation import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.dataset_schema import SingleTurnSample

from stripe_rag.config import Settings
from stripe_rag.evaluation.eval_set import load_eval_set
from stripe_rag.generation.generator import AnswerGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvalSummary:
    total: int
    avg_faithfulness: float
    avg_relevancy: float
    avg_context_recall: float
    avg_factual_correctness: float
    avg_latency_ms: float
    experiment_name: str = field(default="")


def _sync_dataset(client: LangSmithClient, eval_set: list, dataset_name: str) -> None:
    """Create or update the LangSmith dataset with eval examples."""
    # Check if dataset already exists
    existing_questions: set[str] = set()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        for example in client.list_examples(dataset_id=dataset.id):
            q = (example.inputs or {}).get("question", "")
            if q:
                existing_questions.add(q)
        logger.info("Dataset '%s' exists with %d examples.", dataset_name, len(existing_questions))
    except Exception:
        # Dataset doesn't exist — create it
        client.create_dataset(dataset_name=dataset_name, description="Stripe RAG eval set")
        logger.info("Created dataset '%s'.", dataset_name)

    # Upload only new examples
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
    """Return async target function for aevaluate."""

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


def _make_ragas_evaluator(metric, llm, embeddings):
    """Return async per-item evaluator closure for a single RAGAS metric."""
    metric.llm = llm
    metric.embeddings = embeddings

    async def evaluator(run: Run, example: Example) -> EvaluationResult:
        outputs = run.outputs or {}
        sample = SingleTurnSample(
            user_input=(example.inputs or {}).get("question", ""),
            response=outputs.get("answer", ""),
            retrieved_contexts=outputs.get("retrieved_contexts", []),
            reference=(example.outputs or {}).get("reference_answer", ""),
        )
        try:
            score = await metric.single_turn_ascore(sample)
        except Exception:
            logger.exception("RAGAS metric '%s' failed for one sample.", metric.name)
            score = None
        return EvaluationResult(key=metric.name, score=score)

    return evaluator


async def _latency_evaluator(run: Run, example: Example) -> EvaluationResult:
    return EvaluationResult(
        key="latency_ms",
        score=(run.outputs or {}).get("latency_ms", 0.0),
    )


async def run_evaluation(settings: Settings) -> EvalSummary:
    """Run the full evaluation suite and return RAGAS metrics."""
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    eval_set = load_eval_set()
    generator = AnswerGenerator(settings)

    logger.info("Running evaluation on %d questions…", len(eval_set))

    _eval_model = settings.eval_llm_model
    _bypass_temp = _eval_model.startswith("gpt-5") or _eval_model.startswith("o")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=_eval_model),
        bypass_temperature=_bypass_temp,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    )

    m_faithfulness = Faithfulness()
    m_relevancy = ResponseRelevancy()
    m_context_recall = LLMContextRecall()
    m_factual = FactualCorrectness()

    ls_client = LangSmithClient(api_key=settings.langsmith_api_key)
    _sync_dataset(ls_client, eval_set, "stripe-rag-eval")

    experiment_results = await aevaluate(
        _make_target(generator),
        data="stripe-rag-eval",
        evaluators=[
            _make_ragas_evaluator(m_faithfulness, evaluator_llm, evaluator_embeddings),
            _make_ragas_evaluator(m_relevancy, evaluator_llm, evaluator_embeddings),
            _make_ragas_evaluator(m_context_recall, evaluator_llm, evaluator_embeddings),
            _make_ragas_evaluator(m_factual, evaluator_llm, evaluator_embeddings),
            _latency_evaluator,
        ],
        experiment_prefix="stripe-rag",
        max_concurrency=4,
        client=ls_client,
        metadata={
            "llm_model": settings.llm_model,
            "embedding_model": settings.openai_embedding_model,
            "retrieval_top_k": settings.retrieval_final_top_k,
        },
    )

    # Aggregate scores by iterating async results
    scores: dict[str, list[float]] = {
        m_faithfulness.name: [],
        m_relevancy.name: [],
        m_context_recall.name: [],
        m_factual.name: [],
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
        avg_faithfulness=_mean(scores[m_faithfulness.name]),
        avg_relevancy=_mean(scores[m_relevancy.name]),
        avg_context_recall=_mean(scores[m_context_recall.name]),
        avg_factual_correctness=_mean(scores[m_factual.name]),
        avg_latency_ms=_mean(scores["latency_ms"]),
        experiment_name=experiment_name,
    )

    logger.info(
        "\nEvaluation complete:\n"
        "  faithfulness         : %.2f\n"
        "  response_relevancy   : %.2f\n"
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
