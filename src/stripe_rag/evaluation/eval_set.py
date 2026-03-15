"""Corpus-grounded eval set loaded from data/eval_set.json."""
from __future__ import annotations

import json
from dataclasses import dataclass

from stripe_rag.config import get_settings


@dataclass
class EvalItem:
    """One evaluation sample loaded from ``data/eval_set.json``.

    Fields:
        question: The natural-language question posed to the RAG system.
        reference_answer: Gold-standard answer used by the LLM judges for
            context-recall and factual-correctness scoring.
        source_url: Canonical URL of the page that should be retrieved (used for
            source hit-rate if that metric is added later).
        section_prefix: Stripe doc section label (e.g. ``"payments"``); passed as
            ``section_filter`` to the retriever to scope the search.
    """

    question: str
    reference_answer: str
    source_url: str
    section_prefix: str


def load_eval_set() -> list[EvalItem]:
    path = get_settings().data_dir / "eval_set.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalItem(**item) for item in data]
