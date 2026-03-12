"""Corpus-grounded eval set loaded from data/eval_set.json."""
from __future__ import annotations

import json
from dataclasses import dataclass

from stripe_rag.config import get_settings


@dataclass
class EvalItem:
    question: str
    reference_answer: str
    source_url: str
    section_prefix: str


def load_eval_set() -> list[EvalItem]:
    path = get_settings().data_dir / "eval_set.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalItem(**item) for item in data]
