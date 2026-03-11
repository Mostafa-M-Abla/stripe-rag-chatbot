#!/usr/bin/env python3
"""
Interactive RAG test — ask any question and see retrieved chunks + generated answer.

Usage:
    python scripts/ask.py "How do I create a PaymentIntent?"
    python scripts/ask.py "What is metered billing?" --section billing
    python scripts/ask.py   # prompts for a question interactively
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from stripe_rag.config import get_settings
from stripe_rag.generation.generator import AnswerGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask a question to the Stripe RAG chatbot")
    p.add_argument("question", nargs="?", help="Question to ask (prompted if omitted)")
    p.add_argument(
        "--section",
        choices=["payments", "billing", "connect", "api", "webhooks"],
        default=None,
        help="Restrict retrieval to a specific Stripe doc section",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    question = args.question
    if not question:
        question = input("Question: ").strip()
        if not question:
            print("No question provided.")
            sys.exit(1)

    settings = get_settings()
    # Override top_k if specified
    settings_dict = settings.model_dump()
    settings_dict["retrieval_final_top_k"] = args.top_k

    print(f"\n{'=' * 70}")
    print(f"  QUESTION: {question}")
    if args.section:
        print(f"  SECTION : {args.section}")
    print(f"{'=' * 70}\n")

    gen = AnswerGenerator(settings)
    # Patch top_k on the settings object
    gen._settings.model_config  # ensure initialised
    object.__setattr__(gen._settings, "retrieval_final_top_k", args.top_k)

    start = time.perf_counter()
    answer_text, chunks = await gen.answer(question, section_filter=args.section)
    elapsed = time.perf_counter() - start

    # ── Retrieved chunks ───────────────────────────────────────────────────
    print(f"RETRIEVED CHUNKS ({len(chunks)})")
    print("-" * 70)
    for i, c in enumerate(chunks, 1):
        print(f"[Source {i}]  score={c.score:.4f}")
        print(f"  Title   : {c.page_title}")
        print(f"  Section : {c.heading_path}")
        print(f"  URL     : {c.source_url}")
        print(f"  Tokens  : {c.token_count}")
        print(f"  Content : {c.content[:300].strip()}{'…' if len(c.content) > 300 else ''}")
        print()

    # ── Generated answer ───────────────────────────────────────────────────
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer_text)
    print(f"\n[latency: {elapsed * 1000:.0f}ms]")


if __name__ == "__main__":
    asyncio.run(main())
