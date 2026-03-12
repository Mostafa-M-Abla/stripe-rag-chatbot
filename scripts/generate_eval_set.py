#!/usr/bin/env python3
"""One-time script: generate corpus-grounded eval set from sampled chunks.

Run once, review printed output, then commit data/eval_set.json.

Usage:
    .venv/Scripts/python scripts/generate_eval_set.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stripe_rag.config import get_settings

# Proportional sample targets: api=7, payments=7, billing=4, webhooks=2
SAMPLES_PER_SECTION: dict[str, int] = {
    "api": 7,
    "payments": 7,
    "billing": 4,
    "webhooks": 2,
}

_SYSTEM_PROMPT = (
    "Generate a specific, factual question that can only be answered using the provided text, "
    "and a concise reference answer (1-3 sentences). "
    'Return JSON only: {"question": "...", "reference_answer": "..."}'
)


def load_chunks(data_dir: Path) -> list[dict]:
    path = data_dir / "chunks.jsonl"
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def sample_chunks(chunks: list[dict]) -> list[dict]:
    """Group by section_prefix and sample proportionally."""
    by_section: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        section = chunk.get("section_prefix", "")
        if section in SAMPLES_PER_SECTION:
            by_section[section].append(chunk)

    rng = random.Random(42)
    sampled: list[dict] = []
    for section, count in SAMPLES_PER_SECTION.items():
        pool = by_section[section]
        # Filter to chunks with enough content (>100 chars)
        pool = [c for c in pool if len(c.get("content", "")) > 100]
        selected = rng.sample(pool, min(count, len(pool)))
        sampled.extend(selected)
    return sampled


def generate_qa(client: OpenAI, chunk: dict) -> dict | None:
    """Call gpt-4o-mini to generate a Q&A pair from a chunk."""
    content = chunk.get("content", "").strip()
    if not content:
        return None

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content[:2000]},  # cap at 2000 chars
        ],
        temperature=0.3,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        return {
            "question": data["question"],
            "reference_answer": data["reference_answer"],
            "source_url": chunk.get("source_url", ""),
            "section_prefix": chunk.get("section_prefix", ""),
        }
    except (KeyError, json.JSONDecodeError) as e:
        print(f"  [WARN] Failed to parse response for chunk {chunk.get('chunk_id')}: {e}")
        return None


def main() -> None:
    settings = get_settings()
    data_dir = settings.data_dir

    print("Loading chunks...")
    chunks = load_chunks(data_dir)
    print(f"Loaded {len(chunks)} chunks")

    print("Sampling chunks...")
    sampled = sample_chunks(chunks)
    print(f"Sampled {len(sampled)} chunks across sections: "
          + ", ".join(f"{s}={sum(1 for c in sampled if c.get('section_prefix') == s)}"
                      for s in SAMPLES_PER_SECTION))

    client = OpenAI(api_key=settings.openai_api_key)

    print("\nGenerating Q&A pairs...")
    items: list[dict] = []
    for i, chunk in enumerate(sampled, start=1):
        section = chunk.get("section_prefix", "?")
        print(f"  [{i}/{len(sampled)}] section={section} chunk_id={chunk.get('chunk_id', '?')[:40]}")
        item = generate_qa(client, chunk)
        if item:
            items.append(item)

    output_path = data_dir / "eval_set.json"
    output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(items)} items to {output_path}")

    print("\n" + "=" * 70)
    print("GENERATED EVAL SET — REVIEW BEFORE COMMITTING")
    print("=" * 70)
    for i, item in enumerate(items, start=1):
        print(f"\n[{i}] section={item['section_prefix']}")
        print(f"  Q: {item['question']}")
        print(f"  A: {item['reference_answer']}")
        print(f"  URL: {item['source_url']}")
    print("\n" + "=" * 70)
    print(f"Total: {len(items)} items saved to {output_path}")


if __name__ == "__main__":
    main()
