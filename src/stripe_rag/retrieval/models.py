"""Retrieval data models."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_url: str
    page_title: str
    section_prefix: str
    heading_path: str
    content: str
    token_count: int
    score: float
