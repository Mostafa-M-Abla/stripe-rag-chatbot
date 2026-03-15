"""Retrieval data models."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A single search result returned by ``HybridRetriever``.

    Fields:
        chunk_id: UUID of the Qdrant point.
        source_url: Canonical URL of the originating page (used as citation link).
        page_title: Title of the originating page.
        section_prefix: Short section label (e.g. ``"payments"``).
        heading_path: Heading breadcrumb that provides structural context.
        content: Full chunk text (heading path prepended).
        token_count: ``cl100k_base`` token count of ``content``.
        score: Relevance score — RRF fusion score (~0.25–0.50) when ``NoOpReranker``
            is used, or Cohere semantic relevance score (0–1) after reranking.
    """

    chunk_id: str
    source_url: str
    page_title: str
    section_prefix: str
    heading_path: str
    content: str
    token_count: int
    score: float
