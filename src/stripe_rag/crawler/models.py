from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawPage:
    """Raw HTTP result: the original HTML bytes and metadata before any processing."""

    url: str
    html: str
    fetched_at: datetime
    status_code: int


@dataclass
class ExtractedPage:
    """Cleaned, markdown-converted page ready for chunking and indexing.

    Fields:
        url: canonical page URL (used as Qdrant payload and citation link)
        title: page title stripped of " | Stripe Documentation" suffix
        headings: ordered list of {level, text, id} dicts extracted from the HTML
        markdown: ATX-formatted markdown produced by markdownify after noise removal
        word_count: whitespace-split word count used to gate sparse pages (< 50 skipped)
        section_prefix: short label derived from the URL path, e.g. "payments" or "api"
        fetched_at: UTC timestamp of the HTTP response
    """

    url: str
    title: str
    headings: list[dict[str, str]]  # [{level, text, id}]
    markdown: str
    word_count: int
    section_prefix: str
    fetched_at: datetime

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for JSONL persistence."""
        return {
            "url": self.url,
            "title": self.title,
            "headings": self.headings,
            "markdown": self.markdown,
            "word_count": self.word_count,
            "section_prefix": self.section_prefix,
            "fetched_at": self.fetched_at.isoformat(),
        }
