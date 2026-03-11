from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawPage:
    url: str
    html: str
    fetched_at: datetime
    status_code: int


@dataclass
class ExtractedPage:
    url: str
    title: str
    headings: list[dict[str, str]]  # [{level, text, id}]
    markdown: str
    word_count: int
    section_prefix: str
    fetched_at: datetime

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "headings": self.headings,
            "markdown": self.markdown,
            "word_count": self.word_count,
            "section_prefix": self.section_prefix,
            "fetched_at": self.fetched_at.isoformat(),
        }
