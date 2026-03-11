from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify

from stripe_rag.crawler.models import ExtractedPage, RawPage

_SEED_PREFIXES = [
    "https://docs.stripe.com/api",
    "https://docs.stripe.com/payments",
    "https://docs.stripe.com/webhooks",
    "https://docs.stripe.com/billing",
]

# Tags to strip before converting to markdown
_NOISE_TAGS = [
    "nav", "header", "footer", "aside", "script", "style",
    "noscript", "iframe", "svg", "button", "form",
]


def get_section_prefix(url: str) -> str:
    for prefix in _SEED_PREFIXES:
        if url.startswith(prefix):
            # e.g. "payments", "billing", "connect", "api"
            return prefix.split("/")[-1]
    return "other"


def url_to_slug(url: str) -> str:
    slug = url.replace("https://", "").replace("http://", "")
    slug = re.sub(r"[/.]", "_", slug)
    return slug[:200]  # cap length for filesystem safety


def extract_page(raw: RawPage) -> ExtractedPage:
    soup = BeautifulSoup(raw.html, "lxml")

    # Remove noise elements
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True).replace(" | Stripe Documentation", "").strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else raw.url

    # Extract headings
    headings: list[dict[str, str]] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        headings.append({
            "level": tag.name,
            "text": tag.get_text(strip=True),
            "id": tag.get("id", ""),
        })

    # Find main content area
    content_el = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find(class_=re.compile(r"content|docs|article", re.I))
        or soup.body
    )

    # Convert to markdown
    raw_md = markdownify(
        str(content_el) if content_el else raw.html,
        heading_style="ATX",
        bullets="-",
        strip=["a"],  # strip anchor tags but keep text
    )

    # Collapse excessive blank lines
    markdown = re.sub(r"\n{3,}", "\n\n", raw_md).strip()

    word_count = len(markdown.split())
    section_prefix = get_section_prefix(raw.url)

    return ExtractedPage(
        url=raw.url,
        title=title,
        headings=headings,
        markdown=markdown,
        word_count=word_count,
        section_prefix=section_prefix,
        fetched_at=raw.fetched_at,
    )
