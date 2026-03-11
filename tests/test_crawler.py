"""Phase 2 unit tests — crawler & extractor."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
import respx
import httpx

from stripe_rag.crawler.extractor import extract_page, get_section_prefix, url_to_slug
from stripe_rag.crawler.models import RawPage, ExtractedPage


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Charges | Stripe Documentation</title></head>
<body>
  <nav>Should be removed</nav>
  <header>Should be removed</header>
  <main>
    <h1>Charges</h1>
    <p>A Charge object represents a payment attempt.</p>
    <h2 id="create">Create a charge</h2>
    <p>To create a charge, send a POST request.</p>
    <h3 id="params">Parameters</h3>
    <p>amount: Required. A positive integer.</p>
  </main>
  <footer>Should be removed</footer>
</body>
</html>"""


def make_raw(url: str = "https://docs.stripe.com/api/charges", html: str = SAMPLE_HTML) -> RawPage:
    return RawPage(url=url, html=html, fetched_at=datetime.now(timezone.utc), status_code=200)


# ── url_to_slug ───────────────────────────────────────────────────────────────

def test_url_to_slug_basic():
    slug = url_to_slug("https://docs.stripe.com/api/charges")
    assert "https" not in slug
    assert "/" not in slug
    assert "." not in slug


def test_url_to_slug_no_dots_or_slashes():
    slug = url_to_slug("https://docs.stripe.com/payments/checkout/how-checkout-works")
    assert "/" not in slug
    assert "." not in slug


def test_url_to_slug_max_length():
    long_url = "https://docs.stripe.com/" + "a" * 300
    assert len(url_to_slug(long_url)) <= 200


# ── get_section_prefix ────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://docs.stripe.com/api/charges", "api"),
    ("https://docs.stripe.com/payments/intents", "payments"),
    ("https://docs.stripe.com/billing/subscriptions", "billing"),
    ("https://docs.stripe.com/webhooks/signature", "webhooks"),
    ("https://docs.stripe.com/other/page", "other"),
])
def test_get_section_prefix(url: str, expected: str):
    assert get_section_prefix(url) == expected


# ── extract_page ──────────────────────────────────────────────────────────────

def test_extract_page_title():
    raw = make_raw()
    page = extract_page(raw)
    assert page.title == "Charges"  # strips " | Stripe Documentation"


def test_extract_page_headings():
    raw = make_raw()
    page = extract_page(raw)
    texts = [h["text"] for h in page.headings]
    assert "Charges" in texts
    assert "Create a charge" in texts
    assert "Parameters" in texts


def test_extract_page_heading_levels():
    raw = make_raw()
    page = extract_page(raw)
    levels = {h["text"]: h["level"] for h in page.headings}
    assert levels["Charges"] == "h1"
    assert levels["Create a charge"] == "h2"
    assert levels["Parameters"] == "h3"


def test_extract_page_markdown_has_content():
    raw = make_raw()
    page = extract_page(raw)
    assert "Charge" in page.markdown
    assert "POST" in page.markdown or "charge" in page.markdown.lower()


def test_extract_page_noise_removed():
    raw = make_raw()
    page = extract_page(raw)
    assert "Should be removed" not in page.markdown


def test_extract_page_no_excessive_blank_lines():
    raw = make_raw()
    page = extract_page(raw)
    assert "\n\n\n" not in page.markdown


def test_extract_page_word_count():
    raw = make_raw()
    page = extract_page(raw)
    assert page.word_count > 0


def test_extract_page_section_prefix():
    raw = make_raw()
    page = extract_page(raw)
    assert page.section_prefix == "api"


def test_extract_page_url_preserved():
    raw = make_raw()
    page = extract_page(raw)
    assert page.url == "https://docs.stripe.com/api/charges"


# ── link extraction (via _extract_links internal) ─────────────────────────────

def test_link_extraction_filters_allowed_prefixes():
    from stripe_rag.crawler.crawler import _extract_links

    html = """<html><body>
    <a href="/payments/intents">Payment Intents</a>
    <a href="/billing/subscriptions">Subscriptions</a>
    <a href="/dashboard">Dashboard</a>
    <a href="https://stripe.com/blog">Blog</a>
    <a href="#anchor">Anchor</a>
    </body></html>"""

    links = _extract_links(html, "https://docs.stripe.com/api")
    assert all(
        any(link.startswith(p) for p in [
            "https://docs.stripe.com/payments",
            "https://docs.stripe.com/billing",
            "https://docs.stripe.com/connect",
            "https://docs.stripe.com/api",
        ])
        for link in links
    )
    # Non-allowed URLs must not appear
    assert not any("blog" in link or "dashboard" in link for link in links)


def test_link_extraction_strips_fragments():
    from stripe_rag.crawler.crawler import _extract_links

    html = '<html><body><a href="/api/charges#section">Charges</a></body></html>'
    links = _extract_links(html, "https://docs.stripe.com")
    assert all("#" not in link for link in links)


def test_link_extraction_strips_query_strings():
    from stripe_rag.crawler.crawler import _extract_links

    html = '<html><body><a href="/api/charges?ref=nav">Charges</a></body></html>'
    links = _extract_links(html, "https://docs.stripe.com")
    assert all("?" not in link for link in links)


# ── StripeCrawler with mocked HTTP ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_crawler_respects_max_pages():
    """Crawler stops after max_pages even if more links exist."""
    import tempfile
    from pathlib import Path
    from stripe_rag.crawler.crawler import StripeCrawler

    page_html = """<html><head><title>Page | Stripe Documentation</title></head>
    <body><main>
    <h1>Test Page</h1>
    <p>This is a test page with enough words to pass the filter threshold.</p>
    <a href="/payments/page1">Link 1</a>
    <a href="/payments/page2">Link 2</a>
    <a href="/payments/page3">Link 3</a>
    </main></body></html>"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with respx.mock(assert_all_called=False) as mock:
            mock.get(url__startswith="https://docs.stripe.com").mock(
                return_value=httpx.Response(200, text=page_html)
            )

            crawler = StripeCrawler(
                seed_urls=["https://docs.stripe.com/payments"],
                raw_html_dir=tmp / "raw_html",
                markdown_dir=tmp / "markdown",
                documents_jsonl=tmp / "documents.jsonl",
                max_pages=3,
                concurrency=2,
                delay_seconds=0,
            )
            pages = await crawler.run()

        assert len(pages) <= 3


@pytest.mark.asyncio
async def test_crawler_saves_files():
    """Crawler saves .html, .md, and documents.jsonl."""
    import tempfile
    import json
    from pathlib import Path
    from stripe_rag.crawler.crawler import StripeCrawler

    page_html = """<html><head><title>Payments | Stripe Documentation</title></head>
    <body><main>
    <h1>Payments</h1>
    <p>Accept payments on your website or mobile app using Stripe's payment APIs.
    Stripe provides a set of programmable APIs and tools to let you accept payments
    and build custom payment experiences. You can use Stripe to accept credit cards,
    debit cards, and other payment methods from customers around the world.</p>
    </main></body></html>"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://docs.stripe.com/payments").mock(
                return_value=httpx.Response(200, text=page_html)
            )

            crawler = StripeCrawler(
                seed_urls=["https://docs.stripe.com/payments"],
                raw_html_dir=tmp / "raw_html",
                markdown_dir=tmp / "markdown",
                documents_jsonl=tmp / "documents.jsonl",
                max_pages=1,
                concurrency=1,
                delay_seconds=0,
            )
            pages = await crawler.run()

        assert len(pages) == 1
        assert list((tmp / "raw_html").glob("*.html"))
        assert list((tmp / "markdown").glob("*.md"))
        assert (tmp / "documents.jsonl").exists()

        lines = (tmp / "documents.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        doc = json.loads(lines[0])
        assert doc["url"] == "https://docs.stripe.com/payments"
        assert doc["title"] == "Payments"
        assert "section_prefix" in doc
        assert "headings" in doc


@pytest.mark.asyncio
async def test_crawler_skips_http_errors():
    """Crawler continues after 404/500 without crashing."""
    import tempfile
    from pathlib import Path
    from stripe_rag.crawler.crawler import StripeCrawler

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://docs.stripe.com/payments").mock(
                return_value=httpx.Response(404)
            )

            crawler = StripeCrawler(
                seed_urls=["https://docs.stripe.com/payments"],
                raw_html_dir=tmp / "raw_html",
                markdown_dir=tmp / "markdown",
                documents_jsonl=tmp / "documents.jsonl",
                max_pages=5,
                concurrency=1,
                delay_seconds=0,
            )
            # Should not raise
            pages = await crawler.run()
        assert pages == []
