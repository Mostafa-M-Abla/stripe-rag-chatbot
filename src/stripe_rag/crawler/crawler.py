from __future__ import annotations

import asyncio
import json
import logging
from asyncio import Queue, Semaphore
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from stripe_rag.crawler.extractor import extract_page, url_to_slug
from stripe_rag.crawler.models import ExtractedPage, RawPage

logger = logging.getLogger(__name__)

_ALLOWED_PREFIXES = [
    "https://docs.stripe.com/api",
    "https://docs.stripe.com/payments",
    "https://docs.stripe.com/webhooks",
    "https://docs.stripe.com/billing",
]

_HEADERS = {
    "User-Agent": "stripe-rag-chatbot-crawler/0.1 (educational project)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_allowed_url(url: str) -> bool:
    return any(url.startswith(p) for p in _ALLOWED_PREFIXES)


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        # Strip fragment and query string
        parsed = urlparse(full)
        clean = parsed._replace(fragment="", query="").geturl()
        if _is_allowed_url(clean):
            links.append(clean)
    return links


class StripeCrawler:
    def __init__(
        self,
        seed_urls: list[str],
        raw_html_dir: Path,
        markdown_dir: Path,
        documents_jsonl: Path,
        max_pages: int = 2000,
        concurrency: int = 10,
        delay_seconds: float = 0.5,
    ) -> None:
        self.seed_urls = seed_urls
        self.raw_html_dir = raw_html_dir
        self.markdown_dir = markdown_dir
        self.documents_jsonl = documents_jsonl
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.delay_seconds = delay_seconds

        self.visited: set[str] = set()
        self.pages: list[ExtractedPage] = []

    async def run(self) -> list[ExtractedPage]:
        self.raw_html_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.documents_jsonl.parent.mkdir(parents=True, exist_ok=True)

        queue: Queue[str] = Queue()
        for url in self.seed_urls:
            await queue.put(url)

        sem = Semaphore(self.concurrency)

        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            tasks: list[asyncio.Task] = []
            while not queue.empty() or any(not t.done() for t in tasks):
                # Prune finished tasks
                tasks = [t for t in tasks if not t.done()]

                if not queue.empty() and len(self.visited) < self.max_pages:
                    url = await queue.get()
                    if url in self.visited:
                        continue
                    self.visited.add(url)
                    task = asyncio.create_task(
                        self._process(client, sem, queue, url)
                    )
                    tasks.append(task)
                else:
                    if tasks:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    else:
                        break

            # Await any remaining tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Write documents.jsonl
        with self.documents_jsonl.open("w", encoding="utf-8") as f:
            for page in self.pages:
                f.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Crawl complete: {len(self.pages)} pages indexed")
        return self.pages

    async def _process(
        self,
        client: httpx.AsyncClient,
        sem: Semaphore,
        queue: Queue,
        url: str,
    ) -> None:
        async with sem:
            try:
                raw = await self._fetch(client, url)
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                return

            await asyncio.sleep(self.delay_seconds)

        # Extract content
        try:
            page = extract_page(raw)
        except Exception as e:
            logger.warning(f"Failed to extract {url}: {e}")
            return

        if page.word_count < 50:
            logger.debug(f"Skipping sparse page: {url}")
            return

        # Save raw HTML
        slug = url_to_slug(url)
        html_path = self.raw_html_dir / f"{slug}.html"
        html_path.write_text(raw.html, encoding="utf-8")

        # Save markdown
        md_path = self.markdown_dir / f"{slug}.md"
        md_path.write_text(page.markdown, encoding="utf-8")

        self.pages.append(page)
        logger.info(f"[{len(self.pages)}/{self.max_pages}] {url}")

        # Enqueue discovered links
        if len(self.visited) < self.max_pages:
            links = _extract_links(raw.html, url)
            for link in links:
                if link not in self.visited:
                    await queue.put(link)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch(self, client: httpx.AsyncClient, url: str) -> RawPage:
        response = await client.get(url)
        response.raise_for_status()
        return RawPage(
            url=url,
            html=response.text,
            fetched_at=datetime.now(UTC),
            status_code=response.status_code,
        )
