"""Redis-backed response cache with exact-match normalized question keys."""
from __future__ import annotations

import dataclasses
import json
import logging
import re

import redis.asyncio as aioredis

from stripe_rag.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def normalize_question(question: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", question.lower().strip())


class ResponseCache:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds

    def _key(self, question: str, section_filter: str | None) -> str:
        return f"rag:v1:{section_filter or ''}:{normalize_question(question)}"

    async def get(
        self, question: str, section_filter: str | None
    ) -> tuple[str, list[RetrievedChunk]] | None:
        try:
            raw = await self._client.get(self._key(question, section_filter))
            if raw is None:
                return None
            data = json.loads(raw)
            chunks = [RetrievedChunk(**c) for c in data["chunks"]]
            logger.debug("Cache hit: %.80s", question)
            return data["answer"], chunks
        except Exception:
            logger.warning("Cache get failed — treating as miss", exc_info=True)
            return None

    async def set(
        self,
        question: str,
        section_filter: str | None,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> None:
        try:
            payload = json.dumps({
                "answer": answer,
                "chunks": [dataclasses.asdict(c) for c in chunks],
            })
            await self._client.set(self._key(question, section_filter), payload, ex=self._ttl)
            logger.debug("Cache set: %.80s", question)
        except Exception:
            logger.warning("Cache set failed — continuing without caching", exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()
