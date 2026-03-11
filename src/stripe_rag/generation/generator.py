"""Answer generation: non-streaming, streaming, and full RAG pipeline."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator

from langsmith import traceable
from openai import AsyncOpenAI

from stripe_rag.config import Settings
from stripe_rag.generation.prompts import (
    REFUSAL_PATTERNS,
    SYSTEM_PROMPT,
    format_context_blocks,
)
from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder
from stripe_rag.retrieval.models import RetrievedChunk
from stripe_rag.retrieval.reranker import get_reranker
from stripe_rag.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_REFUSAL_RE = re.compile(
    "|".join(REFUSAL_PATTERNS),
    re.IGNORECASE,
)

_GUARDRAIL_REPLY = (
    "I'm sorry, but I can't process that request. "
    "Please ask a question about Stripe documentation."
)


def check_guardrails(question: str) -> bool:
    """Return True if the question is safe, False if it matches a refusal pattern."""
    return _REFUSAL_RE.search(question) is None


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = AsyncOpenAI(api_key=settings.openai_api_key)

        dense_embedder = OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
        sparse_embedder = SparseEmbedder()

        self._retriever = HybridRetriever(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            dense_top_k=settings.retrieval_dense_top_k,
            sparse_top_k=settings.retrieval_sparse_top_k,
        )
        self._reranker = get_reranker(settings.cohere_api_key, settings.cohere_rerank_top_n)

    def _build_messages(self, question: str, chunks: list[RetrievedChunk]) -> list[dict]:
        context = format_context_blocks(chunks)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}",
            },
        ]

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Non-streaming generation for evaluation."""
        response = await self._llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=self._build_messages(question, chunks),
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> AsyncGenerator[str]:
        """
        Async generator yielding SSE-ready JSON strings:
          {"type": "token", "text": "..."}  — one per delta
          {"type": "sources", "sources": [...]}
          {"type": "done"}
        """
        stream = await self._llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=self._build_messages(question, chunks),
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield json.dumps({"type": "token", "text": delta})

        sources = [
            {
                "url": c.source_url,
                "title": c.page_title,
                "heading_path": c.heading_path,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]
        yield json.dumps({"type": "sources", "sources": sources})
        yield json.dumps({"type": "done"})

    @traceable(name="rag_pipeline")
    async def answer(
        self,
        question: str,
        section_filter: str | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Full RAG pipeline: guard → retrieve → rerank → generate."""
        if not check_guardrails(question):
            logger.warning("Guardrail triggered for question: %.80s", question)
            return _GUARDRAIL_REPLY, []

        chunks = await self._retriever.retrieve(
            query=question,
            top_k=self._settings.retrieval_final_top_k,
            section_filter=section_filter,
        )
        reranked = await self._reranker.rerank(
            query=question,
            chunks=chunks,
            top_n=self._settings.cohere_rerank_top_n,
        )
        answer_text = await self.generate(question, reranked)
        return answer_text, reranked

    async def answer_stream(
        self,
        question: str,
        section_filter: str | None = None,
    ) -> AsyncGenerator[str]:
        """Full RAG pipeline with streaming generation."""
        if not check_guardrails(question):
            logger.warning("Guardrail triggered for question: %.80s", question)
            yield json.dumps({"type": "token", "text": _GUARDRAIL_REPLY})
            yield json.dumps({"type": "sources", "sources": []})
            yield json.dumps({"type": "done"})
            return

        chunks = await self._retriever.retrieve(
            query=question,
            top_k=self._settings.retrieval_final_top_k,
            section_filter=section_filter,
        )
        reranked = await self._reranker.rerank(
            query=question,
            chunks=chunks,
            top_n=self._settings.cohere_rerank_top_n,
        )
        async for event in self.generate_stream(question, reranked):
            yield event
