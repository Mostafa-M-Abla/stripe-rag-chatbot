"""Answer generation: non-streaming, streaming, and full RAG pipeline."""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncGenerator

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from stripe_rag.config import AGENTIC_RETRIEVAL_ENABLED, Settings, _is_new_api_model
from stripe_rag.generation.prompts import (
    REFUSAL_PATTERNS,
    SOURCES_QUERY_PROMPT,
    SYSTEM_PROMPT,
    format_context_blocks,
)
from stripe_rag.ingestion.embedder import OpenAIEmbedder, SparseEmbedder
from stripe_rag.retrieval.agentic_retriever import AgenticRetriever
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


def _parse_source_list(text: str, max_n: int) -> list[int]:
    """Extract ``[Source N]`` citation numbers from the attribution response.

    Parses a comma/space-separated string such as ``"1, 4"`` and returns a sorted,
    deduplicated list of 1-based indices clamped to ``[1, max_n]``.
    """
    indices = []
    for tok in re.split(r"[,\s]+", text.strip()):
        try:
            n = int(tok)
            if 1 <= n <= max_n:
                indices.append(n)
        except ValueError:
            pass
    return sorted(set(indices))


def check_guardrails(question: str) -> bool:
    """Return True if the question is safe, False if it matches a refusal pattern."""
    return _REFUSAL_RE.search(question) is None


class AnswerGenerator:
    """Orchestrates the full RAG pipeline: guardrail → retrieve → rerank → generate.

    Supports both blocking (``answer()``) and SSE streaming (``answer_stream()``) modes.
    All entry-point methods are decorated with ``@traceable`` so every invocation appears
    as a named run in the LangSmith project.
    """

    def __init__(self, settings: Settings) -> None:
        """Wire up the LLM client, embedders, retriever, and reranker from ``settings``.

        Args:
            settings: Parsed ``Settings`` instance supplying API keys and tuning params.
                The ``HybridRetriever`` and reranker are instantiated here so startup
                errors surface early (during lifespan, not on the first request).
        """
        self._settings = settings
        self._llm = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))

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
            mmr_enabled=settings.mmr_enabled,
            mmr_lambda=settings.mmr_lambda,
            mmr_top_k=settings.mmr_top_k,
        )
        self._reranker = get_reranker(
            settings.cohere_api_key, settings.cohere_rerank_top_n, settings.cohere_rerank_model
        )
        self._agentic: AgenticRetriever | None = (
            AgenticRetriever(self._retriever, AsyncOpenAI(api_key=settings.openai_api_key), settings)
            if AGENTIC_RETRIEVAL_ENABLED
            else None
        )

        from stripe_rag.cache import ResponseCache

        self._cache: ResponseCache | None = (
            ResponseCache(settings.redis_url, settings.cache_ttl_seconds)
            if settings.cache_enabled
            else None
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _llm_create(self, **kwargs):  # type: ignore[no-untyped-def]
        """Wrapper around chat.completions.create with tenacity retry (3 attempts, 1–10s backoff)."""
        return await self._llm.chat.completions.create(**kwargs)

    def _model_kwargs(self) -> dict:
        """Return model-specific kwargs for chat completions.

        Newer OpenAI models (o-series, gpt-5+):
          - require max_completion_tokens instead of max_tokens
          - only support temperature=1 (default), so omit it
        """
        model = self._settings.llm_model
        is_new = _is_new_api_model(model)
        kwargs: dict = {
            "max_completion_tokens" if is_new else "max_tokens": self._settings.llm_max_tokens
        }
        if not is_new:
            kwargs["temperature"] = self._settings.llm_temperature
        return kwargs

    def _build_messages(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """Assemble the ``[system, …history…, user]`` message list for the LLM.

        Injects ``SYSTEM_PROMPT`` at position 0, appends any prior conversation turns,
        then appends a user message containing the numbered context blocks and the question.
        """
        context = format_context_blocks(chunks)
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"})
        return messages

    @traceable(name="query_rewrite")
    async def _rewrite_query(self, question: str) -> str:
        """Rewrite the user question into a retrieval-optimised search query.

        Uses the configured LLM at temperature=0 to produce a single focused query
        string that expands abbreviations, removes conversational filler, and adds
        Stripe-specific terminology where appropriate.

        Returns the rewritten query, or the original question if the LLM returns
        an empty response.
        """
        prompt = (
            "Rewrite the following user question into a concise, specific search query "
            "optimised for retrieving Stripe documentation. "
            "Expand abbreviations, remove conversational filler, and use Stripe-specific "
            "terms where appropriate. "
            "Output only the rewritten query — no explanation, no punctuation around it.\n\n"
            f"Question: {question}\n"
            "Rewritten query:"
        )
        t0 = time.perf_counter()
        response = await self._llm_create(
            model=self._settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0,
        )
        rewrite_ms = (time.perf_counter() - t0) * 1000
        rewritten = (response.choices[0].message.content or "").strip()
        logger.info(
            "Query rewrite %.0f ms: %r → %r",
            rewrite_ms,
            question,
            rewritten if rewritten else "(empty — using original)",
        )
        if not rewritten:
            return question
        return rewritten

    async def generate(
        self, question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None
    ) -> str:
        """Non-streaming generation for evaluation."""
        response = await self._llm_create(
            model=self._settings.llm_model,
            messages=self._build_messages(question, chunks, history),
            **self._model_kwargs(),
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self, question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None
    ) -> AsyncGenerator[str]:
        """
        Async generator yielding SSE-ready JSON strings:
          {"type": "token", "text": "..."}  — one per delta (real-time)
          {"type": "sources", "sources": [...]}
          {"type": "done"}

        Two-call strategy:
          Call 1: stream clean prose token-by-token (no [Source N] markers).
          Call 2: fast non-streaming call to identify which sources were used.
        """
        # --- Call 1: stream clean prose ---
        base_messages = self._build_messages(question, chunks, history)
        stream = await self._llm_create(
            model=self._settings.llm_model,
            messages=base_messages,
            **self._model_kwargs(),
            stream=True,
        )

        full_text = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                yield json.dumps({"type": "token", "text": delta})

        # --- Call 2: identify which sources were used ---
        attribution_messages = base_messages + [
            {"role": "assistant", "content": full_text},
            {"role": "user", "content": SOURCES_QUERY_PROMPT.format(n=len(chunks))},
        ]
        attr_response = await self._llm_create(
            model=self._settings.llm_model,
            messages=attribution_messages,
            max_tokens=self._settings.attribution_max_tokens,
            temperature=0,
        )
        used_indices = _parse_source_list(
            attr_response.choices[0].message.content or "", len(chunks)
        )
        cited_chunks = [chunks[i - 1] for i in used_indices]

        sources = [
            {
                "url": c.source_url,
                "title": c.page_title,
                "heading_path": c.heading_path,
                "score": round(c.score, 4),
                "content": c.content,
            }
            for c in cited_chunks
        ]
        yield json.dumps({"type": "sources", "sources": sources})
        yield json.dumps({"type": "done"})

    async def close(self) -> None:
        """Release async resources (Redis connection)."""
        if self._cache:
            await self._cache.close()

    @traceable(name="rag_pipeline")
    async def answer(
        self,
        question: str,
        section_filter: str | None = None,
        history: list[dict] | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """Full RAG pipeline: guard → retrieve → rerank → generate."""
        if not check_guardrails(question):
            logger.warning("Guardrail triggered for question: %.80s", question)
            return _GUARDRAIL_REPLY, []

        if self._cache and not history:
            cached = await self._cache.get(question, section_filter)
            if cached:
                return cached

        retrieval_query = (
            await self._rewrite_query(question)
            if self._settings.query_rewriting_enabled
            else question
        )
        if self._agentic:
            chunks = await self._agentic.retrieve(retrieval_query, section_filter)
        else:
            chunks = await self._retriever.retrieve(
                query=retrieval_query,
                top_k=self._settings.retrieval_final_top_k,
                section_filter=section_filter,
            )
        reranked = await self._reranker.rerank(
            query=retrieval_query,
            chunks=chunks,
            top_n=self._settings.cohere_rerank_top_n,
        )
        answer_text = await self.generate(question, reranked, history)

        # Identify which sources were used via a fast follow-up call
        attribution_messages = self._build_messages(question, reranked, history) + [
            {"role": "assistant", "content": answer_text},
            {"role": "user", "content": SOURCES_QUERY_PROMPT.format(n=len(reranked))},
        ]
        attr_response = await self._llm_create(
            model=self._settings.llm_model,
            messages=attribution_messages,
            max_tokens=self._settings.attribution_max_tokens,
            temperature=0,
        )
        used_indices = _parse_source_list(
            attr_response.choices[0].message.content or "", len(reranked)
        )
        cited_chunks = [reranked[i - 1] for i in used_indices]

        if self._cache and not history:
            await self._cache.set(question, section_filter, answer_text, cited_chunks)

        return answer_text, cited_chunks

    async def answer_stream(
        self,
        question: str,
        section_filter: str | None = None,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[str]:
        """Full RAG pipeline with streaming generation."""
        if not check_guardrails(question):
            logger.warning("Guardrail triggered for question: %.80s", question)
            yield json.dumps({"type": "token", "text": _GUARDRAIL_REPLY})
            yield json.dumps({"type": "sources", "sources": []})
            yield json.dumps({"type": "done"})
            return

        if self._cache and not history:
            cached = await self._cache.get(question, section_filter)
            if cached:
                answer_text, cited_chunks = cached
                yield json.dumps({"type": "token", "text": answer_text})
                sources = [
                    {
                        "url": c.source_url,
                        "title": c.page_title,
                        "heading_path": c.heading_path,
                        "score": round(c.score, 4),
                        "content": c.content,
                    }
                    for c in cited_chunks
                ]
                yield json.dumps({"type": "sources", "sources": sources})
                yield json.dumps({"type": "done"})
                return

        retrieval_query = (
            await self._rewrite_query(question)
            if self._settings.query_rewriting_enabled
            else question
        )
        if self._agentic:
            chunks = await self._agentic.retrieve(retrieval_query, section_filter)
        else:
            chunks = await self._retriever.retrieve(
                query=retrieval_query,
                top_k=self._settings.retrieval_final_top_k,
                section_filter=section_filter,
            )
        reranked = await self._reranker.rerank(
            query=retrieval_query,
            chunks=chunks,
            top_n=self._settings.cohere_rerank_top_n,
        )

        full_text = ""
        cited_chunks_from_stream: list[RetrievedChunk] = []
        async for event in self.generate_stream(question, reranked, history):
            yield event
            parsed = json.loads(event)
            if parsed["type"] == "token":
                full_text += parsed["text"]
            elif parsed["type"] == "sources":
                for src in parsed["sources"]:
                    for chunk in reranked:
                        if chunk.source_url == src["url"] and chunk.heading_path == src["heading_path"]:
                            cited_chunks_from_stream.append(chunk)
                            break

        if self._cache and not history and full_text:
            await self._cache.set(question, section_filter, full_text, cited_chunks_from_stream)
