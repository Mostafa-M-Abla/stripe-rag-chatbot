"""Chat endpoints: POST /chat and POST /chat/stream."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from stripe_rag.api.schemas import ChatRequest, ChatResponse, SourceRef
from stripe_rag.config import get_settings
from stripe_rag.generation.generator import AnswerGenerator

router = APIRouter()


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

@dataclass
class Session:
    history: list[dict] = field(default_factory=list)
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    MAX_HISTORY = 20   # messages = 10 turns
    TTL_SECONDS = 7_200  # 2 hours

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()  # guards dict mutations only

    async def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        async with self._lock:
            if session_id and session_id in self._sessions:
                s = self._sessions[session_id]
                s.last_accessed = datetime.now(timezone.utc)
                return session_id, s
            new_id = str(uuid.uuid4())
            s = Session()
            self._sessions[new_id] = s
            return new_id, s

    async def append_turn(self, session_id: str, question: str, answer: str) -> None:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return
            s.history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            if len(s.history) > self.MAX_HISTORY:
                s.history = s.history[-self.MAX_HISTORY:]
            s.last_accessed = datetime.now(timezone.utc)

    async def evict_expired(self) -> int:
        now = datetime.now(timezone.utc)
        async with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if (now - s.last_accessed).total_seconds() > self.TTL_SECONDS
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)


# ---------------------------------------------------------------------------
# Module-level singletons; initialised in lifespan
# ---------------------------------------------------------------------------

_generator: AnswerGenerator | None = None
_store: SessionStore | None = None


def get_generator() -> AnswerGenerator:
    if _generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialised")
    return _generator


def set_generator(gen: AnswerGenerator) -> None:
    global _generator
    _generator = gen


def get_store() -> SessionStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Session store not initialised")
    return _store


def set_store(store: SessionStore) -> None:
    global _store
    _store = store


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    generator: AnswerGenerator = Depends(get_generator),
    store: SessionStore = Depends(get_store),
) -> ChatResponse:
    settings = get_settings()
    session_id, session = await store.get_or_create(request.session_id)

    async with session.lock:
        history_snapshot = list(session.history)

    start = time.perf_counter()
    answer_text, chunks = await generator.answer(
        question=request.question,
        section_filter=request.section_filter,
        history=history_snapshot,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if chunks:
        await store.append_turn(session_id, request.question, answer_text)

    sources = [
        SourceRef(
            url=c.source_url,
            title=c.page_title,
            heading_path=c.heading_path,
            score=round(c.score, 4),
        )
        for c in chunks
    ]
    return ChatResponse(
        answer=answer_text,
        sources=sources,
        latency_ms=latency_ms,
        model=settings.llm_model,
        session_id=session_id,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    generator: AnswerGenerator = Depends(get_generator),
    store: SessionStore = Depends(get_store),
) -> StreamingResponse:
    session_id, session = await store.get_or_create(request.session_id)

    async with session.lock:
        history_snapshot = list(session.history)

    async def event_stream() -> AsyncGenerator[str, None]:
        accumulated = ""
        guardrail_hit = False

        async for event_json in generator.answer_stream(
            question=request.question,
            section_filter=request.section_filter,
            history=history_snapshot,
        ):
            yield f"data: {event_json}\n\n"
            parsed = json.loads(event_json)
            if parsed["type"] == "token":
                accumulated += parsed["text"]
            elif parsed["type"] == "sources" and not parsed["sources"]:
                guardrail_hit = True

        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"

        if not guardrail_hit and accumulated:
            await store.append_turn(session_id, request.question, accumulated)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
