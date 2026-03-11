"""Chat endpoints: POST /chat and POST /chat/stream."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from stripe_rag.api.schemas import ChatRequest, ChatResponse, SourceRef
from stripe_rag.config import get_settings
from stripe_rag.generation.generator import AnswerGenerator

router = APIRouter()

# Module-level singleton; initialised in lifespan
_generator: AnswerGenerator | None = None


def get_generator() -> AnswerGenerator:
    if _generator is None:
        raise HTTPException(status_code=503, detail="Generator not initialised")
    return _generator


def set_generator(gen: AnswerGenerator) -> None:
    global _generator
    _generator = gen


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    generator: AnswerGenerator = Depends(get_generator),
) -> ChatResponse:
    settings = get_settings()
    start = time.perf_counter()

    answer_text, chunks = await generator.answer(
        question=request.question,
        section_filter=request.section_filter,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
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
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    generator: AnswerGenerator = Depends(get_generator),
) -> StreamingResponse:
    async def event_stream():
        async for event_json in generator.answer_stream(
            question=request.question,
            section_filter=request.section_filter,
        ):
            yield f"data: {event_json}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
