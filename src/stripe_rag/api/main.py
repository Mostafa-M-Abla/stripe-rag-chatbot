"""FastAPI application factory with lifespan, CORS, and middleware."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stripe_rag.api.middleware import RequestIDMiddleware
from stripe_rag.api.routes import chat as chat_module
from stripe_rag.api.routes.chat import router as chat_router
from stripe_rag.api.routes.health import router as health_router
from stripe_rag.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Session store (no external deps)
    store = chat_module.SessionStore()
    chat_module.set_store(store)

    # Background cleanup: evict expired sessions every 5 minutes
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(300)
            evicted = await store.evict_expired()
            if evicted:
                logger.info("Session cleanup: evicted %d expired sessions", evicted)

    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info("Initialising AnswerGenerator…")
    try:
        from stripe_rag.generation.generator import AnswerGenerator

        generator = AnswerGenerator(settings)
        chat_module.set_generator(generator)
        logger.info("AnswerGenerator ready")
    except Exception as exc:
        logger.error(f"AnswerGenerator init failed: {exc} — /chat endpoints will return 503")

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stripe RAG Chatbot",
        description="Grounded Q&A over Stripe documentation with citations",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health_router, tags=["health"])
    app.include_router(chat_router, prefix="/chat", tags=["chat"])

    return app


app = create_app()
