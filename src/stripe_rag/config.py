from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to this file so they work regardless of CWD
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _is_new_api_model(model_name: str) -> bool:
    """Return True for o-series and gpt-5+ models (use max_completion_tokens, no temperature)."""
    return model_name.startswith("o") or model_name.startswith("gpt-5")


class Settings(BaseSettings):
    """Single source of truth for all runtime configuration.

    Loaded from the `.env` file (and environment variables) via pydantic-settings.
    Sections: OpenAI (embeddings + LLM), Qdrant (vector DB), retrieval tuning,
    Cohere reranking, crawler parameters, LangSmith observability, and filesystem paths.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4.1"
    eval_llm_model: str = "gpt-4o-mini"
    eval_embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "stripe_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_dense_top_k: int = 40
    retrieval_sparse_top_k: int = 40
    retrieval_final_top_k: int = 25

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 50

    # ── Ingestion ─────────────────────────────────────────────────────────────
    embedding_batch_size: int = 100
    indexing_upsert_batch: int = 100
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 200
    qdrant_hnsw_full_scan_threshold: int = 10_000

    # ── Reranking ─────────────────────────────────────────────────────────────
    cohere_api_key: str | None = None
    cohere_rerank_top_n: int = 5
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── MMR ───────────────────────────────────────────────────────────────────
    mmr_enabled: bool = True       # Toggle MMR diversity re-ranking
    mmr_lambda: float = 0.5         # 0 = max diversity, 1 = max relevance
    mmr_top_k: int = 10             # Chunks to select via MMR before passing to reranker

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 60 * 60 * 24 * 14  # 2 weeks

    # ── Generation ────────────────────────────────────────────────────────────
    attribution_max_tokens: int = 20
    query_rewriting_enabled: bool = False

    # ── Session ───────────────────────────────────────────────────────────────
    session_max_history: int = 20
    session_ttl_seconds: int = 7_200
    session_cleanup_interval_seconds: int = 300

    # ── Crawler ───────────────────────────────────────────────────────────────
    crawler_concurrency: int = 10
    crawler_delay_seconds: float = 0.5
    crawler_max_pages: int = 2000
    crawler_seed_urls: list[str] = Field(
        default=[
            "https://docs.stripe.com/api",
            "https://docs.stripe.com/payments",
            "https://docs.stripe.com/webhooks",
            "https://docs.stripe.com/billing",
        ]
    )

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langsmith_api_key: str | None = None
    langsmith_project: str = "stripe-rag-chatbot"
    langsmith_tracing: bool = True

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=_PROJECT_ROOT / "data")

    @property
    def raw_html_dir(self) -> Path:
        return self.data_dir / "raw_html"

    @property
    def markdown_dir(self) -> Path:
        return self.data_dir / "markdown"

    @property
    def documents_jsonl(self) -> Path:
        return self.data_dir / "documents.jsonl"

    @property
    def chunks_jsonl(self) -> Path:
        return self.data_dir / "chunks.jsonl"


# ── Agentic Retrieval ─────────────────────────────────────────────────────
# Toggle here in code only — not readable from env vars / fly secrets.
AGENTIC_RETRIEVAL_ENABLED: bool = False
AGENTIC_RETRIEVAL_MAX_QUERIES: int = 3  # max queries the planner may emit


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance, parsing `.env` only once per process.

    The `@lru_cache` ensures that `.env` is read and validated a single time; subsequent
    calls return the cached object, making00 it safe to call from anywhere without overhead.
    """
    return Settings()  # type: ignore[call-arg]
