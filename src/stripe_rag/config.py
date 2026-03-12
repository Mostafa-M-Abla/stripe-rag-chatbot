from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to this file so they work regardless of CWD
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
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
    eval_llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Qdrant ────────────────────────────────────────────────────────────────config
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "stripe_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_dense_top_k: int = 20
    retrieval_sparse_top_k: int = 20
    retrieval_final_top_k: int = 5

    # ── Reranking ─────────────────────────────────────────────────────────────
    cohere_api_key: str | None = None
    cohere_rerank_top_n: int = 5

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
