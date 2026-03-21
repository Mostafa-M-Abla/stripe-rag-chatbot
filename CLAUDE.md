# Stripe RAG Chatbot — Project Guide for Claude

## Project Overview

Production-quality RAG chatbot over Stripe documentation. Crawls 4 Stripe doc sections, indexes them in Qdrant with hybrid retrieval (dense + sparse), and serves grounded answers with citations via a FastAPI backend.

## Tech Stack

| Concern | Choice |
|---|---|
| Crawler | `httpx` + `BeautifulSoup4` + `markdownify` (Stripe docs are SSR'd — no JS needed, confirmed) |
| Embeddings | `text-embedding-3-large` (3072-dim) |
| Vector DB | Qdrant Cloud free tier |
| Sparse retrieval | `fastembed` BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`) |
| Reranking | Cohere Rerank v3 with `NoOpReranker` fallback |
| LLM | `gpt-4.1` for answers, `gpt-4o-mini` for eval/judges (abstracted behind `settings.llm_model`) |
| Cache | Redis-backed response cache (`ResponseCache`, 2-week TTL) |
| Rate limiting | `slowapi` (`Limiter` by remote address) |
| Observability | LangSmith `@traceable` decorators |
| Streaming | FastAPI `StreamingResponse` + SSE |
| API | FastAPI + uvicorn |

## Commands

```bash
# Install deps
.venv/Scripts/pip install -e ".[dev]"

# Lint / type check
.venv/Scripts/ruff check src/
.venv/Scripts/mypy src/

# Tests
.venv/Scripts/pytest tests/ -v

# Crawl (10 pages smoke test)
.venv/Scripts/python scripts/run_crawl.py --max-pages 10

# Full crawl
.venv/Scripts/python scripts/run_crawl.py

# Index — drops+recreates collection by default, then chunks → embeds → indexes
.venv/Scripts/python scripts/run_index.py
# Keep existing collection (upsert only):
.venv/Scripts/python scripts/run_index.py --no-recreate

# Phase 4/5 smoke test (collection health + retrieval)
.venv/Scripts/python scripts/smoke_test.py

# Run API
.venv/Scripts/uvicorn stripe_rag.api.main:app --reload

# Evaluation
.venv/Scripts/python scripts/run_eval.py
```

## Environment Variables (`.env`)

```
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=stripe-rag-chatbot
LANGSMITH_TRACING=true
QDRANT_URL=...
QDRANT_API_KEY=...

# Optional — NoOpReranker used if absent:
COHERE_API_KEY=...

# Optional — caching disabled if not set (cache_enabled=true by default, needs Redis running):
REDIS_URL=redis://localhost:6379

# Optional overrides (defaults shown):
LLM_MODEL=gpt-4.1
MMR_ENABLED=true
CACHE_ENABLED=true
QUERY_REWRITING_ENABLED=false
```

## Project Structure

```
stripe-rag-chatbot/
├── pyproject.toml
├── Dockerfile
├── fly.toml
├── CLAUDE.md                             # this file
├── .env                                  # secrets (gitignored)
│
├── src/stripe_rag/
│   ├── __init__.py                       # __version__ = "0.1.0"
│   ├── config.py                         # pydantic-settings BaseSettings, get_settings(),
│   │                                     #   AGENTIC_RETRIEVAL_ENABLED, _is_new_api_model()
│   ├── cache.py                          # ResponseCache: Redis-backed, normalize_question()
│   │
│   ├── crawler/
│   │   ├── models.py                     # RawPage, ExtractedPage dataclasses
│   │   ├── extractor.py                  # HTML → title, headings list, markdown
│   │   └── crawler.py                    # Async BFS crawler
│   │
│   ├── ingestion/
│   │   ├── cleaner.py                    # clean_markdown() — applied before chunking
│   │   ├── chunker.py                    # HeadingAwareChunker, Chunk dataclass
│   │   ├── embedder.py                   # OpenAIEmbedder (dense), SparseEmbedder (BM42)
│   │   ├── indexer.py                    # QdrantIndexer: create/drop collection + upsert
│   │   └── pipeline.py                   # chunk_documents(), embed_and_index()
│   │
│   ├── retrieval/
│   │   ├── models.py                     # RetrievedChunk dataclass
│   │   ├── retriever.py                  # HybridRetriever (Qdrant prefetch + RRF)
│   │   ├── reranker.py                   # CohereReranker + NoOpReranker + get_reranker()
│   │   └── agentic_retriever.py          # AgenticRetriever: plan-then-execute parallel queries
│   │
│   ├── generation/
│   │   ├── prompts.py                    # SYSTEM_PROMPT, format_context_blocks(), REFUSAL_PATTERNS
│   │   └── generator.py                  # AnswerGenerator: generate(), generate_stream(),
│   │                                     #   answer(), answer_stream(), check_guardrails()
│   │                                     #   + history, cache, query rewriting, agentic retrieval
│   │
│   ├── api/
│   │   ├── main.py                       # FastAPI app factory, lifespan events
│   │   ├── schemas.py                    # ChatRequest, ChatResponse, HealthResponse
│   │   ├── middleware.py                 # RequestID middleware, structured JSON logging
│   │   ├── limiter.py                    # SlowAPI Limiter singleton (get_remote_address)
│   │   └── routes/
│   │       ├── chat.py                   # POST /chat, POST /chat/stream
│   │       └── health.py                 # GET /health, GET /ready
│   │
│   └── evaluation/
│       ├── eval_set.py                   # 25 hand-crafted Q&A pairs (all 4 sections)
│       └── runner.py                     # run_evaluation(), source_hit_rate, keyword_hit_rate
│
├── scripts/
│   ├── run_crawl.py                      # CLI: crawl → data/
│   ├── run_index.py                      # CLI: data/ → Qdrant (default: drop+recreate)
│   ├── smoke_test.py                     # Phase 4/5 health check + retrieval test
│   └── run_eval.py                       # CLI: evaluation suite
│
├── data/                                 # gitignored
│   ├── raw_html/                         # {url_slug}.html
│   ├── markdown/                         # {url_slug}.md
│   ├── documents.jsonl                   # 1154 pages
│   └── chunks.jsonl                      # 8377 chunks
│
└── tests/
    ├── conftest.py
    ├── test_config.py                    # Phase 1 smoke tests (2 tests)
    ├── test_crawler.py                   # Phase 2 unit tests (23 tests, respx mocks)
    ├── test_chunker.py                   # Phase 3 unit tests (19 tests)
    ├── test_retrieval.py                 # Phase 5 (pending)
    └── test_api.py                       # Phase 7 (pending)
```

---

## Implementation Status

### ✅ Phase 1 — Scaffold & Configuration (COMPLETE)

**Files:** `pyproject.toml`, `src/stripe_rag/__init__.py`, `src/stripe_rag/config.py`

**Key `config.py` fields:**
- OpenAI: `openai_api_key`, `openai_embedding_model="text-embedding-3-large"`, `llm_model="gpt-4.1"`, `eval_llm_model="gpt-4o-mini"`, `eval_embedding_model="text-embedding-3-small"`, `llm_temperature=0.1`, `llm_max_tokens=1024`
- Qdrant: `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name="stripe_docs"`
- Retrieval: `retrieval_dense_top_k=40`, `retrieval_sparse_top_k=40`, `retrieval_final_top_k=25`
- Chunking: `chunk_max_tokens=512`, `chunk_overlap_tokens=64`, `chunk_min_tokens=50`
- Ingestion: `embedding_batch_size=100`, `indexing_upsert_batch=100`, `qdrant_hnsw_m=16`, `qdrant_hnsw_ef_construct=200`, `qdrant_hnsw_full_scan_threshold=10000`
- Reranking: `cohere_api_key=None`, `cohere_rerank_top_n=5`, `cohere_rerank_model="rerank-english-v3.0"`
- MMR: `mmr_enabled=True`, `mmr_lambda=0.5`, `mmr_top_k=10`
- Cache: `cache_enabled=True`, `redis_url="redis://localhost:6379"`, `cache_ttl_seconds=1209600` (2 weeks)
- Generation: `attribution_max_tokens=20`, `query_rewriting_enabled=False`
- Session: `session_max_history=20`, `session_ttl_seconds=7200`, `session_cleanup_interval_seconds=300`
- Crawler: `crawler_concurrency=10`, `crawler_delay_seconds=0.5`, `crawler_max_pages=2000`
- LangSmith: `langsmith_api_key`, `langsmith_project="stripe-rag-chatbot"`, `langsmith_tracing=True`
- Paths: resolved absolute from `__file__` — works regardless of CWD
- **Module-level constants** (not env-readable): `AGENTIC_RETRIEVAL_ENABLED=False`, `AGENTIC_RETRIEVAL_MAX_QUERIES=3`
- **Helper**: `_is_new_api_model(model_name)` — returns `True` for o-series and gpt-5+ models

**Verified:**
- `get_settings().llm_model` → `gpt-4.1` ✓
- `ruff check src/` → clean ✓
- `pytest tests/` → 44 passed ✓

---

### ✅ Phase 2 — Async Crawler & Content Extraction (COMPLETE)

**Files:** `src/stripe_rag/crawler/` (models, extractor, crawler), `scripts/run_crawl.py`

**Key design:**
- `StripeCrawler`: async BFS with `asyncio.Queue` + `Semaphore(concurrency)`
- `tenacity` retry: 3 attempts, exponential backoff (2–10s)
- `extract_page()`: removes nav/header/footer/aside/script/style, extracts `<main>/<article>`, converts to markdown via `markdownify`, collapses `\n{3,}` → `\n\n`
- Skips pages with `word_count < 50`
- Saves: `data/raw_html/{slug}.html`, `data/markdown/{slug}.md`, `data/documents.jsonl`

**Verified:**
- 23 crawler unit tests pass ✓
- Full crawl: 1154 pages → `data/documents.jsonl` ✓

---

### ✅ Phase 3 — Heading-Aware Chunking (COMPLETE)

**Files:** `src/stripe_rag/ingestion/chunker.py`, `src/stripe_rag/ingestion/cleaner.py`, `src/stripe_rag/ingestion/pipeline.py`, `tests/test_chunker.py`

**Key design:**
- `clean_markdown()` applied before chunking (truncates large code blocks, strips UI noise)
- `HeadingAwareChunker`: splits on h1–h4 boundaries, maintains heading stack
- `heading_path` = `"Charges > Create a charge > Parameters"`, prepended to every chunk
- `MAX_TOKENS=512`, `OVERLAP_TOKENS=64`, `MIN_CHUNK_TOKENS=50` — `tiktoken cl100k_base`
- Recursive split: `\n\n` → `\n` → sentence → token windows

**Verified:**
- 19 unit tests pass ✓
- 1154 pages → 8377 chunks in `data/chunks.jsonl` ✓

---

### ✅ Phase 4 — Embeddings & Qdrant Indexing (COMPLETE)

**Files:** `src/stripe_rag/ingestion/embedder.py`, `src/stripe_rag/ingestion/indexer.py`, `scripts/run_index.py`

**Key design:**
- `OpenAIEmbedder`: `BATCH_SIZE=100`, async, tenacity retry (3 attempts)
- `SparseEmbedder`: `fastembed` BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`)
- `QdrantIndexer`: HNSW (m=16, ef_construct=200), 3072-dim Cosine dense + sparse vectors
- Payload indexes: `section_prefix` (KEYWORD), `source_url` (KEYWORD)
- `drop_collection()` + `ensure_collection()` — run_index.py drops+recreates by default
- Upsert retry: tenacity 5 attempts, exponential backoff (4–60s) — handles Qdrant Cloud timeouts
- `run_index.py`: default drops+recreates collection; use `--no-recreate` to upsert into existing

**Verified:**
- `scripts/smoke_test.py` → 8377 points, status green, Dense dim 3072, Distance Cosine ✓
- Hybrid retrieval query returns 5 relevant chunks with RRF scores ✓

---

### ✅ Phase 5 — Hybrid Retrieval & Reranking (COMPLETE)

**Files:** `src/stripe_rag/retrieval/models.py`, `src/stripe_rag/retrieval/retriever.py`, `src/stripe_rag/retrieval/reranker.py`

**Key design:**
- `HybridRetriever.retrieve()`: embeds query dense+sparse → Qdrant prefetch with two branches + `FusionQuery(fusion=Fusion.RRF)` → `list[RetrievedChunk]`
- Optional `Filter` on `section_prefix` for scoped queries
- `CohereReranker` (rerank-english-v3.0) + `NoOpReranker` fallback — `get_reranker()` factory
- Both retriever and rerankers decorated with `@traceable`
- RRF scores are relative rank-based (0.25–0.5 range is normal and expected)
- `HybridRetriever._apply_mmr()`: static method, pure numpy, ~1 ms; selects `k` diverse chunks via MMR; controlled by `mmr_enabled` / `mmr_lambda` / `mmr_top_k` config fields

**Retrieval chain (chunk counts):**
```
Dense prefetch (40) ─┐
                      ├─ RRF fusion (Qdrant) → 25 chunks
Sparse prefetch (40) ─┘
                             ↓ [MMR if enabled — selects diverse top-k (~1 ms)]
                      Cohere / NoOp Reranker → 5 chunks
                             ↓
                            LLM
```
- `retrieval_dense_top_k=40`, `retrieval_sparse_top_k=40` → up to 80 unique candidates
- Qdrant returns top `retrieval_final_top_k=25` after server-side RRF fusion
- **MMR** (`mmr_enabled`, default `True`): if enabled, selects `mmr_top_k=10` diverse chunks from the 25 post-RRF candidates using numpy (`λ=0.5` default); duration logged at DEBUG level. Set `MMR_ENABLED=false` in `.env` to disable. When off: zero overhead.
- Cohere reranker receives 25 (or `mmr_top_k=10` if MMR enabled), filters to `cohere_rerank_top_n=5` for LLM
- **RRF score**: rank-based fusion (`Σ 1/(60+rank)`), values ~0.25–0.50 — *not* cosine similarity. Without Cohere this is the final score the UI displays. With Cohere it is replaced by Cohere's semantic relevance_score (0–1).
- **MMR score**: cosine similarity to query (numpy dot product on normalised vectors) balanced against similarity to already-selected chunks.

**Verified via smoke_test.py:**
- Query "How do I create a PaymentIntent?" → 5 chunks, all `docs.stripe.com` URLs ✓
- Top result: "Create a PaymentIntent | Stripe API Reference"`

**Next:** Add `COHERE_API_KEY` to `.env` to enable Cohere reranking (currently using NoOpReranker)

---

### ✅ Phase 6 — Answer Generation with Citations & Streaming (COMPLETE)

**Files:** `src/stripe_rag/generation/prompts.py`, `src/stripe_rag/generation/generator.py`, `src/stripe_rag/retrieval/agentic_retriever.py`, `src/stripe_rag/cache.py`

**Key design:**
- `SYSTEM_PROMPT`: answer only from context, cite as `[Source N]`, say "insufficient evidence" if weak
- `format_context_blocks()`: `[Source N] {title} > {heading_path}\nURL: {url}\n{content}\n---`
- `AnswerGenerator.generate()` (non-streaming) + `generate_stream()` (async generator of SSE JSON)
- `answer()` + `answer_stream()`: full pipeline with `@traceable(name="rag_pipeline")`; both support `history: list[dict]` for multi-turn conversations
- `check_guardrails()`: regex check for prompt injection patterns
- **Two-call attribution**: streaming uses a follow-up non-streaming call (capped at `attribution_max_tokens=20`) to identify which `[Source N]` citations were used — keeps streamed prose clean
- **Query rewriting** (`query_rewriting_enabled=False`): optional `@traceable` `_rewrite_query()` pre-step that expands abbreviations and adds Stripe-specific terms; off by default
- **Model compatibility** (`_model_kwargs()`): uses `max_completion_tokens` and omits `temperature` for o-series / gpt-5+ models; uses `max_tokens` + `temperature` for all others
- **Agentic retrieval** (`AGENTIC_RETRIEVAL_ENABLED=False` in `config.py`): when enabled, `AgenticRetriever` replaces the single `HybridRetriever.retrieve()` call. The LLM planner generates 1–3 `{query, section}` pairs (JSON), all executed in parallel via `asyncio.gather`, then merged/deduplicated by `chunk_id` (highest score wins). Toggled via module-level constant only (not env var) to avoid accidental production activation.
- **Response cache** (`cache_enabled=True`): `ResponseCache` wraps Redis with normalized question keys (`rag:v1:{section}:{normalized_q}`), TTL 2 weeks. Cache is bypassed when `history` is provided (multi-turn conversations are not cached). On Redis errors, fails silently (cache miss).
- `close()`: releases Redis connection pool (called on lifespan shutdown)

---

### 🔲 Phase 7 — FastAPI Backend (PENDING)

**Files:** `src/stripe_rag/api/main.py`, `schemas.py`, `middleware.py`, `routes/chat.py`, `routes/health.py`

**Routes:**
- `POST /chat` → `ChatResponse(answer, sources, latency_ms, model)`
- `POST /chat/stream` → SSE: `{"type":"token","text":"..."}` → `{"type":"sources","sources":[...]}` → `{"type":"done"}`
- `GET /health` → `{"status":"ok","qdrant_connected":bool,"version":"0.1.0"}`
- `GET /ready` → 200 when Qdrant reachable

**Middleware:** CORS, `RequestIDMiddleware` (X-Request-ID + structured JSON logging)

---

### 🔧 Phase 8 — Evaluation, Docker & fly.io Deployment (IN PROGRESS)

**Files:** `src/stripe_rag/evaluation/eval_set.py`, `runner.py`, `scripts/run_eval.py`, `Dockerfile`, `fly.toml`

**Evaluation pipeline (COMPLETE):**
- `eval_set.py`: 25 hand-crafted Q&A pairs across all 4 Stripe doc sections
- `runner.py`: Custom LLM-as-judge metrics — 4 direct OpenAI calls with JSON-mode prompts (faithfulness, answer_relevancy, context_recall, factual_correctness); no RAGAS dependency
- `run_eval.py`: CLI — prints results table + LangSmith project URL
- LangSmith visibility: per-question scores via LangSmith `aevaluate()` Datasets & Experiments UI
- Each judge: single `chat.completions.create()` call with `response_format={"type":"json_object"}`, returns `{"score": float, "reasoning": str}`, score clamped to [0, 1]

**Key gotchas:**
- LangSmith SDK reads from `os.environ`, not pydantic Settings — `run_eval.py` calls `os.environ.setdefault()` before `run_evaluation()` to bridge them
- RAGAS was removed: produced unreliable 0% scores due to opaque multi-step LangChain chains internally failing silently. Replaced with transparent direct OpenAI calls
- No `LANGCHAIN_*` env vars needed anymore — those were only for RAGAS's LangChain internals

**Remaining:**
- Docker: `python:3.13-slim`, pre-download BM42 model, non-root `appuser`
- fly.io: `memory=512mb`, `auto_stop_machines=true`, `min_machines_running=0`

**Targets:**
- `faithfulness ≥ 0.80`, `response_relevancy ≥ 0.75`, `context_recall ≥ 0.75`, `factual_correctness ≥ 0.75`

---

## Key Design Decisions & Gotchas

- **`pyproject.toml` build backend**: use `setuptools.build_meta`, NOT `setuptools.backends.legacy:build`
- **mypy + pydantic-settings**: `Settings()` call in `get_settings()` needs `# type: ignore[call-arg]`
- **Stripe docs rendering**: fully SSR'd — confirmed, no Playwright needed
- **`data/` directory**: gitignored. Must run crawl before indexing
- **Windows paths**: venv is at `.venv/Scripts/` (not `.venv/bin/`)
- **config.py paths**: `.env` and `data_dir` resolved relative to `__file__` — works from any CWD/IDE
- **Embedding model**: `text-embedding-3-large` (3072-dim) — changed from small for better quality
- **run_index.py default**: drops+recreates Qdrant collection on every run to avoid stale duplicate points
- **RRF scores**: values like 0.25–0.5 are normal — these are rank-based fusion scores, not cosine similarity
- **Qdrant Cloud upsert timeouts**: fixed with tenacity retry (5 attempts, 4–60s backoff) on each batch
- **ruff ignores**: `B008` (FastAPI Depends pattern), `E741` (legacy `l` variable in pre-existing files)
- **LangSmith env vars**: pydantic-settings loads `.env` into `Settings` only, not `os.environ`. LangSmith SDK reads `os.environ` directly — bridge with `os.environ.setdefault()` in CLI scripts before any SDK import is used
- **Custom LLM-as-judge**: RAGAS removed (produced 0% factual correctness on correct answers due to opaque internal failures). Each metric is now a direct OpenAI `chat.completions.create()` call with `response_format={"type":"json_object"}` — transparent and debuggable
- **Eval vs. answer models**: `eval_llm_model="gpt-4o-mini"` and `eval_embedding_model="text-embedding-3-small"` are used by the evaluation runner only; the RAG pipeline uses `llm_model="gpt-4.1"` and `openai_embedding_model="text-embedding-3-large"`
- **New API model compat**: o-series and gpt-5+ models require `max_completion_tokens` (not `max_tokens`) and don't accept `temperature` — handled by `_is_new_api_model()` + `_model_kwargs()` in `generator.py`
- **Agentic retrieval toggle**: `AGENTIC_RETRIEVAL_ENABLED` is a module-level constant in `config.py`, intentionally not readable from env vars so it can't be accidentally flipped in production via fly secrets
- **Cache bypass on history**: `ResponseCache.get/set` are skipped when `history` is provided — multi-turn answers depend on prior context so exact-match caching is unsafe
- **Rate limiter**: `src/stripe_rag/api/limiter.py` exports a single `limiter` (SlowAPI `Limiter`) instance shared by `main.py` and route handlers
