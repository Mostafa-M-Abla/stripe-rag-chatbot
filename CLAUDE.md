# Stripe RAG Chatbot — Project Guide for Claude

## Project Overview

Production-quality RAG chatbot over Stripe documentation. Crawls 4 Stripe doc sections, indexes them in Qdrant with hybrid retrieval (dense + sparse), and serves grounded answers with citations via a FastAPI backend.

## Tech Stack

| Concern | Choice |
|---|---|
| Crawler | `httpx` + `BeautifulSoup4` + `markdownify` (Stripe docs are SSR'd — no JS needed, confirmed) |
| Embeddings | `text-embedding-3-small` (1536-dim) |
| Vector DB | Qdrant Cloud free tier |
| Sparse retrieval | `fastembed` BM42 (`Qdrant/bm42-all-minilm-l6-v2-attentions`) |
| Reranking | Cohere Rerank v3 with `NoOpReranker` fallback |
| LLM | `gpt-4o-mini` (abstracted behind `settings.LLM_MODEL`) |
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

# Index (chunk → embed → index)
.venv/Scripts/python scripts/run_index.py --stage all

# Run API
.venv/Scripts/uvicorn stripe_rag.api.main:app --reload

# Evaluation
.venv/Scripts/python scripts/run_eval.py
```

## Environment Variables (`.env`)

```
OPENAI_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=stripe-rag-chatbot      # fixed from rag-papers-eval
LANGSMITH_TRACING=true

# Add for Phase 4+:
QDRANT_URL=...
QDRANT_API_KEY=...

# Add for Phase 5+:
COHERE_API_KEY=...
```

## Project Structure

```
stripe-rag-chatbot/
├── pyproject.toml
├── CLAUDE.md                             # this file
├── .env                                  # secrets (gitignored)
│
├── src/stripe_rag/
│   ├── __init__.py                       # __version__ = "0.1.0"
│   ├── config.py                         # pydantic-settings BaseSettings, get_settings()
│   │
│   ├── crawler/
│   │   ├── models.py                     # RawPage, ExtractedPage dataclasses
│   │   ├── extractor.py                  # HTML → title, headings list, markdown
│   │   └── crawler.py                    # Async BFS crawler
│   │
│   ├── ingestion/
│   │   ├── chunker.py                    # HeadingAwareChunker, Chunk dataclass
│   │   ├── embedder.py                   # OpenAIEmbedder (dense), SparseEmbedder (BM42)
│   │   ├── indexer.py                    # QdrantIndexer: create collection + upsert
│   │   └── pipeline.py                   # Orchestrates crawl → chunk → embed → index
│   │
│   ├── retrieval/
│   │   ├── models.py                     # RetrievedChunk dataclass
│   │   ├── retriever.py                  # HybridRetriever (Qdrant prefetch + RRF)
│   │   └── reranker.py                   # CohereReranker + NoOpReranker fallback
│   │
│   ├── generation/
│   │   ├── prompts.py                    # SYSTEM_PROMPT, format_context_blocks()
│   │   └── generator.py                  # AnswerGenerator: generate(), generate_stream()
│   │
│   ├── api/
│   │   ├── main.py                       # FastAPI app factory, lifespan events
│   │   ├── schemas.py                    # ChatRequest, ChatResponse, HealthResponse
│   │   ├── middleware.py                 # RequestID middleware, structured JSON logging
│   │   └── routes/
│   │       ├── chat.py                   # POST /chat, POST /chat/stream
│   │       └── health.py                 # GET /health, GET /ready
│   │
│   └── evaluation/
│       ├── eval_set.py                   # 25 hand-crafted Q&A pairs
│       └── runner.py                     # run_evaluation(), logs to LangSmith
│
├── scripts/
│   ├── run_crawl.py                      # CLI: crawl → data/
│   ├── run_index.py                      # CLI: data/ → Qdrant
│   └── run_eval.py                       # CLI: evaluation suite
│
├── data/                                 # gitignored
│   ├── raw_html/                         # {url_slug}.html
│   ├── markdown/                         # {url_slug}.md
│   ├── documents.jsonl                   # one page per line
│   └── chunks.jsonl                      # one chunk per line (after Phase 3)
│
└── tests/
    ├── conftest.py
    ├── test_config.py                    # Phase 1 smoke tests
    ├── test_crawler.py                   # Phase 2 unit tests (23 tests, respx mocks)
    ├── test_chunker.py                   # Phase 3 (pending)
    ├── test_retrieval.py                 # Phase 5 (pending)
    └── test_api.py                       # Phase 7 (pending)
```

---

## Implementation Status

### ✅ Phase 1 — Scaffold & Configuration (COMPLETE)

**Files:** `pyproject.toml`, `src/stripe_rag/__init__.py`, `src/stripe_rag/config.py`

**Key `config.py` fields:**
- OpenAI: `openai_api_key`, `openai_embedding_model="text-embedding-3-small"`, `llm_model="gpt-4o-mini"`, `llm_temperature=0.1`, `llm_max_tokens=1024`
- Qdrant: `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name="stripe_docs"`
- Retrieval: `retrieval_dense_top_k=20`, `retrieval_sparse_top_k=20`, `retrieval_final_top_k=5`
- Reranking: `cohere_api_key=None`, `cohere_rerank_top_n=5`
- Crawler: `crawler_concurrency=10`, `crawler_delay_seconds=0.5`, `crawler_max_pages=2000`
- LangSmith: `langsmith_api_key`, `langsmith_project="stripe-rag-chatbot"`, `langsmith_tracing=True`

**Verified:**
- `get_settings().llm_model` → `gpt-4o-mini` ✓
- `ruff check src/` → clean ✓
- `mypy src/` → clean ✓
- `pytest tests/` → 2 passed ✓

---

### ✅ Phase 2 — Async Crawler & Content Extraction (COMPLETE)

**Files:** `src/stripe_rag/crawler/` (models, extractor, crawler), `scripts/run_crawl.py`

**Key design:**
- `StripeCrawler`: async BFS with `asyncio.Queue` + `Semaphore(concurrency)`
- `tenacity` retry: 3 attempts, exponential backoff (2–10s)
- `extract_page()`: removes nav/header/footer/aside/script/style, extracts `<main>/<article>`, converts to markdown via `markdownify`, collapses `\n{3,}` → `\n\n`
- Skips pages with `word_count < 50`
- Saves: `data/raw_html/{slug}.html`, `data/markdown/{slug}.md`, `data/documents.jsonl`

**Confirmed findings:**
- Stripe docs are fully SSR'd — `httpx` gets real content, no Playwright needed
- `/api` section returns 1320 words of real content

**Verified:**
- 25/25 unit tests pass (23 crawler + 2 config) ✓
- `run_crawl.py --max-pages 10` → 10 pages in ~3.6s ✓
- All 4 seed sections (payments, billing, connect, api) crawled ✓

---

### 🔲 Phase 3 — Heading-Aware Chunking (PENDING)

**Files to create:** `src/stripe_rag/ingestion/chunker.py`, `tests/test_chunker.py`

**Strategy:**
- Split markdown on h1–h4 boundaries → `(heading_path, section_text)` pairs
- Heading stack: level-N replaces everything at level ≥ N
- `heading_path` = `"Charges > Create a charge > Parameters"`
- `MAX_TOKENS=512`, `OVERLAP_TOKENS=64`, `MIN_CHUNK_TOKENS=50`
- Recursive split: `\n\n` → `\n` → sentence → char
- Prepend `heading_path` to each chunk content
- Token counting: `tiktoken` `cl100k_base`

**`Chunk` dataclass:** `chunk_id (uuid4)`, `source_url`, `page_title`, `section_prefix`, `heading_path`, `content`, `token_count`, `char_count`, `chunk_index`, `total_chunks`

---

### 🔲 Phase 4 — Embeddings & Qdrant Indexing (PENDING)

**Files to create:** `src/stripe_rag/ingestion/embedder.py`, `src/stripe_rag/ingestion/indexer.py`, `src/stripe_rag/ingestion/pipeline.py`, `scripts/run_index.py`

**Key design:**
- `OpenAIEmbedder`: `BATCH_SIZE=100`, async with `tqdm`, tenacity retry
- `SparseEmbedder`: `fastembed` BM42
- Qdrant collection: dense vector (1536-dim, Cosine) + sparse vector, HNSW (m=16, ef_construct=200)
- Payload indexes: `section_prefix` (KEYWORD), `source_url` (KEYWORD)
- `run_index.py --stage [chunk|embed|index|all]` with `--dry-run`

**Requires:** Add `QDRANT_URL` and `QDRANT_API_KEY` to `.env`

---

### 🔲 Phase 5 — Hybrid Retrieval & Reranking (PENDING)

**Files to create:** `src/stripe_rag/retrieval/models.py`, `src/stripe_rag/retrieval/retriever.py`, `src/stripe_rag/retrieval/reranker.py`

**Key design:**
- `HybridRetriever.retrieve()`: parallel dense+sparse embed → Qdrant `QueryRequest` with two `Prefetch` branches + `FusionQuery(fusion=Fusion.RRF)` → `list[RetrievedChunk]`
- Optional `Filter` on `section_prefix` for scoped queries
- `CohereReranker` + `NoOpReranker` fallback (activated when `COHERE_API_KEY` not set)
- Both decorated with `@traceable`

**Requires:** Add `COHERE_API_KEY` to `.env`

---

### 🔲 Phase 6 — Answer Generation with Citations & Streaming (PENDING)

**Files to create:** `src/stripe_rag/generation/prompts.py`, `src/stripe_rag/generation/generator.py`

**Key design:**
- `SYSTEM_PROMPT`: answer only from context, cite as `[Source N]`, say "insufficient evidence" if weak
- `format_context_blocks()`: `[Source N] {title} > {heading_path}\nURL: {url}\n{content}\n---`
- `AnswerGenerator.generate()` (non-streaming) + `generate_stream()` (async iterator of tokens then sources SSE event)
- `answer()`: full pipeline with `@traceable(name="rag_pipeline")`
- Inline guardrails: regex check for prompt injection patterns

---

### 🔲 Phase 7 — FastAPI Backend (PENDING)

**Files to create:** `src/stripe_rag/api/main.py`, `schemas.py`, `middleware.py`, `routes/chat.py`, `routes/health.py`

**Routes:**
- `POST /chat` → `ChatResponse(answer, sources, latency_ms, model)`
- `POST /chat/stream` → SSE: `{"type":"token","text":"..."}` → `{"type":"sources","sources":[...]}` → `{"type":"done"}`
- `GET /health` → `{"status":"ok","qdrant_connected":bool,"version":"0.1.0"}`
- `GET /ready` → 200 when Qdrant reachable

**Middleware:** CORS, `RequestIDMiddleware` (X-Request-ID + structured JSON logging)

---

### 🔲 Phase 8 — Evaluation, Docker & fly.io Deployment (PENDING)

**Files to create:** `src/stripe_rag/evaluation/eval_set.py`, `runner.py`, `scripts/run_eval.py`, `Dockerfile`, `fly.toml`

**Targets:**
- `source_hit_rate ≥ 0.80`, `keyword_hit_rate ≥ 0.75`
- Docker: `python:3.13-slim`, pre-download BM42 model, non-root `appuser`
- fly.io: `memory=512mb`, `auto_stop_machines=true`, `min_machines_running=0`

---

## Key Design Decisions & Gotchas

- **`pyproject.toml` build backend**: use `setuptools.build_meta`, NOT `setuptools.backends.legacy:build` (requires newer setuptools not available in this venv)
- **mypy + pydantic-settings**: `Settings()` call in `get_settings()` needs `# type: ignore[call-arg]`
- **Stripe docs rendering**: fully SSR'd — confirmed with live smoke test, no Playwright needed
- **`data/` directory**: gitignored. Must run crawl before indexing
- **Windows paths**: venv is at `.venv/Scripts/` (not `.venv/bin/`)
- **LangSmith project**: was `rag-papers-eval` in `.env`, fixed to `stripe-rag-chatbot`
