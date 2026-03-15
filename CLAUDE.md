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

# Add for Phase 5 Cohere reranking (optional — NoOpReranker used if absent):
COHERE_API_KEY=...
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
│   ├── config.py                         # pydantic-settings BaseSettings, get_settings()
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
│   │   └── reranker.py                   # CohereReranker + NoOpReranker + get_reranker()
│   │
│   ├── generation/
│   │   ├── prompts.py                    # SYSTEM_PROMPT, format_context_blocks(), REFUSAL_PATTERNS
│   │   └── generator.py                  # AnswerGenerator: generate(), generate_stream(),
│   │                                     #   answer(), answer_stream(), check_guardrails()
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
- OpenAI: `openai_api_key`, `openai_embedding_model="text-embedding-3-large"`, `llm_model="gpt-4o-mini"`, `llm_temperature=0.1`, `llm_max_tokens=1024`
- Qdrant: `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name="stripe_docs"`
- Retrieval: `retrieval_dense_top_k=40`, `retrieval_sparse_top_k=40`, `retrieval_final_top_k=25`
- Reranking: `cohere_api_key=None`, `cohere_rerank_top_n=5`
- Crawler: `crawler_concurrency=10`, `crawler_delay_seconds=0.5`, `crawler_max_pages=2000`
- LangSmith: `langsmith_api_key`, `langsmith_project="stripe-rag-chatbot"`, `langsmith_tracing=True`
- Paths: resolved absolute from `__file__` — works regardless of CWD

**Verified:**
- `get_settings().llm_model` → `gpt-4o-mini` ✓
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

**Retrieval chain (chunk counts):**
```
Dense prefetch (40) ─┐
                      ├─ RRF fusion → top 25 → Cohere Reranker → top 5 → LLM
Sparse prefetch (40) ─┘
```
- `retrieval_dense_top_k=40`, `retrieval_sparse_top_k=40` → up to 80 unique candidates
- Qdrant returns top `retrieval_final_top_k=25` after server-side RRF fusion
- Cohere reranker receives 25, aggressively filters to `cohere_rerank_top_n=5` for LLM
- **RRF score**: rank-based fusion (`Σ 1/(60+rank)`), values ~0.25–0.50 — *not* cosine similarity.
  Without Cohere this is the final score the UI displays. With Cohere it is replaced by Cohere's
  semantic relevance_score (0–1).
- **No MMR**: diversity is not enforced; top chunks can be semantically similar.

**Verified via smoke_test.py:**
- Query "How do I create a PaymentIntent?" → 5 chunks, all `docs.stripe.com` URLs ✓
- Top result: "Create a PaymentIntent | Stripe API Reference"`

**Next:** Add `COHERE_API_KEY` to `.env` to enable Cohere reranking (currently using NoOpReranker)

---

### ✅ Phase 6 — Answer Generation with Citations & Streaming (COMPLETE)

**Files:** `src/stripe_rag/generation/prompts.py`, `src/stripe_rag/generation/generator.py`

**Key design:**
- `SYSTEM_PROMPT`: answer only from context, cite as `[Source N]`, say "insufficient evidence" if weak
- `format_context_blocks()`: `[Source N] {title} > {heading_path}\nURL: {url}\n{content}\n---`
- `AnswerGenerator.generate()` (non-streaming) + `generate_stream()` (async generator of SSE JSON)
- `answer()` + `answer_stream()`: full pipeline with `@traceable(name="rag_pipeline")`
- `check_guardrails()`: regex check for prompt injection patterns

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
