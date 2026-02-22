# Harel Insurance Customer Support Chatbot

A production-grade RAG (Retrieval-Augmented Generation) chatbot for [Harel Insurance](https://www.harel-group.co.il), Israel's largest insurance and financial services group. The system ingests real insurance policy data, answers customer questions across 8 insurance domains, and grounds every answer in official documentation with explicit citations.

Built as an APEX Data Science capstone project. Competes against a GPT-5.2 baseline using retrieval-augmented generation over open-source models.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Walkthrough: End-to-End Pipeline](#walkthrough-end-to-end-pipeline)
  - [Step 1: Data Scraping](#step-1-data-scraping)
  - [Step 2: Document Parsing](#step-2-document-parsing)
  - [Step 3: Semantic Chunking](#step-3-semantic-chunking)
  - [Step 3b: Contextual Chunk Enrichment](#step-3b-contextual-chunk-enrichment)
  - [Step 4: Embedding and Storage](#step-4-embedding-and-storage)
  - [Step 5: Retrieval (Hybrid Search)](#step-5-retrieval-hybrid-search)
  - [Step 6: Agentic RAG Pipeline](#step-6-agentic-rag-pipeline)
  - [Step 7: API and UI](#step-7-api-and-ui)
- [Evaluation Framework](#evaluation-framework)
- [Architectural Decisions](#architectural-decisions)
- [API Reference](#api-reference)
- [Production Deployment (Railway)](#production-deployment-railway)
- [Development](#development)

---

## Architecture Overview

```
                                ┌─────────────────────────────────────────┐
                                │              User Interface             │
                                │    RTL Hebrew/English Chat (HTML/JS)    │
                                └─────────────────┬───────────────────────┘
                                                  │ POST /chat
                                                  ▼
                                ┌─────────────────────────────────────────┐
                                │            FastAPI Server               │
                                │   Conversation history · CORS · Auth    │
                                └─────────────────┬───────────────────────┘
                                                  │
                                                  ▼
               ┌──────────────────────────────────────────────────────────────────┐
               │                     LangGraph Agent (8 nodes)                    │
               │                                                                  │
               │  ┌──────────┐   ┌───────┐   ┌──────────┐   ┌───────┐           │
               │  │ Analyze  │──▶│ Route │──▶│ Retrieve │──▶│ Grade │           │
               │  │ (lang,   │   │(domain│   │(hybrid   │   │(LLM   │           │
               │  │  rewrite)│   │ detect)│   │ search)  │   │ judge) │           │
               │  └──────────┘   └───┬───┘   └──────────┘   └───┬───┘           │
               │                     │                           │                │
               │                     │ off-topic          no relevant docs        │
               │                     ▼                           ▼                │
               │              ┌──────────┐              ┌──────────────┐          │
               │              │ Fallback │◀─────────────│ Retry (max 3)│          │
               │              └──────────┘              └──────────────┘          │
               │                     ▲                          ▲                 │
               │                     │ fail            reroute/rephrase           │
               │              ┌──────┴──────┐    ┌──────────┐                    │
               │              │  Quality    │◀───│ Generate  │                    │
               │              │  Checker    │    │(+ cite)   │                    │
               │              └─────────────┘    └──────────┘                    │
               └──────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
               ┌──────────────────────────────────────────────────────────────────┐
               │                      Retrieval Layer                             │
               │                                                                  │
               │           ┌────────────────────┐                                │
               │           │ BM25 Broad Search  │  Stage 1: keyword candidates  │
               │           │ (rank-bm25, 3×k)   │                                │
               │           └─────────┬──────────┘                                │
               │                     ▼                                            │
               │           ┌────────────────────┐                                │
               │           │ Dense Reranking     │  Stage 2: semantic scoring    │
               │           │ (E5 cosine sim)     │                                │
               │           └─────────┬──────────┘                                │
               │                     ▼                                            │
               │           ┌────────────────────┐                                │
               │           │ Cross-Encoder       │                                │
               │           │ Reranker (optional) │                                │
               │           └────────────────────┘                                │
               │                                                                  │
               │   Fallback: if BM25 < 3 candidates → RRF(BM25 + Dense)         │
               └──────────────────────────────────────────────────────────────────┘
                                                  │
               ┌──────────────────────────────────────────────────────────────────┐
               │                      Data Pipeline                               │
               │                                                                  │
               │  Scrape ──▶ Parse ──▶ Chunk ──▶ Enrich ──▶ Embed ──▶ Store       │
               │  (Playwright) (Docling)  (semantic)  (Claude)  (E5-large) (ChromaDB) │
               │                                                                  │
               │  8 domains · ~350 docs · ASPX + PDF · Hebrew + English          │
               └──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM (inference)** | Gemma 3 via Ollama, Claude API, or Gemini API | Configurable at runtime via admin UI (`INFERENCE_LLM` setting). Ollama for zero-cost local inference; Claude/Gemini for faster debugging |
| **LLM (eval/preprocessing)** | Claude API (Anthropic) | High-quality LLM-as-judge for evaluation, query preprocessing |
| **LLM Tracing** | Per-query JSON traces | Every LLM call (prompt, response, timing, config) saved to `data/traces/` for debugging |
| **Vector DB** | ChromaDB (persistent, local) | Zero-infrastructure embedded vector store, no Docker needed |
| **Embeddings** | `intfloat/multilingual-e5-large` | 1024-dim, strong Hebrew+English, purpose-built query/passage separation |
| **Sparse Search** | BM25 (rank-bm25) | Catches exact terms and rare phrases that dense search misses |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking for precision at the top of the result list |
| **Agent** | LangGraph | Explicit state machine with conditional edges, retries, and fallbacks |
| **Document Processing** | Docling | Handles both PDF and ASPX HTML with table and structure extraction |
| **Web Scraping** | Playwright | Renders JavaScript-heavy ASPX pages that static crawlers miss |
| **API** | FastAPI | Async, auto-docs, Pydantic validation, static file serving |
| **UI** | Custom HTML/CSS/JS | RTL Hebrew layout, citation accordions, domain badges, mobile-responsive |
| **Evaluation** | LLM-as-judge + custom citation scorer | Matches competition scoring: relevance (65%), citations (15%), efficiency (10%), quality (10%) |

---

## Project Structure

```
apexChatbot/
├── agent/                          # LangGraph state machine
│   ├── graph.py                    # Graph construction (3 modes: RAG, agentic, combined) with trace wrappers
│   ├── state.py                    # AgentState TypedDict (incl. reasoning traces)
│   ├── edges/                      # Conditional edge logic
│   └── nodes/
│       ├── query_analyzer.py       # Language detection + query rewriting
│       ├── router.py               # Keyword pre-classification + LLM domain routing
│       ├── retriever_node.py       # Hybrid search invocation (RAG mode)
│       ├── navigate_node.py        # LLM-guided hierarchy navigation (agentic mode)
│       ├── combined_retriever_node.py  # Parallel RAG + navigator retrieval (combined mode)
│       ├── grader.py               # Per-document relevance grading with reasoning
│       ├── generator.py            # Answer generation with numbered citations + full document context
│       ├── quality_checker.py      # Self-correcting quality gate (PASS/REROUTE/REPHRASE/FAIL)
│       └── fallback.py             # Bilingual safe fallback responses
│
├── config/
│   ├── domains.py                  # 8 insurance domains with Hebrew names and URLs
│   ├── settings.py                 # Pydantic settings from .env
│   └── prompts/
│       ├── system_prompt.py        # System prompts (English + Hebrew)
│       ├── routing_prompt.py       # Few-shot domain routing + query rewrite
│       └── grading_prompt.py       # Relevance grading, hallucination check, generation
│
├── data_pipeline/
│   ├── pipeline.py                 # Orchestrator: scrape → parse → chunk → enrich → embed → store
│   ├── scraper/
│   │   ├── sitemap_crawler.py      # Discover URLs + PDFs via Playwright
│   │   ├── aspx_scraper.py         # Render and save ASPX pages
│   │   ├── pdf_downloader.py       # Async PDF download with rate limiting
│   │   └── rate_limiter.py         # Request rate limiting utilities
│   ├── parser/
│   │   ├── docling_parser.py       # PDF/HTML → structured markdown via Docling
│   │   └── metadata_extractor.py   # Language detection, domain mapping, doc classification
│   ├── chunker/
│   │   ├── semantic_chunker.py     # Header-aware splitting with overlap
│   │   └── chunk_models.py         # Chunk + ChunkMetadata Pydantic models (summary, keywords, key_facts)
│   ├── enricher/
│   │   └── contextual_enricher.py  # LLM-powered chunk enrichment (Claude API)
│   ├── embedder/
│   │   ├── embedding_model.py      # E5 multilingual wrapper with query/passage prefixing
│   │   └── batch_embedder.py       # Batch embedding with progress tracking
│   └── store/
│       ├── vector_store.py         # ChromaDB persistent client (incl. neighbor lookup)
│       └── schema.py               # Collection configuration
│
├── retrieval/
│   ├── hybrid_search.py            # Sequential cascade: BM25 → dense reranking (RRF fallback)
│   ├── bm25_search.py              # In-memory BM25 index built from ChromaDB
│   ├── retriever.py                # High-level interface: search → rerank → neighbor expand → top-k
│   ├── query_processor.py          # Hebrew normalization, query cleaning
│   ├── reranker.py                 # Cross-encoder reranking
│   └── navigator/                  # LLM-guided hierarchy navigation (agentic retrieval)
│       ├── navigator.py            # 3-level navigation: domain → document → chunks
│       ├── navigator_prompts.py    # Domain/document selection prompts
│       └── hierarchy_store.py      # Document catalog and hierarchy storage
│
├── llm/
│   ├── ollama_client.py            # Ollama wrapper (Gemma inference) with trace recording
│   ├── claude_client.py            # Anthropic Claude wrapper (eval + preprocessing) with trace recording
│   ├── gemini_client.py            # Google Gemini wrapper with trace recording
│   └── trace.py                    # Per-query LLM call tracing (thread-local collector → JSON files)
│
├── api/
│   ├── main.py                     # FastAPI app with lifespan, CORS, static mount
│   ├── dependencies.py             # Singleton resource initialization
│   ├── schemas.py                  # ChatRequest, ChatResponse, Citation, HealthResponse
│   ├── query_log.py                # SQLite query logging with trace_id linking
│   └── routes/
│       ├── chat.py                 # POST /chat — main endpoint with per-query tracing
│       ├── health.py               # GET /health — system status
│       ├── admin.py                # Admin API: settings, prompts, logs, traces, pipeline graph
│       ├── tester.py               # Automated testing API: run queries, score answers, Ex2 eval
│       └── explorer.py             # Document explorer API: browse chunks and documents
│
├── evaluation/
│   ├── ragas_eval.py               # Full evaluation harness with LLM-as-judge + keyword scoring
│   ├── baseline_eval.py            # GPT-4o / GPT-5 baseline comparison
│   ├── llm_judge.py                # Claude-based relevance + quality scoring
│   ├── citation_scorer.py          # Precision, recall, F1 for citations
│   ├── keyword_scorer.py           # Required/forbidden keyword precision + recall
│   ├── metrics.py                  # EvalResult model + weighted aggregation
│   └── dataset/
│       └── questions.json          # Sample eval questions (12, all 8 domains, keyword annotations)
│
├── ui/
│   ├── index.html                  # Full chat interface (RTL, Hebrew, Harel-branded)
│   ├── admin/                      # Admin dashboard (settings, logs, traces, prompts, pipeline graph)
│   ├── tester/                     # Automated tester UI (run queries, view results, Ex2 eval)
│   └── explorer/                   # Document explorer UI (browse indexed documents and chunks)
│
├── telegram_bot/
│   └── bot.py                      # Telegram bot integration
│
├── quizzer/                        # Automated stress testing
│   ├── runner.py                   # Test orchestrator
│   ├── api_client.py               # Chat API client for testing
│   ├── document_sampler.py         # Sample documents for question generation
│   ├── question_generator.py       # LLM-powered question generation
│   ├── question_types.py           # Question type definitions
│   ├── answer_scorer.py            # Automated answer scoring
│   └── report_generator.py         # Test report generation
│
├── scripts/
│   ├── run_pipeline.py             # CLI: python -m scripts.run_pipeline [scrape|parse|chunk|enrich|embed|all]
│   ├── run_eval.py                 # CLI: python -m scripts.run_eval
│   ├── run_baseline.py             # CLI: python -m scripts.run_baseline --model gpt-4o
│   ├── run_quizzer.py              # CLI: python -m scripts.run_quizzer [-n 50]
│   └── run_telegram.py             # CLI: python -m scripts.run_telegram
│
├── Dockerfile                      # Production Docker image (python:3.11-slim + uv)
├── entrypoint.sh                   # Docker entrypoint: volume check + uvicorn start
├── requirements-prod.txt           # Production-only dependencies (no dev/eval extras)
├── Makefile                        # make setup | pipeline | serve | eval | quiz | telegram | test | lint
├── pyproject.toml                  # Dependencies and tool config
├── docker-compose.yml              # Milvus stack (legacy, optional)
├── .env.example                    # Environment variable template
├── CLAUDE.md                       # AI assistant instructions
├── OBJECTIVE.md                    # Competition brief and scoring criteria
└── UPGRADES.md                     # Competitive analysis and upgrade plan
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- Anthropic API key (for evaluation)
- OpenAI API key (optional, for baseline comparison)

### 1. Setup

```bash
git clone https://github.com/idokazma/apexChatbot.git
cd apexChatbot

# Create virtual environment and install dependencies
make setup

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Pull the LLM model

```bash
ollama pull gemma3:12b
```

### 3. Run the data pipeline

```bash
# Full pipeline: scrape → parse → chunk → enrich → embed → store
make pipeline

# Or run individual steps
make scrape     # Crawl Harel website (takes ~30 min)
make parse      # Convert HTML/PDF to markdown
make chunk      # Split into semantic chunks
make enrich     # LLM-powered chunk enrichment (Claude API)
make embed      # Generate embeddings and store in ChromaDB
```

### 4. Start the server

```bash
make serve
# API:      http://localhost:8000
# Chat UI:  http://localhost:8000/ui
# Admin:    http://localhost:8000/admin-ui
# Tester:   http://localhost:8000/tester-ui
# Explorer: http://localhost:8000/explorer-ui
# Docs:     http://localhost:8000/docs
```

### 5. Run evaluation

```bash
# Evaluate the RAG system
make eval

# Compare against GPT-4o baseline
python -m scripts.run_baseline --model gpt-4o
```

### 6. Optional: Telegram bot

```bash
# Start the Telegram bot (requires TELEGRAM_BOT_TOKEN in .env)
make telegram
```

### 7. Optional: Automated stress testing

```bash
# Run the quizzer — generates questions from documents and scores answers
make quiz

# Smaller run (50 questions)
make quiz-small
```

---

## Walkthrough: End-to-End Pipeline

### Step 1: Data Scraping

**Files:** `data_pipeline/scraper/sitemap_crawler.py`, `aspx_scraper.py`, `pdf_downloader.py`

The scraper targets all 8 insurance domains on [harel-group.co.il](https://www.harel-group.co.il/insurance/). Harel's website uses ASP.NET (ASPX) pages with heavy JavaScript rendering, which means static HTTP requests return empty shells. We solve this with **Playwright** — a headless Chromium browser that executes JavaScript and waits for content to load.

```
Harel Website
    │
    ├── sitemap_crawler.py    Discovers all page URLs + PDF links per domain
    │   └── Uses Playwright to render each page and extract <a> links
    │
    ├── aspx_scraper.py       Downloads rendered HTML for each discovered URL
    │   └── Saves as {md5_hash}.html to avoid macOS path length limits
    │
    └── pdf_downloader.py     Downloads PDF policy documents
        └── Rate-limited (1 req/sec) with async httpx
```

Each domain produces a manifest file mapping hashed filenames back to their source URLs. This is critical for citations — every chunk must trace back to its original document.

**Why Playwright over requests/BeautifulSoup:** Harel's ASPX pages return empty HTML without JavaScript execution. Playwright renders the full page in ~2 seconds and captures the actual content.

**Why hash-based filenames:** Harel URLs contain Hebrew characters and deep paths that exceed macOS filename limits. MD5 hashing gives us safe, short, unique filenames while the manifest preserves the full URL mapping.

### Step 2: Document Parsing

**Files:** `data_pipeline/parser/docling_parser.py`, `metadata_extractor.py`

Raw HTML and PDF files are converted to structured markdown using [Docling](https://github.com/DS4SD/docling), a document understanding library that preserves:

- **Heading hierarchy** (H1 → H2 → H3)
- **Tables** (as markdown tables)
- **Lists and bullet points**
- **Page boundaries** (for PDFs)

After Docling parsing, the metadata extractor enriches each document with:

| Field | How it's detected |
|---|---|
| `language` | `langdetect` library (Hebrew vs English) |
| `domain` | URL pattern matching against the 8 registered domains |
| `doc_type` | Heuristic: PDF → "policy", FAQ page → "faq", else → "webpage" |
| `source_url` | Resolved from the manifest hash mapping |
| `section_path` | Extracted from markdown heading hierarchy ("H1 > H2 > H3") |

**Why Docling over manual parsing:** Insurance documents contain complex tables (coverage limits, exclusion lists, premium schedules) that require document understanding, not just text extraction. Docling handles both HTML tables and PDF tables consistently.

### Step 3: Semantic Chunking

**Files:** `data_pipeline/chunker/semantic_chunker.py`, `chunk_models.py`

Documents are split into overlapping chunks optimized for retrieval:

```
┌──────────────────────────────────────────────────┐
│  Document: "ביטוח רכב מקיף - תנאי הפוליסה"       │
│                                                    │
│  H1: ביטוח רכב                                    │
│  ├── H2: כיסויים                                  │
│  │   ├── Chunk 1 (480 tokens)  ──────┐            │
│  │   │   "ביטוח מקיף מכסה נזקי..."     │ overlap   │
│  │   ├── Chunk 2 (510 tokens)  ◀─────┘            │
│  │   │   "...בנוסף לכיסוי צד שלישי..."             │
│  │   └── Chunk 3 (350 tokens)                     │
│  ├── H2: חריגים                                   │
│  │   └── Chunk 4 (490 tokens)                     │
│  └── H2: תביעות                                   │
│      └── Chunk 5 (420 tokens)                     │
└──────────────────────────────────────────────────┘
```

The chunker:
1. **Splits by headers first** — respects document structure boundaries
2. **Enforces token limits** — max 512 tokens per chunk
3. **Adds overlap** — 50 tokens of overlap between adjacent chunks to preserve context at boundaries
4. **Filters noise** — removes chunks below 15 tokens (image placeholders, empty sections)
5. **Preserves section path** — each chunk knows its position in the document hierarchy ("ביטוח רכב > כיסויים > צד שלישי")

Each chunk is wrapped in a `Chunk` Pydantic model with a `ChunkMetadata` that carries full provenance (source URL, document title, section path, domain, language, page number) and a `source_doc_id` hash for neighbor chunk lookups. This metadata is what makes citations and context expansion possible.

**Why 512 tokens with 50-token overlap:** 512 tokens is the sweet spot for E5-large (the embedding model). Smaller chunks lose context; larger chunks dilute the signal. The 50-token overlap ensures that information near chunk boundaries isn't lost to retrieval.

### Step 3b: Contextual Chunk Enrichment

**Files:** `data_pipeline/enricher/contextual_enricher.py`

After chunking, each chunk is enriched using the Claude API to generate:

| Field | Purpose |
|---|---|
| `summary` | 1-2 sentence Hebrew summary of the chunk content |
| `keywords` | 5-10 searchable Hebrew keywords for BM25 boosting |
| `key_facts` | Structured facts extracted from the chunk (coverage amounts, conditions, etc.) |

Chunks are processed sequentially within each document (so the enricher has access to the previous chunk for context) and in parallel across documents. The enrichment step uses Claude for high-quality Hebrew understanding — the local Gemma model is not used here since preprocessing quality directly impacts retrieval accuracy.

**Why LLM-enriched chunks:** A winning competitor demonstrated that adding LLM-generated metadata to chunks during preprocessing significantly improves retrieval. Keywords boost BM25 matching on domain-specific terms, while summaries give the embedding model a denser semantic signal.

### Step 4: Embedding and Storage

**Files:** `data_pipeline/embedder/embedding_model.py`, `batch_embedder.py`, `data_pipeline/store/vector_store.py`

Chunks are embedded with **`intfloat/multilingual-e5-large`** (1024 dimensions) and stored in **ChromaDB**:

```
"passage: [Document: ביטוח רכב מקיף | Section: כיסויים > צד שלישי]
הביטוח מכסה נזקי גוף ורכוש שנגרמו לצד שלישי..."
                    │
                    ▼
            E5-large encoder
                    │
                    ▼
    [0.023, -0.112, 0.087, ...] (1024 floats)
                    │
                    ▼
              ChromaDB
    ┌──────────────────────────────┐
    │ id: "a3f8c2e1b9d04f67"       │
    │ embedding: [0.023, ...]      │
    │ document: "passage text..."  │
    │ metadata:                    │
    │   domain: "car"              │
    │   source_url: "https://..."  │
    │   section_path: "כיסויים..." │
    │   language: "he"             │
    └──────────────────────────────┘
```

The E5 model uses **asymmetric prefixing**: queries get `"query: "` prefix, documents get `"passage: "` prefix. This is a key design requirement of the E5 architecture that significantly improves retrieval quality.

**Why E5-large multilingual over OpenAI embeddings:** (1) Runs locally, no API cost during search. (2) Purpose-built for retrieval with query/passage separation. (3) Strong Hebrew performance (trained on multilingual data). (4) 1024 dimensions provides good quality/size tradeoff.

**Why ChromaDB over Milvus:** The project originally used Milvus (Docker-based), but we migrated to ChromaDB for simpler deployment. ChromaDB runs as an embedded database — no Docker, no etcd, no MinIO. Just a local directory. For a ~350 document corpus, this is more than sufficient.

### Step 5: Retrieval (Hybrid Search)

**Files:** `retrieval/hybrid_search.py`, `bm25_search.py`, `retriever.py`, `reranker.py`

We use a **sequential retrieval cascade** — BM25 keyword search for broad candidate retrieval, followed by dense embedding reranking for semantic precision:

```
        User Query: "מה הכיסוי לנזקי צנרת בביטוח דירה?"
                              │
                              ▼
                    ┌──────────────────┐
                    │ Stage 1: BM25    │  Broad keyword matching
                    │ (3× top_k)       │  → 30 candidates
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Stage 2: Dense   │  Cosine similarity reranking
                    │ Reranking (E5)   │  → 10 candidates
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Cross-Encoder    │  Precision reranking
                    │ Reranker         │  → top 5
                    └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Neighbor Expand  │  Prepend/append adjacent chunks
                    │ (prev + next)    │  from same document
                    └────────┬─────────┘
                              │
                              ▼
                    Top-5 documents with expanded context
```

If BM25 returns fewer than 3 candidates (e.g., for novel queries with no keyword overlap), the system falls back to parallel dense + BM25 search merged via RRF.

**Why sequential cascade over parallel fusion:** A winning competitor demonstrated that sequential BM25 → dense reranking outperforms parallel RRF fusion. BM25 first casts a wide net using exact keyword matches, then dense embeddings rerank by semantic meaning. This avoids the score-scale mismatch problem of parallel fusion.

**Why neighbor chunk expansion:** Insurance answers often span multiple chunks. When a chunk is retrieved, we fetch its adjacent chunks from the same document (via `source_doc_id` + `chunk_index`) and concatenate them as `content_expanded`. This gives the generator fuller context without increasing the retrieval search space.

**BM25 index lifecycle:** The BM25 index is built eagerly at server startup by reading all documents from ChromaDB into memory. For ~6,970 chunks, this takes a few seconds and is cached for the lifetime of the process.

### Step 6: Agentic RAG Pipeline

**Files:** `agent/graph.py`, `agent/state.py`, `agent/nodes/*.py`

The agent is a **LangGraph state machine** with three retrieval modes, configurable at runtime via the admin UI:

- **RAG mode:** Traditional hybrid search (BM25 → dense reranking → cross-encoder)
- **Agentic mode:** LLM-guided 3-level hierarchy navigation (domain → document → chunks) via the navigator
- **Combined mode:** Parallel execution of both RAG and navigator, merging results

All modes share the same analyze → route → grade → generate → quality_check pipeline. The full graph:

```
analyze → route ─── off-topic? ──── yes ───▶ fallback ──▶ END
                        │
                       no
                        │
                        ▼
                    retrieve → grade ─── relevant docs? ─── no ──▶ retry (max 3)
                                              │                        ▲
                                             yes                       │
                                              │                        │
                                              ▼                        │
                                          generate                     │
                                              │                        │
                                              ▼                        │
                                       quality_check ── reroute ───────┘
                                              │         (new domain)
                                     ┌────────┼──────── rephrase ──────┘
                                     │        │         (new query)
                                   pass      fail
                                     │        │
                                     ▼        ▼
                                    END    fallback ──▶ END
```

**Node details:**

| Node | What it does | Key design choice |
|---|---|---|
| **analyze** | Detects language (Hebrew/English) via `langdetect`, rewrites query using LLM for better retrieval | Query rewriting expands abbreviations and adds insurance terminology |
| **route** | Keyword regex matching (50+ patterns, Hebrew+English) first, LLM fallback second | Keywords are fast and deterministic; LLM handles ambiguous cases |
| **retrieve** | Sequential cascade search (BM25 → dense reranking) with neighbor expansion | Searches detected domains or all domains if none detected |
| **grade** | LLM grades each document as "yes" or "no" relevant to the query | Binary grading is more reliable than scoring with small LLMs |
| **generate** | LLM generates answer with numbered citations `[1]`, `[2]`, using expanded context | Explicit citation format in the prompt + extraction via regex |
| **quality_check** | Self-correcting quality gate with 4 actions: PASS, REROUTE, REPHRASE, FAIL | Can fix wrong-domain or weak-answer errors by retrying with corrections |
| **increment_retry** | Increments retry counter, preserves rerouted domains or rephrased query | Max 3 retries before falling back |
| **fallback** | Returns bilingual safe response with customer service contact info | Includes partial info from graded docs if available |

**Reasoning traces:** Every node appends its decision rationale to the `reasoning_trace` list in the agent state. This provides a full audit trail: why the router picked a domain, which documents the grader kept/rejected, how many citations the generator produced, and what the quality checker decided. Useful for debugging and evaluation.

**Why LangGraph over LangChain chains:** The RAG pipeline has conditional logic (retry on no relevant docs, reroute on wrong domain, rephrase on weak answer, fallback on failure). LangGraph's state machine makes these control flows explicit and debuggable. A linear chain can't express "go back to retrieve with a rephrased query after quality check suggests the answer is weak."

**Why keyword routing before LLM:** The router was the most fragile point — the LLM might say "vehicle" instead of "car" and break string matching. Keyword pre-classification with 50+ Hebrew and English patterns handles 90%+ of queries instantly without an LLM call. The LLM is only invoked for genuinely ambiguous queries.

**Citation extraction:** The generator formats documents as numbered references `[1] [Document: title | Section: path | URL: link]` and instructs the LLM to cite using `[1]`, `[2]`. Only explicitly referenced numbers are extracted as citations. If the LLM doesn't cite, we report zero citations — we don't inflate scores by falling back to all documents.

### Step 7: API and UI

**Files:** `api/main.py`, `api/routes/*.py`, `ui/`

The **FastAPI server** exposes endpoints across four route groups:

- **Chat:** `POST /chat` — Main chat endpoint with conversation history and per-query LLM tracing
- **Health:** `GET /health` — System health check (ChromaDB, Ollama, embedding model)
- **Admin:** `GET/PUT /admin/settings`, `GET /admin/prompts`, `GET /admin/logs`, `GET /admin/logs/stream` (SSE), `GET /admin/traces`, `GET /admin/traces/{trace_id}`, `GET /admin/graph`
- **Tester:** `POST /tester/run`, `GET /tester/results`, `POST /tester/ex2` — automated query testing and scoring
- **Explorer:** Browse indexed documents and chunks

The server initializes all resources once at startup (embedding model, LLM client, vector store, reranker, compiled agent graph) and shares them via a singleton `AppResources` object. LLM inference runs in a `ThreadPoolExecutor` (4 workers), so the async event loop stays responsive during long-running queries.

#### Per-Query LLM Tracing

Every query generates a unique JSON trace file in `data/traces/` capturing:
- Full system config snapshot (retrieval mode, LLM backend, models, top-k settings)
- Every LLM call: node name, full prompt, system prompt, response, temperature, max_tokens, duration
- Query metadata: original query, detected domains, confidence, final answer, reasoning trace

Traces are viewable in the admin dashboard and linkable from individual query log entries.

#### User Interfaces

The project includes four web UIs, all served as static files by FastAPI:

| UI | URL | Purpose |
|---|---|---|
| **Chat** | `/ui` | Customer-facing chat with RTL Hebrew, citations, domain badges |
| **Admin Dashboard** | `/ui/admin` | Settings management, live query logs (SSE), LLM call traces, prompt viewer, pipeline graph topology |
| **Tester** | `/ui/tester` | Run individual or batch queries, view per-query results and scores, Ex2 evaluation mode |
| **Explorer** | `/ui/explorer` | Browse indexed documents and chunks across all domains |

---

## Evaluation Framework

**Files:** `evaluation/ragas_eval.py`, `baseline_eval.py`, `llm_judge.py`, `citation_scorer.py`, `keyword_scorer.py`, `metrics.py`

The evaluation system matches the competition scoring criteria exactly:

| Metric | Weight | How we measure it |
|---|---|---|
| **Relevance** | 65% | Claude-as-judge: scores generated answer against the question and expected answer (0.0–1.0) |
| **Citation Accuracy** | 15% | Custom F1 scorer: precision (are cited sources real?) + recall (do factual claims have citations?) |
| **Efficiency** | 10% | Latency-based: <5s → 1.0, >15s → 0.0, linear interpolation between |
| **Conversational Quality** | 10% | Claude-as-judge: rates clarity, tone, structure, language correctness (0.0–1.0) |

Additionally, the **keyword scorer** (`evaluation/keyword_scorer.py`) provides a supplementary metric that checks whether answers contain required domain-specific terms (e.g., "מקיף", "גניבה" for car insurance) and flags forbidden terms (e.g., competitor names). Test questions can be annotated with `required_keywords` and `forbidden_keywords` arrays.

### Running evaluation

```bash
# Evaluate the RAG system with LLM-as-judge
make eval

# Run GPT-4o baseline (no RAG, no citations)
python -m scripts.run_baseline --model gpt-4o

# Run GPT-5 baseline
python -m scripts.run_baseline --model gpt-5
```

Reports are saved to `evaluation/reports/`:
- `eval_report.json` — Aggregated scores + per-domain breakdown
- `eval_details.json` — Per-question scores, answers, and metadata
- `baseline_gpt_4o_report.json` — GPT baseline scores for comparison

**Why LLM-as-judge over RAGAS metrics:** RAGAS answer_relevancy requires an embedding model and doesn't handle Hebrew well out of the box. Using Claude as a judge gives us a strong multilingual evaluator that can reason about answer correctness in both Hebrew and English. The scoring prompt asks for a 0.0–1.0 score with a brief justification, parsed deterministically.

### Automated Stress Testing (Quizzer)

**Files:** `quizzer/runner.py`, `api_client.py`, `document_sampler.py`, `question_generator.py`, `question_types.py`, `answer_scorer.py`, `report_generator.py`

The quizzer is an automated end-to-end stress testing system that:
1. **Samples documents** from the vector store across all domains
2. **Generates questions** using an LLM based on actual document content
3. **Sends questions** to the live chat API
4. **Scores answers** for relevance, citation accuracy, and quality
5. **Produces reports** with per-domain breakdowns and failure analysis

```bash
# Full stress test
make quiz

# Smaller run (50 questions)
make quiz-small
```

---

## Architectural Decisions

### Local inference with Gemma over API-based models

We use Gemma 3 12B (via Ollama) for all inference rather than GPT or Claude for chat:
- **Zero API cost** — the system can handle unlimited queries without billing
- **Low latency** — no network round-trip; ~2-5s per generation on Apple Silicon
- **Data privacy** — insurance data never leaves the local machine
- **Competition requirement** — the goal is to beat GPT-5 using open-source models

Claude API is used only for evaluation (LLM-as-judge) and preprocessing, not for customer-facing inference.

### ChromaDB over Milvus

The project originally used Milvus (a distributed vector database requiring Docker with etcd and MinIO). We migrated to ChromaDB because:
- **No infrastructure** — ChromaDB is an embedded database, just a directory on disk
- **Simpler deployment** — no Docker compose, no port management
- **Sufficient scale** — ~350 documents and ~5,000 chunks don't need distributed vector search
- **Same query interface** — vector search with metadata filtering works identically

For a production deployment with millions of documents, Milvus would be the right choice. For this scale, ChromaDB is simpler and faster to set up.

### Sequential cascade over parallel fusion

Insurance documents contain highly specific terms (policy clause references, Hebrew legal terms, exact coverage amounts) that dense embeddings can misrank. BM25 catches these exact matches as a broad first stage. Dense reranking then sorts by semantic relevance. This sequential cascade (BM25 → dense → cross-encoder) outperforms parallel RRF fusion by avoiding score-scale mismatches between retrieval methods.

### Binary grading over scoring

The grader node asks the LLM a yes/no question ("is this document relevant?") rather than asking for a relevance score. Small local LLMs (Gemma 12B) are unreliable at producing calibrated numerical scores, but very reliable at binary classification. The yes/no format also makes the prompt shorter and faster.

### Keyword routing as first pass

Domain routing is the most latency-sensitive node — every query passes through it. By using regex keyword matching (50+ patterns for 8 domains in Hebrew and English), we skip the LLM call entirely for 90%+ of queries. This saves ~1-2 seconds per query and is 100% deterministic.

### Self-correcting quality checker over binary hallucination check

The original system used a binary hallucination checker ("grounded" / "not grounded"). The upgraded quality checker can take 4 actions: PASS the answer, REROUTE to a different domain, REPHRASE the query for better retrieval, or FAIL to fallback. This self-correction loop (up to 3 retries) catches and fixes two common failure modes: (1) wrong domain detected by the router, and (2) weak answers that need a better-phrased search query. Each retry adds to the reasoning trace for full auditability.

### No citation fallback

When the LLM doesn't include citation markers in its answer, we report zero citations rather than attributing the answer to all source documents. This is intentional — it surfaces citation failures in evaluation rather than hiding them. The generation prompt explicitly requires `[1]`, `[2]` numbered citations to make extraction reliable.

---

## API Reference

### Chat

#### `POST /chat`

Send a message and receive a grounded answer with citations. Each query generates a trace file in `data/traces/`.

**Request:**
```json
{
  "message": "מה כולל ביטוח רכב מקיף?",
  "conversation_id": "optional-session-id",
  "language": "he"
}
```

**Response:**
```json
{
  "answer": "ביטוח רכב מקיף של הראל מכסה... [1] ... [2]",
  "citations": [
    {
      "source_url": "https://www.harel-group.co.il/insurance/car/...",
      "document_title": "ביטוח רכב מקיף - תנאי פוליסה",
      "section": "כיסויים > ביטוח מקיף",
      "relevant_text": "הביטוח מכסה נזקי גוף ורכוש..."
    }
  ],
  "domain": "car",
  "confidence": 0.9,
  "conversation_id": "abc123",
  "language": "he"
}
```

### Health

#### `GET /health`

Check system component status.

### Admin

| Endpoint | Method | Description |
|---|---|---|
| `/admin/settings` | GET | Get current system settings (LLM backend, models, retrieval mode, top-k) |
| `/admin/settings` | PUT | Update settings at runtime (hot-swappable, no restart needed) |
| `/admin/prompts` | GET | View all prompt templates used by the pipeline |
| `/admin/graph` | GET | Get pipeline graph topology as JSON (nodes + edges) |
| `/admin/logs` | GET | List recent query logs with filtering |
| `/admin/logs/stream` | GET | SSE stream of live query logs |
| `/admin/logs/{log_id}` | GET | Get full detail for a specific query log entry |
| `/admin/traces` | GET | List recent LLM call trace files |
| `/admin/traces/{trace_id}` | GET | Get full trace JSON (all LLM calls, prompts, responses, timing) |

### Tester

| Endpoint | Method | Description |
|---|---|---|
| `/tester/load-ex2` | POST | Load Ex2 evaluation questions |
| `/tester/run` | POST | Run full automated test against the chat API |
| `/tester/run-single` | POST | Run a single question by index |
| `/tester/questions` | GET | Get all questions with status |
| `/tester/stop` | POST | Stop a running test |

### Seed (Production)

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload-seed` | POST | Upload seed data (tar.gz) to the volume (requires `SEED_TOKEN`) |

---

## Production Deployment (Railway)

The application is deployed on [Railway](https://railway.com) with auto-deploy from the `main` branch.

**Live URL:** [https://apexchatbot-production.up.railway.app](https://apexchatbot-production.up.railway.app)

| Interface | URL |
|---|---|
| **Chat UI** | [/ui](https://apexchatbot-production.up.railway.app/ui) |
| **Admin Dashboard** | [/admin-ui](https://apexchatbot-production.up.railway.app/admin-ui) |
| **Tester** | [/tester-ui](https://apexchatbot-production.up.railway.app/tester-ui) |
| **Explorer** | [/explorer-ui](https://apexchatbot-production.up.railway.app/explorer-ui) |
| **API Docs** | [/docs](https://apexchatbot-production.up.railway.app/docs) |
| **Health** | [/health](https://apexchatbot-production.up.railway.app/health) |

### Infrastructure

- **Docker:** `python:3.11-slim` with `uv` for fast dependency installation
- **Volume:** Persistent Railway volume mounted at `/app/data` (ChromaDB, hierarchy data, chunks)
- **LLM Backend:** Google Gemini API (configurable via admin UI)
- **Seed Data:** Upload via `POST /api/upload-seed` (protected by `SEED_TOKEN`)

### Deployment workflow

1. Push to `main` branch — Railway auto-builds and deploys
2. The `entrypoint.sh` script checks for volume data at startup
3. BM25 index builds eagerly at startup (~6,970 chunks)
4. If volume data is missing, upload via the seed endpoint

### Environment variables (Railway)

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google Gemini API key for inference |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key for eval/preprocessing |
| `INFERENCE_LLM` | Set to `gemini` for production |
| `SEED_TOKEN` | Secret token for the `/api/upload-seed` endpoint |

### OpenAI-Compatible Endpoint

For evaluation scripts, the API exposes an OpenAI-compatible completions endpoint:

```
POST /v1/chat/completions
POST /chat/completions
```

This accepts standard OpenAI chat format and returns answers with source citations, enabling compatibility with evaluation tools like `ex2_evaluation_script`.

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# Auto-format
make format

# Run pipeline step by step
make scrape
make parse
make chunk
make embed

# Full pipeline (scrape → parse → chunk → enrich → embed)
make pipeline

# Start dev server with auto-reload
make serve

# Telegram bot
make telegram

# Automated stress testing
make quiz          # Full run
make quiz-small    # 50-question subset
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | For eval | — | Claude API key (evaluation + preprocessing) |
| `OPENAI_API_KEY` | For baseline | — | OpenAI API key (baseline comparison only) |
| `GOOGLE_API_KEY` | For Gemini | — | Google Gemini API key (optional inference backend) |
| `INFERENCE_LLM` | No | `ollama` | Inference backend: `ollama`, `claude`, or `gemini` (hot-swappable from admin UI) |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `gemma3:12b` | Local LLM model name |
| `EMBEDDING_MODEL` | No | `intfloat/multilingual-e5-large` | Sentence transformer model |
| `TELEGRAM_BOT_TOKEN` | For Telegram | — | Telegram bot token (for Telegram bot integration) |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
