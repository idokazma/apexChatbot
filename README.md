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
  - [Step 4: Embedding and Storage](#step-4-embedding-and-storage)
  - [Step 5: Retrieval (Hybrid Search)](#step-5-retrieval-hybrid-search)
  - [Step 6: Agentic RAG Pipeline](#step-6-agentic-rag-pipeline)
  - [Step 7: API and UI](#step-7-api-and-ui)
- [Evaluation Framework](#evaluation-framework)
- [Architectural Decisions](#architectural-decisions)
- [API Reference](#api-reference)
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
               │                     LangGraph Agent (7 nodes)                    │
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
               │              │ Fallback │◀─────────────│ Retry (max 2)│          │
               │              └──────────┘              └──────────────┘          │
               │                     ▲                                            │
               │                     │ not grounded                               │
               │              ┌──────┴──────┐    ┌──────────┐                    │
               │              │Hallucination│◀───│ Generate  │                    │
               │              │   Check     │    │(+ cite)   │                    │
               │              └─────────────┘    └──────────┘                    │
               └──────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
               ┌──────────────────────────────────────────────────────────────────┐
               │                      Retrieval Layer                             │
               │                                                                  │
               │   ┌────────────────┐        ┌────────────────┐                  │
               │   │  Dense Search  │        │  BM25 Search   │                  │
               │   │  (ChromaDB     │        │  (rank-bm25    │                  │
               │   │   vectors)     │        │   in-memory)   │                  │
               │   └───────┬────────┘        └───────┬────────┘                  │
               │           └──────────┬──────────────┘                           │
               │                      ▼                                           │
               │           ┌────────────────────┐                                │
               │           │   RRF Fusion (k=60)│                                │
               │           └─────────┬──────────┘                                │
               │                     ▼                                            │
               │           ┌────────────────────┐                                │
               │           │ Cross-Encoder       │                                │
               │           │ Reranker (optional) │                                │
               │           └────────────────────┘                                │
               └──────────────────────────────────────────────────────────────────┘
                                                  │
               ┌──────────────────────────────────────────────────────────────────┐
               │                      Data Pipeline                               │
               │                                                                  │
               │  Scrape ──▶ Parse ──▶ Chunk ──▶ Embed ──▶ Store                 │
               │  (Playwright) (Docling)  (semantic)  (E5-large)  (ChromaDB)     │
               │                                                                  │
               │  8 domains · ~350 docs · ASPX + PDF · Hebrew + English          │
               └──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM (inference)** | Gemma 3 12B via Ollama | Runs locally, no API cost, competitive quality for policy Q&A |
| **LLM (eval/preprocessing)** | Claude API (Anthropic) | High-quality LLM-as-judge for evaluation, query preprocessing |
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
│   ├── graph.py                    # Graph construction (7 nodes, conditional edges)
│   ├── state.py                    # AgentState TypedDict
│   └── nodes/
│       ├── query_analyzer.py       # Language detection + query rewriting
│       ├── router.py               # Keyword pre-classification + LLM domain routing
│       ├── retriever_node.py       # Hybrid search invocation
│       ├── grader.py               # Per-document relevance grading
│       ├── generator.py            # Answer generation with numbered citation extraction
│       ├── hallucination_checker.py# Grounding verification
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
│   ├── pipeline.py                 # Orchestrator: scrape → parse → chunk → embed → store
│   ├── scraper/
│   │   ├── sitemap_crawler.py      # Discover URLs + PDFs via Playwright
│   │   ├── aspx_scraper.py         # Render and save ASPX pages
│   │   └── pdf_downloader.py       # Async PDF download with rate limiting
│   ├── parser/
│   │   ├── docling_parser.py       # PDF/HTML → structured markdown via Docling
│   │   └── metadata_extractor.py   # Language detection, domain mapping, doc classification
│   ├── chunker/
│   │   ├── semantic_chunker.py     # Header-aware splitting with overlap
│   │   └── chunk_models.py         # Chunk + ChunkMetadata Pydantic models
│   ├── embedder/
│   │   ├── embedding_model.py      # E5 multilingual wrapper with query/passage prefixing
│   │   └── batch_embedder.py       # Batch embedding with progress tracking
│   └── store/
│       ├── vector_store.py         # ChromaDB persistent client
│       └── schema.py               # Collection configuration
│
├── retrieval/
│   ├── hybrid_search.py            # Dense + BM25 with Reciprocal Rank Fusion
│   ├── bm25_search.py              # In-memory BM25 index built from ChromaDB
│   ├── retriever.py                # High-level interface: search → rerank → top-k
│   ├── query_processor.py          # Hebrew normalization, query cleaning
│   └── reranker.py                 # Cross-encoder reranking
│
├── llm/
│   ├── ollama_client.py            # Ollama wrapper (Gemma inference)
│   └── claude_client.py            # Anthropic Claude wrapper (eval + preprocessing)
│
├── api/
│   ├── main.py                     # FastAPI app with lifespan, CORS, static mount
│   ├── dependencies.py             # Singleton resource initialization
│   ├── schemas.py                  # ChatRequest, ChatResponse, Citation, HealthResponse
│   └── routes/
│       ├── chat.py                 # POST /chat — main endpoint
│       └── health.py               # GET /health — system status
│
├── evaluation/
│   ├── ragas_eval.py               # Full evaluation harness with LLM-as-judge
│   ├── baseline_eval.py            # GPT-4o / GPT-5 baseline comparison
│   ├── llm_judge.py                # Claude-based relevance + quality scoring
│   ├── citation_scorer.py          # Precision, recall, F1 for citations
│   ├── metrics.py                  # EvalResult model + weighted aggregation
│   └── dataset/
│       └── questions.json          # Sample eval questions (12, all 8 domains)
│
├── ui/
│   └── index.html                  # Full chat interface (RTL, Hebrew, Harel-branded)
│
├── scripts/
│   ├── run_pipeline.py             # CLI: python -m scripts.run_pipeline [scrape|parse|chunk|embed|all]
│   ├── run_eval.py                 # CLI: python -m scripts.run_eval
│   └── run_baseline.py             # CLI: python -m scripts.run_baseline --model gpt-4o
│
├── Makefile                        # make setup | pipeline | serve | eval | test | lint
├── pyproject.toml                  # Dependencies and tool config
├── docker-compose.yml              # Milvus stack (legacy, optional)
├── .env.example                    # Environment variable template
├── CLAUDE.md                       # AI assistant instructions
└── OBJECTIVE.md                    # Competition brief and scoring criteria
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
# Full pipeline: scrape → parse → chunk → embed → store
make pipeline

# Or run individual steps
make scrape     # Crawl Harel website (takes ~30 min)
make parse      # Convert HTML/PDF to markdown
make chunk      # Split into semantic chunks
make embed      # Generate embeddings and store in ChromaDB
```

### 4. Start the server

```bash
make serve
# API: http://localhost:8000
# UI:  http://localhost:8000/ui
# Docs: http://localhost:8000/docs
```

### 5. Run evaluation

```bash
# Evaluate the RAG system
make eval

# Compare against GPT-4o baseline
python -m scripts.run_baseline --model gpt-4o
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

Each chunk is wrapped in a `Chunk` Pydantic model with a `ChunkMetadata` that carries full provenance (source URL, document title, section path, domain, language, page number). This metadata is what makes citations possible.

**Why 512 tokens with 50-token overlap:** 512 tokens is the sweet spot for E5-large (the embedding model). Smaller chunks lose context; larger chunks dilute the signal. The 50-token overlap ensures that information near chunk boundaries isn't lost to retrieval.

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

We use **hybrid search** — combining dense vector similarity with sparse BM25 keyword matching, merged via **Reciprocal Rank Fusion (RRF)**:

```
        User Query: "מה הכיסוי לנזקי צנרת בביטוח דירה?"
                              │
                 ┌────────────┴────────────┐
                 ▼                          ▼
        Dense Search                  BM25 Search
        (ChromaDB vectors)            (in-memory index)
                 │                          │
                 ▼                          ▼
        Results ranked by             Results ranked by
        cosine similarity             term frequency
        [doc_A, doc_C, doc_B]         [doc_B, doc_A, doc_D]
                 │                          │
                 └────────────┬────────────┘
                              ▼
                    RRF Fusion (k=60)
                    ──────────────────
                    doc_A: 1/(60+1) + 1/(60+2) = 0.0325
                    doc_B: 1/(60+3) + 1/(60+1) = 0.0323
                    doc_C: 1/(60+2) = 0.0161
                    doc_D: 1/(60+2) = 0.0161
                              │
                              ▼
                    Cross-Encoder Reranker
                    (ms-marco-MiniLM-L-6-v2)
                              │
                              ▼
                    Top-5 documents returned
```

**Why hybrid over dense-only:** Dense embeddings are great at semantic matching ("water damage" ≈ "נזקי מים") but miss exact terms that insurance policies rely on (policy numbers, clause references, specific Hebrew legal terms). BM25 catches these exact matches. RRF merges both without needing to tune weights.

**Why RRF over weighted combination:** RRF is rank-based, not score-based. Dense scores and BM25 scores are on different scales and can't be meaningfully averaged. RRF avoids this entirely by working with ranks only. The k=60 constant is standard in the literature.

**BM25 index lifecycle:** The BM25 index is built lazily on first search by reading all documents from ChromaDB into memory. For ~350 documents, this takes a few seconds and is cached for the lifetime of the process.

### Step 6: Agentic RAG Pipeline

**Files:** `agent/graph.py`, `agent/state.py`, `agent/nodes/*.py`

The agent is a **LangGraph state machine** with 7 nodes connected by conditional edges. The full graph:

```
analyze → route ─── off-topic? ──── yes ───▶ fallback ──▶ END
                        │
                       no
                        │
                        ▼
                    retrieve → grade ─── relevant docs? ─── no ──▶ retry (max 2)
                                              │                        │
                                             yes                   ┌───┘
                                              │                    ▼
                                              ▼            increment_retry
                                          generate              │
                                              │                 ▼
                                              ▼            back to retrieve
                                       hallucination_check     (with original query)
                                              │
                                     ┌────────┴────────┐
                                  grounded         not grounded
                                     │                  │
                                     ▼                  ▼
                                    END             fallback ──▶ END
```

**Node details:**

| Node | What it does | Key design choice |
|---|---|---|
| **analyze** | Detects language (Hebrew/English) via `langdetect`, rewrites query using LLM for better retrieval | Query rewriting expands abbreviations and adds insurance terminology |
| **route** | Keyword regex matching (50+ patterns, Hebrew+English) first, LLM fallback second | Keywords are fast and deterministic; LLM handles ambiguous cases |
| **retrieve** | Hybrid search (dense + BM25) with optional domain filtering | Searches detected domains or all domains if none detected |
| **grade** | LLM grades each document as "yes" or "no" relevant to the query | Binary grading is more reliable than scoring with small LLMs |
| **generate** | LLM generates answer with numbered citations `[1]`, `[2]`, etc. | Explicit citation format in the prompt + extraction via regex |
| **hallucination_check** | LLM verifies every claim in the answer is supported by sources | Binary "grounded" / "not_grounded" check |
| **fallback** | Returns bilingual safe response with customer service contact info | Includes partial info from graded docs if available |

**Why LangGraph over LangChain chains:** The RAG pipeline has conditional logic (retry on no relevant docs, fallback on hallucination, skip retrieval for off-topic). LangGraph's state machine makes these control flows explicit and debuggable. A linear chain can't express "go back to retrieve with the original query after grading fails."

**Why keyword routing before LLM:** The router was the most fragile point — the LLM might say "vehicle" instead of "car" and break string matching. Keyword pre-classification with 50+ Hebrew and English patterns handles 90%+ of queries instantly without an LLM call. The LLM is only invoked for genuinely ambiguous queries.

**Citation extraction:** The generator formats documents as numbered references `[1] [Document: title | Section: path | URL: link]` and instructs the LLM to cite using `[1]`, `[2]`. Only explicitly referenced numbers are extracted as citations. If the LLM doesn't cite, we report zero citations — we don't inflate scores by falling back to all documents.

### Step 7: API and UI

**Files:** `api/main.py`, `api/routes/chat.py`, `ui/index.html`

The **FastAPI server** exposes two endpoints:

- `POST /chat` — Main chat endpoint with conversation history
- `GET /health` — System health check (ChromaDB, Ollama, embedding model)

The server initializes all resources once at startup (embedding model, LLM client, vector store, reranker, compiled agent graph) and shares them via a singleton `AppResources` object.

The **chat UI** at `/ui` is a single-file HTML/CSS/JS application with:
- Right-to-left (RTL) layout for Hebrew
- Harel Insurance branding (navy, gold, azure color scheme)
- Expandable citation cards showing source document, section, and excerpt
- Domain badges with emoji icons
- Confidence level indicator (high/medium/low)
- Typing animation while the agent processes
- Suggested starter questions
- Mobile-responsive design

---

## Evaluation Framework

**Files:** `evaluation/ragas_eval.py`, `baseline_eval.py`, `llm_judge.py`, `citation_scorer.py`, `metrics.py`

The evaluation system matches the competition scoring criteria exactly:

| Metric | Weight | How we measure it |
|---|---|---|
| **Relevance** | 65% | Claude-as-judge: scores generated answer against the question and expected answer (0.0–1.0) |
| **Citation Accuracy** | 15% | Custom F1 scorer: precision (are cited sources real?) + recall (do factual claims have citations?) |
| **Efficiency** | 10% | Latency-based: <5s → 1.0, >15s → 0.0, linear interpolation between |
| **Conversational Quality** | 10% | Claude-as-judge: rates clarity, tone, structure, language correctness (0.0–1.0) |

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

### Hybrid search over dense-only

Insurance documents contain highly specific terms (policy clause references, Hebrew legal terms, exact coverage amounts) that dense embeddings can misrank. BM25 catches these exact matches. The combination via RRF consistently outperforms either method alone.

### Binary grading over scoring

The grader node asks the LLM a yes/no question ("is this document relevant?") rather than asking for a relevance score. Small local LLMs (Gemma 12B) are unreliable at producing calibrated numerical scores, but very reliable at binary classification. The yes/no format also makes the prompt shorter and faster.

### Keyword routing as first pass

Domain routing is the most latency-sensitive node — every query passes through it. By using regex keyword matching (50+ patterns for 8 domains in Hebrew and English), we skip the LLM call entirely for 90%+ of queries. This saves ~1-2 seconds per query and is 100% deterministic.

### No citation fallback

When the LLM doesn't include citation markers in its answer, we report zero citations rather than attributing the answer to all source documents. This is intentional — it surfaces citation failures in evaluation rather than hiding them. The generation prompt explicitly requires `[1]`, `[2]` numbered citations to make extraction reliable.

---

## API Reference

### `POST /chat`

Send a message and receive a grounded answer with citations.

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

### `GET /health`

Check system component status.

**Response:**
```json
{
  "status": "healthy",
  "vector_db": true,
  "ollama": true,
  "embedding_model": true,
  "collection_count": 4872
}
```

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

# Start dev server with auto-reload
make serve
```

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | For eval | — | Claude API key (evaluation + preprocessing) |
| `OPENAI_API_KEY` | For baseline | — | OpenAI API key (baseline comparison only) |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | No | `gemma3:12b` | Local LLM model name |
| `EMBEDDING_MODEL` | No | `intfloat/multilingual-e5-large` | Sentence transformer model |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
