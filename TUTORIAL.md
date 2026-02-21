# Tutorial: Harel Insurance RAG Chatbot

A hands-on guide for developers joining this project. Covers how the pieces fit together, how to work with each subsystem, and how to extend the codebase.

> **Prerequisites:** Read the [README](README.md) first for setup instructions and architecture diagrams. This tutorial assumes you have the environment running (`make setup`, `.env` configured, `ollama pull gemma3:4b`).

---

## Part 1: How a Question Becomes an Answer

Let's trace a real query through the entire system to understand the data flow.

### 1.1 The Request

A user types in the chat UI or sends a POST request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "מה מכסה ביטוח רכב מקיף?"}'
```

This hits `api/routes/chat.py:chat()`, which:
1. Creates a conversation ID
2. Builds an `AgentState` dict with all fields initialized to empty/default
3. Runs `resources.agent.invoke(agent_input)` in a thread pool
4. Formats the result into a `ChatResponse` with citations

### 1.2 The Agent Graph

The agent is a LangGraph state machine defined in `agent/graph.py`. Depending on `settings.retrieval_mode`, one of three graphs is built:

| Mode | Flow | When to use |
|------|------|-------------|
| `rag` | analyze → route → retrieve → grade → generate → quality_check | Default. Classic vector search. |
| `agentic` | analyze → navigate → generate → quality_check | Hierarchy navigator (experimental). |
| `combined` | analyze → combined_retrieve → generate → quality_check | Both in parallel, merged results. |

Every node reads from and writes to `AgentState` (`agent/state.py`) — a `TypedDict` with ~20 fields tracking the query, documents, generation, and control flow.

### 1.3 Node-by-Node Walkthrough (RAG mode)

**Node 1: `analyze`** (`agent/nodes/query_analyzer.py`)
- Detects language (Hebrew/English) via `langdetect`
- Rewrites the query using the LLM to expand abbreviations and add context
- Writes: `detected_language`, `rewritten_query`

**Node 2: `route`** (`agent/nodes/router.py`)
- First tries keyword matching (50+ regex patterns, Hebrew + English)
- Falls back to LLM only if no keywords match
- Can detect multiple domains (e.g., "car and health insurance")
- If off-topic → sets `should_fallback=True` → conditional edge skips to `fallback`
- Writes: `detected_domains`, `should_fallback`

**Node 3: `retrieve`** (`agent/nodes/retriever_node.py`)
- Calls `Retriever.retrieve(query, domains)` from `retrieval/retriever.py`
- The retriever runs a sequential cascade: BM25 (broad) → dense reranking (E5) → cross-encoder → neighbor expansion
- Writes: `retrieved_documents` (list of dicts with `content`, `source_url`, `section_path`, etc.)

**Node 4: `grade`** (`agent/nodes/grader.py`)
- LLM judges each document as "yes" or "no" relevant to the query
- Filters out irrelevant documents
- If zero relevant docs and retries < 3 → conditional edge goes to `increment_retry` → back to `retrieve`
- Writes: `graded_documents`

**Node 5: `generate`** (`agent/nodes/generator.py`)
- Formats graded documents as numbered references `[1]`, `[2]`, `[3]`
- LLM generates an answer citing those numbers
- `_extract_citations()` parses `[1]`, `[2]` references from the output and maps them back to source documents
- Writes: `generation`, `citations`

**Node 6: `quality_check`** (`agent/nodes/quality_checker.py`)
- LLM evaluates: is the answer grounded in the documents?
- Four possible actions:
  - `PASS` → end (answer is good)
  - `REROUTE: health` → retry with a different domain
  - `REPHRASE: better query here` → retry with a rewritten query
  - `FAIL` → fallback
- Writes: `quality_action`, `quality_reasoning`, `is_grounded`

**Node 7: `fallback`** (`agent/nodes/fallback.py`)
- Returns a safe bilingual response with Harel's contact info (*6060)
- Includes up to 2 partially-relevant documents if available

### 1.4 The Response

Back in `chat.py`, the result dict is converted to a `ChatResponse`:

```python
{
    "answer": "ביטוח רכב מקיף של הראל מכסה... [1] ... [2]",
    "citations": [{"source_url": "...", "document_title": "...", "section": "..."}],
    "domain": "car",
    "confidence": 0.9,
    "conversation_id": "a1b2c3d4",
    "language": "he"
}
```

---

## Part 2: The Data Pipeline

Before the chatbot can answer questions, documents must be processed. The pipeline runs as:

```bash
make pipeline   # runs all steps
# or individually:
make scrape     # Step 1: crawl Harel's website
make parse      # Step 2: convert HTML/PDF → markdown
make chunk      # Step 3: split into semantic chunks
make embed      # Step 4: embed and store in ChromaDB
```

The enrichment step (Claude API) runs separately:

```bash
python -m scripts.run_pipeline enrich
```

### 2.1 Pipeline Architecture

```
data_pipeline/
├── pipeline.py              # Orchestrator — calls each step in sequence
├── scraper/                 # Step 1: Playwright-based web crawling
│   ├── sitemap_crawler.py   # Discovers URLs per domain
│   ├── aspx_scraper.py      # Renders ASPX pages with headless Chrome
│   └── pdf_downloader.py    # Downloads PDF policy documents
├── parser/                  # Step 2: Document understanding
│   ├── docling_parser.py    # PDF/HTML → structured markdown
│   └── metadata_extractor.py # Language, domain, doc_type detection
├── chunker/                 # Step 3: Intelligent splitting
│   ├── semantic_chunker.py  # Header-aware chunking with overlap
│   └── chunk_models.py      # Chunk + ChunkMetadata Pydantic models
├── enricher/                # Step 3b: LLM enrichment
│   └── contextual_enricher.py # Claude generates summaries, keywords, key_facts
├── embedder/                # Step 4: Vector embedding
│   ├── embedding_model.py   # E5 multilingual wrapper
│   └── batch_embedder.py    # Batch processing with progress tracking
└── store/                   # Step 4: Persistence
    └── vector_store.py      # ChromaDB client
```

### 2.2 The Chunk Model

Every chunk in the system is a `Chunk` with `ChunkMetadata` (defined in `data_pipeline/chunker/chunk_models.py`). Understanding this model is key:

```python
class ChunkMetadata(BaseModel):
    chunk_id: str           # Unique 16-char hex ID
    source_url: str         # Original page URL (for citations)
    source_doc_title: str   # Document title
    source_doc_id: str      # Hash of URL (for neighbor chunk lookup)
    domain: str             # "car", "health", etc.
    section_path: str       # "כיסויים > צד שלישי" (heading hierarchy)
    page_number: int | None # PDF page number
    language: str           # "he" or "en"
    chunk_index: int        # Position within the document
    total_chunks_in_doc: int
    summary: str            # LLM-generated summary (from enrichment)
    keywords: list[str]     # LLM-generated keywords (boost BM25)
    key_facts: list[str]    # Extracted structured facts

class Chunk(BaseModel):
    content: str                   # Raw chunk text
    content_with_context: str      # "[Document: ... | Domain: ... | Section: ...]\n{content}"
    metadata: ChunkMetadata
    token_count: int               # Estimated token count (words × 2)
```

The `content_with_context` field is what gets embedded — it prepends document metadata so the embedding captures not just the text but its position in the knowledge base.

### 2.3 How Enrichment Works

The contextual enricher (`data_pipeline/enricher/contextual_enricher.py`) sends each chunk to Claude with the previous chunk as context:

```
Given this chunk from document "ביטוח רכב מקיף" in the "car" domain:
[Previous chunk summary for context]

Current chunk:
"הביטוח מכסה נזקי גוף ורכוש שנגרמו לצד שלישי בתאונת דרכים..."

Generate:
1. A 1-2 sentence Hebrew summary
2. 5-10 searchable Hebrew keywords
3. Key facts (coverage amounts, conditions, etc.)
```

This enrichment dramatically improves retrieval because:
- **Keywords** boost BM25 matching on domain-specific Hebrew terms
- **Summaries** give the embedding model a denser semantic signal
- **Key facts** help the grader and generator identify specific information

---

## Part 3: The Retrieval Layer

### 3.1 Search Flow

```python
# retrieval/retriever.py — simplified
class Retriever:
    def retrieve(self, query: str, domains: list[str] | None) -> list[dict]:
        # Stage 1: BM25 keyword search (broad, 3× candidates)
        bm25_results = self.hybrid_search.bm25_search(query, domains, k=30)

        if len(bm25_results) >= 3:
            # Stage 2: Dense reranking of BM25 candidates
            reranked = self.hybrid_search.dense_rerank(query, bm25_results, k=10)
        else:
            # Fallback: parallel BM25 + dense, merged via RRF
            reranked = self.hybrid_search.rrf_search(query, domains, k=10)

        # Stage 3: Cross-encoder reranking (optional)
        if self.reranker:
            reranked = self.reranker.rerank(query, reranked, k=5)

        # Stage 4: Neighbor expansion
        for doc in reranked:
            doc["content_expanded"] = self._expand_with_neighbors(doc)

        return reranked[:self.top_k]
```

### 3.2 BM25 Index

The BM25 index (`retrieval/bm25_search.py`) is built lazily on first search:

```python
# Built once, cached in memory
index = BM25Index()
index.build(all_documents_from_chromadb)  # tokenizes and indexes all chunks

# Search returns (chunk_id, score) pairs
results = index.search("ביטוח רכב מקיף", k=30)
```

Tokenization handles both Hebrew and English: lowercases, removes punctuation, preserves Hebrew characters.

### 3.3 Neighbor Expansion

When a chunk is retrieved, we fetch its adjacent chunks from the same document using `source_doc_id` + `chunk_index`:

```
Chunk 3 retrieved → fetch Chunk 2 (prev) + Chunk 4 (next) → concatenate as content_expanded
```

This gives the generator more context without increasing the search space. The `content_expanded` field is what the generator sees instead of `content`.

---

## Part 4: Working with the Evaluation System

### 4.1 Evaluation Metrics

The system mirrors the competition scoring rubric exactly:

```python
# evaluation/metrics.py
weighted_score = (
    relevance * 0.65 +           # LLM-as-judge
    citation_accuracy * 0.15 +   # Custom F1 scorer
    efficiency * 0.10 +          # Latency-based
    conversational_quality * 0.10 # LLM-as-judge
)
```

### 4.2 Running Evaluation

```bash
# Standard evaluation (12 curated questions across all 8 domains)
make eval

# Baseline comparison
python -m scripts.run_baseline --model gpt-4o
```

The eval dataset lives in `evaluation/dataset/questions.json`. Each question has:

```json
{
    "question": "מה הכיסוי לנזקי צנרת בביטוח דירה?",
    "expected_answer": "ביטוח דירה מכסה נזקי צנרת...",
    "domain": "apartment",
    "required_keywords": ["צנרת", "נזקי מים", "כיסוי"],
    "forbidden_keywords": ["AIG", "מגדל"]
}
```

### 4.3 The Quizzer (Stress Testing)

The quizzer generates questions from actual documents and scores the chatbot at scale:

```bash
make quiz-small  # 50 questions
make quiz        # Full run (default 1000)
```

It operates in two phases:

```
Phase 1: Prepare                          Phase 2: Execute
┌────────────────────────┐               ┌────────────────────────┐
│ Sample docs from       │               │ Send each question     │
│ ChromaDB               │               │ to /chat API           │
│         │              │               │         │              │
│         ▼              │               │         ▼              │
│ Claude generates       │  save/load    │ Score each answer      │
│ realistic questions    │ ───────────▶  │ (LLM judge + metrics)  │
│ (6 types, HE/EN)      │  .json file   │         │              │
│         │              │               │         ▼              │
│         ▼              │               │ Generate HTML report   │
│ Save question set      │               │ (charts, KPIs, tables) │
└────────────────────────┘               └────────────────────────┘
```

Reports are saved to `quizzer/reports/` with timestamped filenames.

---

## Part 5: Key Patterns and Conventions

### 5.1 LLM Client Pattern

All LLM calls go through a unified interface:

```python
# llm/ollama_client.py
class OllamaClient:
    def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt and return the text response."""
        ...
```

Every agent node receives the LLM as a parameter via `functools.partial`:

```python
# agent/graph.py
grade_node = partial(grader, llm=ollama_client)
graph.add_node("grade", grade_node)
```

This makes nodes easy to test — just pass a `MagicMock()` as the LLM:

```python
def test_relevant_document_kept():
    llm = MagicMock()
    llm.generate.return_value = "yes"
    state = {"query": "car insurance", "retrieved_documents": [...], ...}
    result = grader(state, llm)
    assert len(result["graded_documents"]) == 1
```

### 5.2 State Machine Pattern

Every node follows the same contract:

```python
def my_node(state: AgentState, **deps) -> dict:
    """Read from state, do work, return ONLY the fields you want to update."""
    query = state["query"]
    # ... do work ...
    return {"generation": answer, "citations": cits}
```

LangGraph merges the returned dict into the state. You never mutate the state directly.

### 5.3 Conditional Edges

Control flow decisions are plain functions that return a string key:

```python
def _grade_decision(state: AgentState) -> str:
    if len(state.get("graded_documents", [])) >= 1:
        return "generate"       # got relevant docs → generate answer
    elif state.get("retry_count", 0) < 3:
        return "retry"          # no docs but retries left → try again
    else:
        return "fallback"       # exhausted retries → safe fallback

graph.add_conditional_edges("grade", _grade_decision, {
    "generate": "generate",
    "retry": "increment_retry",
    "fallback": "fallback",
})
```

### 5.4 Testing Without Heavy Dependencies

The test suite mocks all heavy dependencies in `tests/conftest.py`:

```python
_MOCK_MODULES = [
    "sentence_transformers", "chromadb", "langdetect",
    "playwright", "anthropic", "docling", "gradio", "tqdm",
]
```

This means you can run `make test` without Ollama, ChromaDB, or any API keys. Tests focus on pure logic: routing rules, citation extraction, scoring formulas, state machine decisions.

### 5.5 Configuration

All settings flow through `config/settings.py`, a Pydantic `BaseSettings` that reads from `.env`:

```python
from config.settings import settings

settings.ollama_model      # "gemma3:4b"
settings.retrieval_mode    # "rag", "agentic", or "combined"
settings.top_k_retrieve    # 10
settings.inference_llm     # "ollama" or "claude"
```

The 8 insurance domains are registered in `config/domains.py` as frozen dataclasses with English names, Hebrew names, base URLs, and descriptions.

---

## Part 6: Common Development Tasks

### Adding a New Agent Node

1. Create `agent/nodes/my_node.py`:

```python
def my_node(state: AgentState, llm: OllamaClient) -> dict:
    """Do something with the state."""
    # Read what you need
    query = state["query"]
    docs = state["graded_documents"]

    # Do work
    result = llm.generate(f"Summarize: {query}")

    # Return only changed fields
    return {"generation": result}
```

2. Wire it into the graph in `agent/graph.py`:

```python
my_node_fn = partial(my_node, llm=ollama_client)
graph.add_node("my_node", my_node_fn)
graph.add_edge("previous_node", "my_node")
```

3. Add tests in `tests/test_my_node.py` using a mocked LLM.

### Adding a New Insurance Domain

1. Add the domain to `config/domains.py`:

```python
DOMAINS["pet"] = InsuranceDomain(
    name="pet",
    name_he="חיות מחמד",
    base_url="https://www.harel-group.co.il/insurance/pet/",
    description="Pet insurance coverage",
)
```

2. Add keyword patterns to `agent/nodes/router.py` in the `_KEYWORD_MAP`.

3. Update `config/domains.py` constants (`DOMAIN_NAMES`, `DOMAIN_NAMES_HE`).

4. Run the pipeline for the new domain: `make pipeline`.

### Adding a New Evaluation Question

Edit `evaluation/dataset/questions.json`:

```json
{
    "question": "האם ביטוח נסיעות מכסה ביטול טיסה?",
    "expected_answer": "כן, ביטוח נסיעות של הראל מכסה ביטול טיסה...",
    "domain": "travel",
    "required_keywords": ["ביטול", "טיסה", "כיסוי"],
    "forbidden_keywords": []
}
```

Then run `make eval` to see how the system scores on the new question.

---

## Part 7: Interfaces

### Web UI

```bash
make serve
# Open http://localhost:8000/ui
```

The UI is a single HTML file (`ui/index.html`) with inline CSS and JS. It's RTL by default (Hebrew), shows domain badges, expandable citation cards, and a typing indicator.

### Telegram Bot

```bash
# Set TELEGRAM_BOT_TOKEN in .env first
make telegram
```

The bot (`telegram_bot/bot.py`) exposes `/start`, `/help`, and handles free-text messages through the same agent pipeline as the API.

### API Documentation

```bash
make serve
# Open http://localhost:8000/docs (Swagger UI)
# Or http://localhost:8000/redoc (ReDoc)
```

---

## Part 8: Debugging a Bad Answer

When the chatbot gives a wrong or incomplete answer, here's how to diagnose it:

### Step 1: Check the reasoning trace

The agent state includes a `reasoning_trace` list. Every node appends its decision:

```
["Router: keyword match → ['car']",
 "Grader: 3/5 relevant",
 "Quality: PASS — answer is grounded"]
```

Enable this in the API response by checking the full agent result (currently only `reasoning_trace` is logged, not returned to the UI).

### Step 2: Check retrieval quality

The most common failure is **bad retrieval** — the right documents weren't found. Check:
- Did the router detect the correct domain?
- Did BM25 find candidates? (Check if the query terms appear in chunk keywords)
- Did the grader keep the right documents?

### Step 3: Check generation quality

If retrieval was good but the answer is bad:
- Is the context too long? (Gemma has a limited context window)
- Did the LLM hallucinate despite grounded documents?
- Did the quality checker catch it? (Check `quality_action`)

### Step 4: Run evaluation on the specific question

Add the question to `evaluation/dataset/questions.json` with an expected answer, then:

```bash
make eval
```

This gives you a quantified score and reveals whether it's a retrieval problem (low relevance) or a citation problem (low citation_accuracy).

---

## Quick Reference

| What | Command |
|------|---------|
| Start the server | `make serve` |
| Run full pipeline | `make pipeline` |
| Run tests | `make test` |
| Run linter | `make lint` |
| Auto-format | `make format` |
| Run evaluation | `make eval` |
| Stress test (50 Qs) | `make quiz-small` |
| Start Telegram bot | `make telegram` |
| Check system health | `curl localhost:8000/health` |

| File | What it does |
|------|-------------|
| `agent/graph.py` | Agent graph construction (3 modes) |
| `agent/state.py` | AgentState TypedDict definition |
| `agent/nodes/*.py` | Individual node implementations |
| `config/settings.py` | All settings (reads `.env`) |
| `config/domains.py` | 8 insurance domain definitions |
| `retrieval/retriever.py` | High-level search interface |
| `retrieval/hybrid_search.py` | BM25 → dense cascade |
| `data_pipeline/pipeline.py` | Pipeline orchestrator |
| `api/routes/chat.py` | Main chat endpoint |
| `evaluation/metrics.py` | Scoring formula and aggregation |
| `quizzer/runner.py` | Stress test orchestrator |
