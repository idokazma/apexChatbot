# Plan: Hierarchical Agentic Search ("Library Navigator")

## Core Concept

Replace vector similarity search with an **LLM that navigates a pre-built hierarchy of summaries** — like a librarian who knows the catalog, browses the right shelf, reads the table of contents, then opens the book to the right page.

**No RAG. No embeddings. No vector DB. No BM25.**
Just an LLM reading summaries at increasing levels of detail until it finds what it needs.

---

## The Hierarchy (4 Levels)

```
Level 0: Library Catalog          (1 file — fits in one prompt)
  "Car insurance covers X,Y,Z across 15 documents..."
  "Health insurance covers A,B,C across 22 documents..."

Level 1: Domain Shelf             (1 file per domain — 8 files)
  "Document 'General Terms for Car Insurance' covers: liability limits,
   deductibles, exclusions, claims process..."
  "Document 'Car FAQ' covers: how to file a claim, renewal..."

Level 2: Document TOC             (1 file per document — ~N files)
  "Section 'Coverages': collision, theft, fire, natural disaster"
  "Section 'Exclusions': DUI, racing, wear and tear"
  "Section 'Claims': filing deadlines, required documents, process"

Level 3: Section Detail           (1 file per section — many files)
  "Chunk 0: Collision coverage up to vehicle value, 10% deductible..."
  "Chunk 1: Theft — full coverage, 5% deductible, 48hr police report..."
  "Chunk 2: Fire — covered except arson by owner..."

Level 4: Raw Chunks               (already exist — the actual text)
```

Each level is a **pre-computed JSON file** with summaries designed to help the LLM decide where to drill down next.

---

## Phase 1: Hierarchy Builder (Offline Pipeline)

New module: `data_pipeline/hierarchy/`

### Step 1: Build Section Summaries (Level 3)

**File: `data_pipeline/hierarchy/section_summarizer.py`**

- Input: All chunks grouped by `(source_doc_id, section_path)`
- For each section group, call Claude API:
  - Concatenate all chunks in the section
  - Generate: a **summary** (2-3 sentences), a **topics list** (what specific questions this section can answer), and **key details** (amounts, dates, conditions)
- Output: `data/hierarchy/sections/{domain}/{doc_id}/{section_hash}.json`

```python
# Section summary schema
class SectionSummary(BaseModel):
    section_id: str              # hash of doc_id + section_path
    source_doc_id: str
    source_doc_title: str
    domain: str
    section_path: str            # "H1 > H2 > H3"
    summary: str                 # 2-3 sentences
    topics: list[str]            # "What questions can this section answer?"
    key_details: list[str]       # specific facts, amounts, conditions
    chunk_ids: list[str]         # pointers to Level 4 raw chunks
    chunk_count: int
```

### Step 2: Build Document TOCs (Level 2)

**File: `data_pipeline/hierarchy/document_summarizer.py`**

- Input: All section summaries for a document
- For each document, call Claude API:
  - Feed all section summaries
  - Generate: a **document summary** (3-5 sentences), a **table of contents** (section name + what it covers), and **document-level keywords**
- Output: `data/hierarchy/documents/{domain}/{doc_id}.json`

```python
class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    domain: str
    source_url: str
    doc_type: str                 # "policy_document", "faq", "webpage"
    language: str
    summary: str                  # 3-5 sentence overview
    table_of_contents: list[TOCEntry]  # section summaries
    key_topics: list[str]         # "What can you learn from this document?"
    total_sections: int
    total_chunks: int

class TOCEntry(BaseModel):
    section_path: str
    summary: str                  # 1-2 sentences
    topics: list[str]             # questions it can answer
    section_id: str               # pointer to Level 3
```

### Step 3: Build Domain Shelves (Level 1)

**File: `data_pipeline/hierarchy/domain_summarizer.py`**

- Input: All document summaries for a domain
- For each domain, call Claude API:
  - Feed all document summaries
  - Generate: **domain overview**, **document catalog** (title + what it covers + type), **common questions this domain handles**
- Output: `data/hierarchy/domains/{domain}.json`

```python
class DomainSummary(BaseModel):
    domain: str
    domain_he: str
    overview: str                  # 3-5 sentence overview of the domain
    document_catalog: list[CatalogEntry]
    common_topics: list[str]       # "what kinds of questions belong here?"
    total_documents: int
    total_chunks: int

class CatalogEntry(BaseModel):
    doc_id: str
    title: str
    doc_type: str
    summary: str                   # 1-2 sentences
    key_topics: list[str]
```

### Step 4: Build Library Catalog (Level 0)

**File: `data_pipeline/hierarchy/catalog_builder.py`**

- Input: All domain summaries
- Call Claude API once:
  - Feed all domain summaries
  - Generate: a **master catalog** — for each domain, what it covers, how many documents, what types of questions it handles
- Output: `data/hierarchy/catalog.json`

```python
class LibraryCatalog(BaseModel):
    domains: list[DomainOverview]
    total_documents: int
    total_domains: int
    generated_at: str

class DomainOverview(BaseModel):
    domain: str
    domain_he: str
    summary: str                   # 2-3 sentences
    handles_questions_like: list[str]  # example question types
    document_count: int
```

---

## Phase 2: Agentic Navigator (Runtime)

New module: `retrieval/navigator/`

### The Navigation Agent

**File: `retrieval/navigator/navigator.py`**

A simple loop that drills down through the hierarchy:

```
User Query
    |
    v
+-----------------------------+
|  Step 1: READ CATALOG       |  LLM reads Level 0 (library catalog)
|  "Which domains are          |  -> Picks 1-2 domains
|   relevant to this query?"   |
+-------------+---------------+
              |
              v
+-----------------------------+
|  Step 2: BROWSE SHELF       |  LLM reads Level 1 (domain summaries)
|  "Which documents in this    |  -> Picks 1-3 documents
|   domain might have the      |
|   answer?"                   |
+-------------+---------------+
              |
              v
+-----------------------------+
|  Step 3: READ TOC           |  LLM reads Level 2 (document TOC)
|  "Which sections should I    |  -> Picks 1-4 sections
|   look at?"                  |
+-------------+---------------+
              |
              v
+-----------------------------+
|  Step 4: SCAN SECTIONS      |  LLM reads Level 3 (section details)
|  "Which chunks contain the   |  -> Picks specific chunk_ids
|   information I need?"       |
+-------------+---------------+
              |
              v
+-----------------------------+
|  Step 5: READ CHUNKS        |  Load Level 4 (raw chunk text)
|  Return selected chunks      |  -> These become the "retrieved documents"
|  with full citation trail    |
+-----------------------------+
```

### Navigator State

```python
class NavigatorState(TypedDict):
    query: str
    language: str

    # Navigation path (trace of decisions)
    selected_domains: list[str]
    selected_documents: list[str]     # doc_ids
    selected_sections: list[str]      # section_ids
    selected_chunks: list[str]        # chunk_ids

    # The actual content retrieved
    retrieved_chunks: list[dict]

    # Decision reasoning (for debugging)
    navigation_trace: list[str]
```

### Navigator Prompts

Each navigation step gets a focused prompt:

**Step 1 prompt** (domain selection):
```
You are a librarian at an insurance company library.
A customer asks: "{query}"

Here is the catalog of our library:
{catalog_json}

Which domain(s) should I look in? Return a JSON list of domain names.
Think step by step: what type of insurance is this about?
```

**Step 2 prompt** (document selection):
```
The customer asks: "{query}"
I'm in the {domain} section. Here are the available documents:
{domain_shelf_json}

Which documents might contain the answer? Return a JSON list of doc_ids.
Consider: is this a policy question? A claims question? A coverage question?
```

**Step 3 prompt** (section selection):
```
The customer asks: "{query}"
I'm looking at document: "{doc_title}"
Here is the table of contents:
{document_toc_json}

Which sections should I read? Return a JSON list of section_ids.
```

**Step 4 prompt** (chunk selection):
```
The customer asks: "{query}"
I'm in section: "{section_path}"
Here are the chunk summaries:
{section_detail_json}

Which chunks contain the specific information needed? Return a JSON list of chunk_ids.
```

### Hierarchy Loader

**File: `retrieval/navigator/hierarchy_store.py`**

Simple file-based store that loads the pre-computed JSON hierarchy:

```python
class HierarchyStore:
    def __init__(self, hierarchy_dir: Path):
        self.hierarchy_dir = hierarchy_dir

    def load_catalog(self) -> dict
    def load_domain(self, domain: str) -> dict
    def load_document(self, domain: str, doc_id: str) -> dict
    def load_section(self, domain: str, doc_id: str, section_id: str) -> dict
    def load_chunks(self, chunk_ids: list[str]) -> list[dict]
```

All JSON files are loaded from disk, no database needed. Cached in memory after first load (the full hierarchy is small — just summaries).

---

## Phase 3: Integration with Existing Agent

### Replace the `retrieve` node

The existing agent graph stays mostly the same. We replace the retrieval mechanism:

**Current flow:**
```
analyze -> route -> retrieve (vector search) -> grade -> generate -> quality_check
```

**New flow:**
```
analyze -> navigate -> generate -> quality_check
```

Changes:
1. **Remove**: `route` node (the navigator handles domain selection itself at Step 1)
2. **Replace**: `retrieve` node with `navigate` node that runs the hierarchy navigator
3. **Remove**: `grade` node (the navigator already selected relevant chunks intentionally — the LLM already "graded" relevance at each hierarchy level)
4. **Keep**: `generate`, `quality_check`, `fallback`, `increment_retry` as-is

The `navigate` node returns `retrieved_documents` in the same format the current retriever does, so generation and quality checking work unchanged.

### Updated AgentState

```python
class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]
    query: str
    rewritten_query: str

    # Navigation (replaces routing + retrieval)
    detected_language: str
    navigation_path: dict         # NEW: trace of hierarchy navigation
    retrieved_documents: list[dict]

    # Generation (unchanged)
    generation: str
    citations: list[dict]

    # Control flow (unchanged)
    is_grounded: bool
    retry_count: int
    should_fallback: bool
    quality_action: str
    quality_reasoning: str
    reasoning_trace: list[str]
```

### Updated Graph

```python
def build_graph(...):
    graph = StateGraph(AgentState)

    graph.add_node("analyze", analyze_node)             # kept
    graph.add_node("navigate", navigate_node)            # NEW - replaces route+retrieve+grade
    graph.add_node("generate", generate_node)            # kept
    graph.add_node("quality_check", quality_node)        # kept
    graph.add_node("fallback", fallback)                 # kept
    graph.add_node("increment_retry", _increment_retry)  # kept

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "navigate")

    graph.add_conditional_edges("navigate", _navigate_decision,
        {"generate": "generate", "fallback": "fallback"})

    graph.add_edge("generate", "quality_check")

    graph.add_conditional_edges("quality_check", _quality_decision,
        {"end": END, "retry": "increment_retry", "fallback": "fallback"})

    graph.add_edge("increment_retry", "navigate")  # retry re-navigates
    graph.add_edge("fallback", END)
```

---

## File Structure

```
data_pipeline/hierarchy/
    __init__.py
    hierarchy_models.py        # Pydantic models for all 4 levels
    section_summarizer.py      # Level 3: section summaries from chunks
    document_summarizer.py     # Level 2: document TOCs from sections
    domain_summarizer.py       # Level 1: domain shelves from documents
    catalog_builder.py         # Level 0: master catalog from domains
    build_hierarchy.py         # Pipeline orchestrator (run all 4 steps)

retrieval/navigator/
    __init__.py
    hierarchy_store.py         # Load/cache pre-built hierarchy JSON
    navigator.py               # The agentic navigator (4-step drill-down)
    navigator_prompts.py       # Prompts for each navigation step

agent/nodes/
    navigate_node.py           # NEW: wraps navigator as a graph node

data/hierarchy/                # Generated output
    catalog.json               # Level 0
    domains/                   # Level 1
        car.json
        health.json
        ...
    documents/                 # Level 2
        car/
            {doc_id}.json
            ...
        ...
    sections/                  # Level 3
        car/
            {doc_id}/
                {section_id}.json
                ...
            ...
        ...
```

---

## Implementation Order

1. `data_pipeline/hierarchy/hierarchy_models.py` — Pydantic schemas for all 4 levels
2. `data_pipeline/hierarchy/section_summarizer.py` — Build Level 3 from existing chunks
3. `data_pipeline/hierarchy/document_summarizer.py` — Build Level 2 from Level 3
4. `data_pipeline/hierarchy/domain_summarizer.py` — Build Level 1 from Level 2
5. `data_pipeline/hierarchy/catalog_builder.py` — Build Level 0 from Level 1
6. `data_pipeline/hierarchy/build_hierarchy.py` — Orchestrate steps 2-5
7. `retrieval/navigator/hierarchy_store.py` — Load hierarchy from JSON files
8. `retrieval/navigator/navigator_prompts.py` — Prompts for each navigation step
9. `retrieval/navigator/navigator.py` — The 4-step agentic navigator
10. `agent/nodes/navigate_node.py` — Graph node wrapper
11. Update `agent/graph.py` — Wire new graph: analyze -> navigate -> generate -> quality_check
12. Update `agent/state.py` — Add `navigation_path` field

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Summary generation LLM | Claude API (offline) | High quality summaries; cost is one-time |
| Navigation LLM | Gemma via Ollama (runtime) | Fast, local, no API cost per query |
| Storage format | JSON files on disk | Simple, no infra needed, easily inspectable |
| Hierarchy depth | 4 levels | Matches natural doc structure: domain > doc > section > chunk |
| Chunks from existing pipeline | Reuse as-is | Already have good chunking with metadata; just build summaries on top |
| Grading node | Remove | Navigation inherently filters — the LLM chose these chunks for a reason |
| Routing node | Remove | Navigator handles domain selection as Step 1 |
| Quality check | Keep | Still valuable as a final sanity check on the generated answer |
