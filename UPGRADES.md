# Competitive Analysis & Upgrade Plan

Analysis of the winning APEX competition solution vs. our current implementation, with concrete upgrade proposals.

---

## Head-to-Head Comparison

### Architecture at a Glance

| Component | Winner | Ours | Verdict |
|---|---|---|---|
| **Preprocessing** | LLM-enriched chunks (summary + keywords per chunk) | Structural chunking (header-split + overlap) | They win — big |
| **Router** | LLM classifier, allows multi-domain | Keyword regex + LLM fallback, multi-domain | Comparable |
| **Retrieval** | Sequential: TF-IDF → Contextual RAG → neighbor chunks | Parallel: BM25 + Dense → RRF fusion → reranker | Different strategy, theirs may be better |
| **Generator** | Heavy prompt engineering, LLM with context | LLM with numbered context + citation extraction | Comparable |
| **Quality Check** | Can re-route domain OR rephrase question, 3 retries | Hallucination check only, retry with original query, 2 retries | They win |
| **Evaluation** | 20 manual questions + keyword precision/recall | LLM-as-judge + citation F1 | Different approach, theirs is more practical |
| **Debugging** | Explanation field on every LLM call | Loguru logging | They win |

---

## Deep Dive: What the Winner Did Differently

### 1. Contextual Chunking (Their Biggest Edge)

**What they did:**
During preprocessing, for every chunk they call an LLM to:
- Summarize the important parts of the chunk
- Extract keywords from the previous chunk (including its context)
- Store this enriched context alongside the raw chunk

Then they index both the enriched version (for RAG/semantic search) and the keywords (for TF-IDF).

**What we do:**
We split by headers, enforce token limits, and add a 50-token text overlap from the previous chunk. No LLM involvement at chunking time. Our `content_with_context` field is just a metadata header prefix (`[Document: X | Domain: Y | Section: Z]`).

**Why this matters:**
Their chunks are essentially "pre-understood" by an LLM. When the retriever searches for "what does car insurance cover", their chunks already contain a summary like "this section describes comprehensive coverage including third-party liability and collision damage" — which matches the query semantically even if the raw text uses different terminology. Our chunks only contain the raw text, so retrieval depends entirely on the embedding model bridging the vocabulary gap.

**Impact: HIGH** — This is likely the single biggest factor in their win. It improves both keyword search (extracted keywords) and semantic search (LLM summaries).

### 2. Sequential Retrieval (TF-IDF → RAG)

**What they did:**
1. First, TF-IDF search to find chunks with matching keywords
2. Then, from those results, run contextual RAG (semantic search) to rank by meaning
3. Finally, include the chunk before and after each result (neighbor expansion)

**What we do:**
We run BM25 and dense search in parallel, merge with RRF fusion, then optionally rerank with a cross-encoder.

**Key difference:** Their approach is **cascading** (narrow down, then refine), ours is **parallel** (get both perspectives, then merge). The cascading approach has an advantage: semantic search operates on a pre-filtered set of keyword-relevant chunks, so it's less likely to return semantically similar but topically irrelevant results.

**The neighbor chunk inclusion is critical:** Insurance policy answers often span multiple paragraphs. By pulling the chunk before and after each hit, they get full context even when the actual answer straddles a chunk boundary. We have 50-token overlap, but that's just a few words — not a full neighboring paragraph.

**Impact: MEDIUM-HIGH** — Neighbor chunks are a quick win. Sequential vs. parallel retrieval is a deeper architectural change.

### 3. Smarter Quality Checker (Re-route + Rephrase)

**What they did:**
Their quality checker can:
- Decide the answer doesn't match the question
- Choose a **different insurance domain** and retry from the router
- **Rephrase the user's question** to be more specific and retry from the retriever
- Do this up to 3 times

**What we do:**
Our hallucination checker only checks if the answer is "grounded" or "not_grounded". On failure, we retry with the **original query** (not rephrased) and only retry from the retriever (not the router). Max 2 retries.

**Why this matters:**
Their system is self-correcting. If the router picks "health" but the answer is about "dental", the quality checker can fix the routing. If the retriever doesn't find good docs, the question gets rephrased to match the document vocabulary better. Our system can only retry with the same query and same domains — if the initial routing or retrieval was wrong, retries won't help.

**Impact: HIGH** — This is a direct improvement to answer quality on hard questions.

### 4. Explanation Fields on Every LLM Call

**What they did:**
Every LLM call includes a field where the model explains its reasoning. For the router: "I chose car insurance because the question mentions vehicle damage." For the generator: "I based my answer on section 3.2 of the policy document."

**What we do:**
We only get the final output from each LLM call. No reasoning traces.

**Why this matters:**
- **Debugging:** When a question fails, they can read the explanation chain and immediately see where it went wrong. We have to guess from logs.
- **Quality:** Asking the LLM to explain forces it to think more carefully (chain-of-thought effect). This likely improves routing accuracy and answer quality.
- **Evaluation:** The explanations can be used to evaluate intermediate steps, not just the final answer.

**Impact: MEDIUM** — Easy to implement, helps with debugging and may improve quality.

### 5. Practical Evaluation (Keywords + Manual Review)

**What they did:**
- Small set of 20 questions with expected keyword lists
- Keyword precision/recall (does the answer contain the words that must be there?)
- Manual review of outputs to catch patterns

**What we do:**
- LLM-as-judge for relevance and quality scoring
- Citation F1 scorer
- 12 sample questions

**Key insight:** Their approach is faster to iterate with. Running 20 questions and checking if "צד שלישי" appears in the car insurance answer gives instant signal. Our LLM-as-judge is more comprehensive but slower and more expensive per iteration.

**Impact: MEDIUM** — We should add keyword-based eval as a fast feedback loop, not replace our LLM judge.

---

## Upgrade Plan

### Priority 1: Contextual Chunk Enrichment (Critical)

**The single highest-impact change.** Enrich each chunk with LLM-generated context during preprocessing.

**Implementation:**

Add a new pipeline step between chunking and embedding:

```
chunk → [NEW: contextual_enricher] → embed → store
```

For each chunk, call Claude API to generate:
```json
{
  "summary": "This section describes comprehensive car insurance coverage...",
  "keywords": ["מקיף", "צד שלישי", "תאונה", "כיסוי", "comprehensive", "collision"],
  "key_facts": ["Covers damage up to 500,000 NIS", "Excludes racing"]
}
```

Store the enriched version in `content_with_context` for semantic search. Store keywords for BM25/TF-IDF indexing.

**Where to implement:**
- New file: `data_pipeline/enricher/contextual_enricher.py`
- Modify: `data_pipeline/pipeline.py` (add step between chunk and embed)
- Modify: `data_pipeline/chunker/chunk_models.py` (add `keywords`, `summary` fields to `ChunkMetadata`)

**Cost:** ~5,000 chunks × ~200 tokens per enrichment call = ~1M tokens via Claude. About $3 on Haiku.

### Priority 2: Neighbor Chunk Retrieval (Quick Win)

**Pull the chunk before and after each retrieved chunk into the context.**

**Implementation:**

Chunks already have `chunk_index` and their parent document info. After retrieval, for each result:
1. Query ChromaDB for chunks with `chunk_index - 1` and `chunk_index + 1` from the same document
2. Prepend/append them to the context window

**Where to implement:**
- Modify: `retrieval/retriever.py` — add `_expand_with_neighbors()` method
- Modify: `data_pipeline/store/vector_store.py` — add `get_neighbors()` method that queries by document title + chunk_index ± 1

**Note:** Need to also store a `source_doc_id` (hash of source_url) in metadata to efficiently query for siblings.

### Priority 3: Self-Correcting Quality Checker (High Impact)

**Replace the binary hallucination checker with a quality checker that can re-route or rephrase.**

**Implementation:**

The quality checker should return one of:
- `"pass"` — answer is good, proceed to END
- `"reroute:<domain>"` — wrong domain, try a different one
- `"rephrase:<new_question>"` — rephrase and re-retrieve

New prompt:
```
Given the question, the detected domain, and the generated answer:
1. Does the answer actually address the question?
2. Is the domain correct?
3. If not, what domain should it be?
4. If the answer is weak, how should the question be rephrased?

Respond with one of:
- PASS
- REROUTE: <correct_domain>
- REPHRASE: <better_question>
```

**Where to implement:**
- Rewrite: `agent/nodes/hallucination_checker.py` → rename to `quality_checker.py`
- Modify: `agent/graph.py` — add conditional edges for reroute and rephrase paths
- Modify: `agent/state.py` — add `quality_check_action` field
- Increase max retries from 2 to 3

### Priority 4: Explanation Fields (Easy Win)

**Add a `reasoning` field to every LLM call.**

**Implementation:**

Modify every prompt to include:
```
Respond with a JSON object:
{
  "answer": "...",
  "reasoning": "I chose this because..."
}
```

Parse the JSON and store reasoning in the agent state for debugging.

**Where to implement:**
- Modify: `agent/nodes/router.py` — add reasoning to routing output
- Modify: `agent/nodes/grader.py` — add reasoning per document
- Modify: `agent/nodes/generator.py` — already has answer, add reasoning
- Modify: `agent/state.py` — add `reasoning_trace: list[str]` field

### Priority 5: Keyword-Based Eval (Fast Feedback Loop)

**Add keyword precision/recall as a fast evaluation method alongside LLM-as-judge.**

**Implementation:**

Extend the evaluation dataset with required keywords per question:
```json
{
  "question": "מה כולל ביטוח רכב מקיף?",
  "domain": "car",
  "required_keywords": ["מקיף", "צד שלישי", "גנבה", "נזק"],
  "forbidden_keywords": ["בריאות", "נסיעות"]
}
```

Add a keyword scorer:
- **Precision:** What fraction of answer terms are on-topic?
- **Recall:** What fraction of required keywords appear in the answer?
- **Forbidden check:** Do any forbidden terms appear (indicates wrong domain)?

**Where to implement:**
- New file: `evaluation/keyword_scorer.py`
- Modify: `evaluation/ragas_eval.py` — integrate keyword scoring
- Modify: `evaluation/dataset/questions.json` — add keyword fields

### Priority 6: Sequential Retrieval (Architectural Change)

**Switch from parallel BM25+Dense to sequential TF-IDF → semantic search.**

This is a larger change and should only be done after priorities 1-5 are validated.

**Implementation:**
1. TF-IDF search over enriched keywords → get candidate set (top 50)
2. Dense search within candidates only → rank by semantic similarity (top 10)
3. Rerank with cross-encoder → final top 5

**Where to implement:**
- Modify: `retrieval/hybrid_search.py` — change from parallel fusion to sequential cascade
- The contextual enrichment (Priority 1) must be done first for this to work well

---

## Implementation Order

```
Week 1 (Immediate):
├── Priority 1: Contextual chunk enrichment (biggest ROI)
├── Priority 2: Neighbor chunk retrieval (quick win)
└── Priority 4: Explanation fields (easy, helps debugging)

Week 2 (With evaluation):
├── Priority 5: Keyword-based eval (fast feedback)
├── Priority 3: Self-correcting quality checker
└── Iterate on prompts using keyword eval results

Week 3 (If time permits):
└── Priority 6: Sequential retrieval cascade
```

---

## Expected Impact

| Upgrade | Relevance (65%) | Citations (15%) | Efficiency (10%) | Quality (10%) |
|---|---|---|---|---|
| Contextual enrichment | +++ | + | - (slower preprocessing, same runtime) | + |
| Neighbor chunks | ++ | ++ | neutral | + |
| Quality checker reroute/rephrase | +++ | + | - (more LLM calls on retries) | ++ |
| Explanation fields | + | neutral | - (slightly more tokens) | + |
| Keyword eval | indirect (better iteration) | indirect | neutral | indirect |
| Sequential retrieval | ++ | + | neutral | + |

**Biggest gap to close:** Contextual chunk enrichment. The winner's chunks are "pre-understood" — ours are raw text. This single change likely accounts for the largest quality difference between the two systems.
