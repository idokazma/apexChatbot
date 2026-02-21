# LLM Call Analysis

Overview of every LLM call across the quizzer, GT evaluator, and chat API pipelines.

## Quizzer Pipeline (per question)

| Step | LLM | Calls | Purpose |
|------|-----|-------|---------|
| Question generation | Gemini | **1** | Generate question + ground truth answer from docs |
| Send to `/chat` | Ollama | **3–8** | *(see chat pipeline below)* |
| Answer scoring | Gemini | **1** | LLM-as-judge: multi-dimensional score |
| Citation scoring | — | 0 | Rule-based string matching, no LLM |
| **Total per question** | | **5–10** | |
| Report generation | Gemini | 1 | Once at end of run, not per-question |

For a 1000-question run: ~2000 Gemini calls + ~5000–8000 Ollama calls.

## GT Evaluator Pipeline (per question)

| Step | LLM | Calls | Purpose |
|------|-----|-------|---------|
| Send to `/chat` | Ollama | **3–8** | *(see chat pipeline below)* |
| Score answer correctness | Gemini | **1** | LLM-as-judge vs. human-written ground truth |
| File/page matching | — | 0 | Fuzzy string matching, no LLM |
| **Total per question** | | **4–9** | |

For 28 GT questions: ~28 Gemini calls + ~100–220 Ollama calls.

## Chat API Pipeline (per `/chat` request)

All nodes use **Ollama (Gemma, local)**.

| Node | Calls | Notes |
|------|-------|-------|
| Query Analyzer | **1** | Rewrites query for better retrieval |
| Router | **0–1** | Keyword match first; LLM only as fallback |
| Retriever (RAG) | 0 | Vector search + cross-encoder reranker, no LLM |
| Navigator (Agentic) | **2–3** | 1 domain selection + 1–2 document selections |
| Grader (RAG) | **N** | 1 call per retrieved doc (typically 3–5) |
| Generator | **1** | Generates the final answer with citations |
| Quality Checker | **1–3** | Can retry/reroute up to 3 times |
| Fallback | 0 | Static templates, no LLM |

Totals by retrieval mode:
- **RAG mode**: ~6–10 Ollama calls
- **Agentic mode**: ~5–8 Ollama calls
- **Combined mode**: both paths run in parallel, so the higher of the two

## LLM Client Summary

| Client | Model | Usage | Cost |
|--------|-------|-------|------|
| OllamaClient | Gemma (local) | All chat inference nodes | Free |
| GeminiClient | Gemini Flash 2.5 (Google API) | Quizzer + GT evaluator scoring | Pay per token |
| ClaudeClient | Claude Sonnet 4 (Anthropic API) | Data pipeline preprocessing only | Pay per token |
