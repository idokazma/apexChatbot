# TODO

## 1. Run enrichment step with Claude API
- Set `ANTHROPIC_API_KEY` in `.env`
- Run `python -m scripts.run_pipeline enrich` (Claude generates summaries, keywords, key_facts for all 13,910 chunks)
- Re-run `python -m scripts.run_pipeline embed` to re-embed enriched chunks into ChromaDB
- This will significantly improve retrieval quality (better BM25 keywords + richer embeddings)

## 2. Add batch-level caching to embedding pipeline
- Currently the embed step only caches after ALL batches complete (saves to `all_embeddings.pkl`)
- If the process crashes mid-way, all progress is lost and it restarts from batch 0
- All embeddings accumulate in RAM (~500-800 MB for 13,910 chunks) before writing — risk of OOM on larger datasets
- Refactor to insert into ChromaDB per batch instead of holding everything in memory:
  - Embed batch → insert to ChromaDB immediately → free memory
  - Track progress (e.g. checkpoint file) so restarts skip completed batches
  - Eliminates both the crash risk and memory bloat
