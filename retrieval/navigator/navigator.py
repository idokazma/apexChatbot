"""Agentic navigator: LLM-driven top-down search through the 3-level hierarchy.

New 3-step flow:
  1. Pick domain(s) from catalog
  2. Pick document(s) from domain shelf (using rich document cards)
  3. Load all chunks from selected documents

With good document-level summaries, the LLM can pick the right 2-3
documents accurately. Those documents have ~10-50 chunks each, which
is a manageable context for the generator.
"""

import json
import re

from loguru import logger

from llm.ollama_client import OllamaClient
from retrieval.navigator.hierarchy_store import HierarchyStore
from retrieval.navigator.navigator_prompts import (
    DOCUMENT_SELECTION_PROMPT,
    DOMAIN_SELECTION_PROMPT,
)


def _parse_json_list(text: str) -> list[str]:
    """Extract a JSON list from LLM output, tolerating extra text."""
    text = text.strip()
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Find JSON array in the text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON list from LLM output: {text[:200]}")
    return []


class Navigator:
    """Navigates the 3-level document hierarchy using an LLM at each level."""

    def __init__(self, store: HierarchyStore, llm: OllamaClient):
        self.store = store
        self.llm = llm

    def navigate(self, query: str, language: str = "he") -> dict:
        """Navigate the hierarchy to find relevant chunks.

        3-step flow: catalog → domain → document → chunks

        Args:
            query: The user's question.
            language: Detected language.

        Returns:
            Dict with:
                - retrieved_documents: list[dict] — chunks in retriever format
                - navigation_path: dict — trace of all decisions
                - should_fallback: bool — True if nothing found
        """
        trace: list[str] = []

        # Step 1: Pick domains
        selected_domains = self._select_domains(query, trace)
        if not selected_domains:
            trace.append("No relevant domains found — falling back")
            return {
                "retrieved_documents": [],
                "navigation_path": {
                    "domains": [],
                    "documents": [],
                    "chunks": [],
                    "trace": trace,
                },
                "should_fallback": True,
            }

        # Step 2: Pick documents within those domains
        selected_docs = self._select_documents(query, selected_domains, trace)
        if not selected_docs:
            trace.append("No relevant documents found — falling back")
            return {
                "retrieved_documents": [],
                "navigation_path": {
                    "domains": selected_domains,
                    "documents": [],
                    "chunks": [],
                    "trace": trace,
                },
                "should_fallback": True,
            }

        # Step 3: Load all chunks from selected documents
        all_chunk_ids = self._collect_chunks(selected_docs, trace)
        if not all_chunk_ids:
            trace.append("No chunks available — falling back")
            return {
                "retrieved_documents": [],
                "navigation_path": {
                    "domains": selected_domains,
                    "documents": [d[1] for d in selected_docs],
                    "chunks": [],
                    "trace": trace,
                },
                "should_fallback": True,
            }

        # Load actual chunks
        chunks = self.store.load_chunks(all_chunk_ids)
        trace.append(f"Loaded {len(chunks)} chunks from {len(selected_docs)} documents")

        return {
            "retrieved_documents": chunks,
            "navigation_path": {
                "domains": selected_domains,
                "documents": [d[1] for d in selected_docs],
                "chunks": all_chunk_ids,
                "trace": trace,
            },
            "should_fallback": False,
        }

    # ── Step 1: Domain selection ───────────────────────────────────

    def _select_domains(self, query: str, trace: list[str]) -> list[str]:
        """Pick 1-2 relevant domains from the catalog."""
        try:
            catalog = self.store.load_catalog()
        except FileNotFoundError:
            trace.append("ERROR: catalog.json not found")
            return []

        catalog_lines = []
        for d in catalog.domains:
            line = f"- {d.domain} ({d.domain_he}): {d.summary}"
            if d.handles_questions_like:
                line += f"\n  Example questions: {', '.join(d.handles_questions_like[:4])}"
            line += f"\n  Documents: {d.document_count}"
            catalog_lines.append(line)
        catalog_text = "\n\n".join(catalog_lines)

        prompt = DOMAIN_SELECTION_PROMPT.format(query=query, catalog_text=catalog_text)
        response = self.llm.generate(prompt, temperature=0.0, max_tokens=128)
        domains = _parse_json_list(response)

        # Validate against known domains
        known = {d.domain for d in catalog.domains}
        domains = [d for d in domains if d in known]

        trace.append(f"Step 1 — Domains: {domains}")
        logger.info(f"Navigator Step 1: selected domains {domains}")
        return domains

    # ── Step 2: Document selection ─────────────────────────────────

    def _select_documents(
        self, query: str, domains: list[str], trace: list[str]
    ) -> list[tuple[str, str]]:
        """Pick 1-3 documents per domain. Returns list of (domain, doc_id)."""
        selected: list[tuple[str, str]] = []

        for domain in domains:
            try:
                ds = self.store.load_domain(domain)
            except FileNotFoundError:
                trace.append(f"WARNING: domain file not found for '{domain}'")
                continue

            doc_lines = []
            for d in ds.all_documents:
                line = f"- doc_id: {d.doc_id} | title: {d.title} (type: {d.doc_type})"
                if d.summary:
                    line += f"\n  Summary: {d.summary}"
                if d.key_topics:
                    line += f"\n  Topics: {', '.join(d.key_topics[:5])}"
                doc_lines.append(line)
            documents_text = "\n\n".join(doc_lines)

            prompt = DOCUMENT_SELECTION_PROMPT.format(
                query=query,
                domain=domain,
                domain_he=ds.domain_he,
                documents_text=documents_text,
            )
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=256)
            doc_ids = _parse_json_list(response)

            # Validate against known doc_ids
            known_ids = {d.doc_id for d in ds.all_documents}
            doc_ids = [d for d in doc_ids if d in known_ids]

            for doc_id in doc_ids[:3]:
                selected.append((domain, doc_id))

        trace.append(f"Step 2 — Documents: {[(d, did) for d, did in selected]}")
        logger.info(f"Navigator Step 2: selected {len(selected)} documents")
        return selected

    # ── Step 3: Collect chunks from documents ──────────────────────

    def _collect_chunks(
        self, documents: list[tuple[str, str]], trace: list[str]
    ) -> list[str]:
        """Collect all chunk_ids from selected documents."""
        all_chunk_ids: list[str] = []

        for domain, doc_id in documents:
            try:
                doc = self.store.load_document(domain, doc_id)
                all_chunk_ids.extend(doc.chunk_ids)
            except FileNotFoundError:
                trace.append(f"WARNING: document card not found for {domain}/{doc_id}")

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for cid in all_chunk_ids:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)

        trace.append(f"Step 3 — Collected {len(deduped)} chunks from {len(documents)} documents")
        logger.info(f"Navigator Step 3: collected {len(deduped)} chunks")
        return deduped
