"""Agentic navigator: LLM-driven top-down search through the hierarchy.

The navigator reads summaries at each level and decides where to drill
down next — like a librarian browsing the catalog, then the shelf, then
the table of contents, then the page.
"""

import json
import re

from loguru import logger

from llm.ollama_client import OllamaClient
from retrieval.navigator.hierarchy_store import HierarchyStore
from retrieval.navigator.navigator_prompts import (
    CHUNK_SELECTION_PROMPT,
    DOCUMENT_SELECTION_PROMPT,
    DOMAIN_SELECTION_PROMPT,
    SECTION_SELECTION_PROMPT,
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
    """Navigates the document hierarchy using an LLM at each level."""

    def __init__(self, store: HierarchyStore, llm: OllamaClient):
        self.store = store
        self.llm = llm

    def navigate(self, query: str, language: str = "he") -> dict:
        """Navigate the hierarchy to find relevant chunks.

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
                "navigation_path": {"domains": [], "documents": [], "sections": [], "chunks": [], "trace": trace},
                "should_fallback": True,
            }

        # Step 2: Pick documents within those domains
        selected_docs = self._select_documents(query, selected_domains, trace)
        if not selected_docs:
            trace.append("No relevant documents found — falling back")
            return {
                "retrieved_documents": [],
                "navigation_path": {"domains": selected_domains, "documents": [], "sections": [], "chunks": [], "trace": trace},
                "should_fallback": True,
            }

        # Step 3: Pick sections within those documents
        selected_sections = self._select_sections(query, selected_docs, trace)
        if not selected_sections:
            trace.append("No relevant sections found — falling back")
            return {
                "retrieved_documents": [],
                "navigation_path": {"domains": selected_domains, "documents": [d[1] for d in selected_docs], "sections": [], "chunks": [], "trace": trace},
                "should_fallback": True,
            }

        # Step 4: Pick chunks within those sections
        selected_chunk_ids = self._select_chunks(query, selected_sections, trace)
        if not selected_chunk_ids:
            # Fallback: grab all chunks from the selected sections
            trace.append("No specific chunks selected — using all chunks from selected sections")
            for domain, doc_id, section_id in selected_sections:
                try:
                    sec = self.store.load_section(domain, doc_id, section_id)
                    selected_chunk_ids.extend(sec.chunk_ids)
                except FileNotFoundError:
                    pass

        if not selected_chunk_ids:
            trace.append("No chunks available — falling back")
            return {
                "retrieved_documents": [],
                "navigation_path": {"domains": selected_domains, "documents": [d[1] for d in selected_docs], "sections": [s[2] for s in selected_sections], "chunks": [], "trace": trace},
                "should_fallback": True,
            }

        # Step 5: Load actual chunks
        chunks = self.store.load_chunks(selected_chunk_ids)
        trace.append(f"Loaded {len(chunks)} chunks")

        return {
            "retrieved_documents": chunks,
            "navigation_path": {
                "domains": selected_domains,
                "documents": [d[1] for d in selected_docs],
                "sections": [s[2] for s in selected_sections],
                "chunks": selected_chunk_ids,
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
            for d in ds.document_catalog:
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
            known_ids = {d.doc_id for d in ds.document_catalog}
            doc_ids = [d for d in doc_ids if d in known_ids]

            for doc_id in doc_ids[:3]:
                selected.append((domain, doc_id))

        trace.append(f"Step 2 — Documents: {[(d, did) for d, did in selected]}")
        logger.info(f"Navigator Step 2: selected {len(selected)} documents")
        return selected

    # ── Step 3: Section selection ──────────────────────────────────

    def _select_sections(
        self, query: str, documents: list[tuple[str, str]], trace: list[str]
    ) -> list[tuple[str, str, str]]:
        """Pick sections from documents. Returns list of (domain, doc_id, section_id)."""
        selected: list[tuple[str, str, str]] = []

        for domain, doc_id in documents:
            try:
                doc = self.store.load_document(domain, doc_id)
            except FileNotFoundError:
                trace.append(f"WARNING: document file not found for {domain}/{doc_id}")
                continue

            toc_lines = []
            for entry in doc.table_of_contents:
                line = f"- section_id: {entry.section_id} | path: {entry.section_path}"
                if entry.summary:
                    line += f"\n  Summary: {entry.summary}"
                if entry.topics:
                    line += f"\n  Topics: {', '.join(entry.topics[:4])}"
                toc_lines.append(line)
            toc_text = "\n\n".join(toc_lines)

            prompt = SECTION_SELECTION_PROMPT.format(
                query=query,
                doc_title=doc.title,
                toc_text=toc_text,
            )
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=256)
            section_ids = _parse_json_list(response)

            known_ids = {e.section_id for e in doc.table_of_contents}
            section_ids = [s for s in section_ids if s in known_ids]

            for sid in section_ids[:4]:
                selected.append((domain, doc_id, sid))

        trace.append(f"Step 3 — Sections: {[s[2] for s in selected]}")
        logger.info(f"Navigator Step 3: selected {len(selected)} sections")
        return selected

    # ── Step 4: Chunk selection ────────────────────────────────────

    def _select_chunks(
        self,
        query: str,
        sections: list[tuple[str, str, str]],
        trace: list[str],
    ) -> list[str]:
        """Pick specific chunks from sections. Returns list of chunk_ids."""
        selected: list[str] = []
        chunk_index = self.store.load_chunk_index()

        for domain, doc_id, section_id in sections:
            try:
                sec = self.store.load_section(domain, doc_id, section_id)
            except FileNotFoundError:
                trace.append(f"WARNING: section file not found for {section_id}")
                continue

            # Build chunk summaries for the prompt
            chunk_lines = []
            for cid in sec.chunk_ids:
                chunk_data = chunk_index.get(cid, {})
                content_preview = chunk_data.get("content", "")[:200]
                chunk_lines.append(f"- chunk_id: {cid}\n  Content: {content_preview}...")

            if not chunk_lines:
                continue

            chunks_text = "\n\n".join(chunk_lines)

            prompt = CHUNK_SELECTION_PROMPT.format(
                query=query,
                section_path=sec.section_path,
                doc_title=sec.source_doc_title,
                chunks_text=chunks_text,
            )
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=256)
            chunk_ids = _parse_json_list(response)

            known_ids = set(sec.chunk_ids)
            chunk_ids = [c for c in chunk_ids if c in known_ids]

            selected.extend(chunk_ids)

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for cid in selected:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)

        trace.append(f"Step 4 — Chunks: {len(deduped)} selected")
        logger.info(f"Navigator Step 4: selected {len(deduped)} chunks")
        return deduped
