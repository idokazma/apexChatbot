"""Answer generation node with structured citation extraction."""

import re
from collections import OrderedDict

from loguru import logger

from agent.state import AgentState
from config.prompts import GENERATION_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_HE
from llm.ollama_client import OllamaClient

MAX_SOURCE_DOCUMENTS = 2


def _group_chunks_by_document(chunks: list[dict]) -> list[dict]:
    """Group chunks by their source document and merge into full documents.

    Takes a flat list of chunks (potentially from the same source document)
    and returns up to MAX_SOURCE_DOCUMENTS full document entries, each with
    all chunks concatenated in order.
    """
    # Group by source_doc_id (or source_url as fallback key)
    doc_groups: OrderedDict[str, list[dict]] = OrderedDict()
    for chunk in chunks:
        key = chunk.get("source_doc_id") or chunk.get("source_url") or chunk.get("chunk_id", "")
        if not key:
            key = id(chunk)
        if key not in doc_groups:
            doc_groups[key] = []
        doc_groups[key].append(chunk)

    # Sort chunks within each group by chunk_index
    merged_docs = []
    for doc_id, group in doc_groups.items():
        group.sort(key=lambda c: c.get("chunk_index", 0))
        # Use the first chunk's metadata as the document representative
        representative = group[0].copy()
        # Concatenate all chunk contents into full document content
        full_content = "\n\n".join(c.get("content", "") for c in group if c.get("content"))
        representative["content_expanded"] = full_content
        representative["_chunk_count"] = len(group)
        merged_docs.append(representative)

    return merged_docs[:MAX_SOURCE_DOCUMENTS]


def _format_context(documents: list[dict]) -> str:
    """Format documents as numbered context for the LLM."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source_info = []
        if doc.get("source_doc_title"):
            source_info.append(f"Document: {doc['source_doc_title']}")
        if doc.get("section_path"):
            source_info.append(f"Section: {doc['section_path']}")
        if doc.get("page_number"):
            source_info.append(f"Page: {doc['page_number']}")
        if doc.get("source_url"):
            source_info.append(f"URL: {doc['source_url']}")

        header = " | ".join(source_info) if source_info else f"Document {i}"
        # Use expanded content (full document) if available, else raw content
        content = doc.get("content_expanded") or doc.get("content", "")
        parts.append(f"[{i}] [{header}]\n{content}")

    return "\n\n---\n\n".join(parts)


def _extract_citations(text: str, documents: list[dict]) -> list[dict]:
    """Extract citation references from the generated text.

    Only returns citations that the LLM explicitly referenced via [1], [2], etc.
    Does NOT fall back to all documents — if the LLM didn't cite, we report that honestly.
    """
    citations = []
    seen_urls = set()

    # Match numbered references: [1], [2], [3], etc.
    numbered_refs = set(re.findall(r"\[(\d+)\]", text))

    for ref in numbered_refs:
        idx = int(ref) - 1
        if 0 <= idx < len(documents):
            doc = documents[idx]
            url = doc.get("source_url", "")
            key = url or f"doc-{idx}"
            if key not in seen_urls:
                seen_urls.add(key)
                citations.append({
                    "source_url": url,
                    "document_title": doc.get("source_doc_title", ""),
                    "section": doc.get("section_path", ""),
                    "relevant_text": doc.get("content", "")[:200],
                    "page_number": doc.get("page_number", 0),
                    "source_file_path": doc.get("source_file_path", ""),
                })

    # Also match [Source: title, section] / [מקור: title, section] patterns
    source_patterns = re.findall(r"\[(?:Source|מקור):\s*([^\]]+)\]", text)
    for match in source_patterns:
        # Try to match against documents by title
        for doc in documents:
            title = doc.get("source_doc_title", "")
            if title and title.lower() in match.lower():
                url = doc.get("source_url", "")
                key = url or title
                if key not in seen_urls:
                    seen_urls.add(key)
                    citations.append({
                        "source_url": url,
                        "document_title": title,
                        "section": doc.get("section_path", ""),
                        "relevant_text": doc.get("content", "")[:200],
                        "page_number": doc.get("page_number", 0),
                        "source_file_path": doc.get("source_file_path", ""),
                    })
                break

    return citations


def generator(state: AgentState, llm: OllamaClient) -> dict:
    """Generate a grounded answer with citations from retrieved documents."""
    query = state["query"]
    documents = state.get("graded_documents", [])
    language = state.get("detected_language", "he")

    if not documents:
        return {
            "generation": "",
            "citations": [],
            "should_fallback": True,
            "reasoning_trace": state.get("reasoning_trace", []) + ["Generator: no documents, falling back"],
        }

    # Group chunks by source document and take top 2 full documents
    documents = _group_chunks_by_document(documents)

    logger.info(
        f"Generator: {len(documents)} source documents "
        f"({', '.join(str(d.get('_chunk_count', 1)) + ' chunks' for d in documents)})"
    )

    # Format context
    context = _format_context(documents)

    # Build prompt
    system = SYSTEM_PROMPT_HE if language == "he" else SYSTEM_PROMPT
    prompt = GENERATION_PROMPT.format(context=context, query=query)

    # Generate answer
    logger.info(f"Generator: calling LLM ({len(prompt)} char prompt)...")
    answer = llm.generate(prompt, system_prompt=system, max_tokens=8192)
    logger.info(f"Generator: LLM returned {len(answer)} chars")

    # Extract only explicitly referenced citations
    citations = _extract_citations(answer, documents)

    logger.info(f"Generated answer ({len(answer)} chars) with {len(citations)} citations")

    return {
        "generation": answer,
        "citations": citations,
        "reasoning_trace": state.get("reasoning_trace", []) + [f"Generator: {len(citations)} citations, {len(answer)} chars"],
    }
