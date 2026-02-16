"""Answer generation node with citation extraction."""

import re

from loguru import logger

from agent.state import AgentState
from config.prompts import GENERATION_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_HE
from llm.ollama_client import OllamaClient


def _format_context(documents: list[dict]) -> str:
    """Format retrieved documents as numbered context for the LLM."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source_info = []
        if doc.get("source_doc_title"):
            source_info.append(f"Document: {doc['source_doc_title']}")
        if doc.get("section_path"):
            source_info.append(f"Section: {doc['section_path']}")
        if doc.get("source_url"):
            source_info.append(f"URL: {doc['source_url']}")

        header = " | ".join(source_info) if source_info else f"Document {i}"
        parts.append(f"[{i}] [{header}]\n{doc['content']}")

    return "\n\n---\n\n".join(parts)


def _extract_citations(text: str, documents: list[dict]) -> list[dict]:
    """Extract citation references from the generated text."""
    citations = []
    seen_urls = set()

    # Match patterns like [Source: title, section] or [מקור: title, section]
    patterns = [
        r"\[(?:Source|מקור):\s*([^\]]+)\]",
        r"\[(\d+)\]",  # Numbered references
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Try to match numbered references to documents
            if match.isdigit():
                idx = int(match) - 1
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    url = doc.get("source_url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append({
                            "source_url": url,
                            "document_title": doc.get("source_doc_title", ""),
                            "section": doc.get("section_path", ""),
                            "relevant_text": doc.get("content", "")[:200],
                        })

    # Also add all graded documents as supporting citations if none were parsed
    if not citations:
        for doc in documents:
            url = doc.get("source_url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append({
                    "source_url": url,
                    "document_title": doc.get("source_doc_title", ""),
                    "section": doc.get("section_path", ""),
                    "relevant_text": doc.get("content", "")[:200],
                })

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
        }

    # Format context
    context = _format_context(documents)

    # Build prompt
    system = SYSTEM_PROMPT_HE if language == "he" else SYSTEM_PROMPT
    prompt = GENERATION_PROMPT.format(context=context, query=query)

    # Generate answer
    answer = llm.generate(prompt, system_prompt=system, temperature=0.1, max_tokens=2048)

    # Extract citations
    citations = _extract_citations(answer, documents)

    logger.info(f"Generated answer ({len(answer)} chars) with {len(citations)} citations")

    return {
        "generation": answer,
        "citations": citations,
    }
