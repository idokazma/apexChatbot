"""Hallucination checking node: verifies answer is grounded in sources."""

import re

from loguru import logger

from agent.state import AgentState
from config.prompts import HALLUCINATION_CHECK_PROMPT
from llm.ollama_client import OllamaClient


def hallucination_checker(state: AgentState, llm: OllamaClient) -> dict:
    """Check if the generated answer is grounded in the source documents."""
    answer = state.get("generation", "")
    documents = state.get("graded_documents", [])

    if not answer or not documents:
        return {"is_grounded": False}

    # Format sources
    sources = "\n\n---\n\n".join(
        f"[{doc.get('source_doc_title', 'Unknown')}]\n{doc['content'][:1500]}"
        for doc in documents
    )

    prompt = HALLUCINATION_CHECK_PROMPT.format(sources=sources, answer=answer)
    response = llm.generate(prompt, temperature=0.0, max_tokens=32)
    response_clean = response.strip().lower()

    # Structured check: accept "grounded" but reject "not_grounded" / "not grounded"
    is_grounded = bool(
        re.search(r"\bgrounded\b", response_clean)
        and not re.search(r"\bnot[_ ]grounded\b", response_clean)
    )

    logger.info(
        f"Hallucination check: {'grounded' if is_grounded else 'NOT grounded'} "
        f"(raw: {response_clean!r})"
    )
    return {"is_grounded": is_grounded}
