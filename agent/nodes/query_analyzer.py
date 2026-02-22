"""Query analysis node: language detection, intent extraction, query rewriting."""

from langdetect import detect
from loguru import logger

from agent.state import AgentState
from config.prompts import QUERY_REWRITE_PROMPT
from llm.ollama_client import OllamaClient


def detect_language(text: str) -> str:
    """Detect query language."""
    try:
        lang = detect(text)
        return "he" if lang == "he" else "en"
    except Exception:
        return "he"


def query_analyzer(state: AgentState, llm: OllamaClient) -> dict:
    """Analyze the query: detect language, rewrite for better retrieval."""
    query = state["query"]

    # Detect language
    language = detect_language(query)

    # Build context from conversation history
    context = ""
    if state.get("messages") and len(state["messages"]) > 1:
        recent = state["messages"][-4:]  # Last 2 turns
        context = "\n".join(
            f"{m.type}: {m.content}" for m in recent if hasattr(m, "content")
        )

    # Rewrite query for better retrieval
    rewrite_prompt = QUERY_REWRITE_PROMPT.format(query=query, context=context or "None")
    rewritten = llm.generate(rewrite_prompt, max_tokens=8192)
    rewritten = rewritten.strip()

    logger.info(f"Query analyzed: lang={language}, rewritten='{rewritten[:80]}...'")

    return {
        "detected_language": language,
        "rewritten_query": rewritten,
    }
