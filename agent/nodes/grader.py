"""Document relevance grading node."""

from loguru import logger

from agent.state import AgentState
from config.prompts import RELEVANCE_GRADING_PROMPT
from llm.ollama_client import OllamaClient


def grader(state: AgentState, llm: OllamaClient) -> dict:
    """Grade each retrieved document for relevance to the query."""
    query = state["query"]
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {"graded_documents": [], "retry_count": state.get("retry_count", 0)}

    graded = []
    for doc in documents:
        prompt = RELEVANCE_GRADING_PROMPT.format(
            query=query,
            document=doc["content"][:2000],
        )
        response = llm.generate(prompt, temperature=0.0, max_tokens=16)
        response = response.strip().lower()

        if "yes" in response:
            doc["is_relevant"] = True
            graded.append(doc)
        else:
            doc["is_relevant"] = False

    logger.info(f"Graded: {len(graded)}/{len(documents)} documents relevant")

    return {
        "graded_documents": graded,
        "retry_count": state.get("retry_count", 0),
    }
