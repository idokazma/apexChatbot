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
        return {
            "graded_documents": [],
            "retry_count": state.get("retry_count", 0),
            "reasoning_trace": state.get("reasoning_trace", []) + ["Grader: 0/0 relevant (no documents)"],
        }

    graded = []
    grading_details = []
    for doc in documents:
        prompt = RELEVANCE_GRADING_PROMPT.format(
            query=query,
            document=doc["content"][:2000],
        )
        logger.info(f"Grader: calling LLM for doc '{doc.get('source_doc_title', 'untitled')}'...")
        response = llm.generate(prompt, max_tokens=8192)
        response = response.strip().lower()

        doc_title = doc.get("source_doc_title", "untitled")
        if "yes" in response:
            doc["is_relevant"] = True
            graded.append(doc)
            grading_details.append(f"  KEPT '{doc_title}': {response}")
        else:
            doc["is_relevant"] = False
            grading_details.append(f"  REJECTED '{doc_title}': {response}")

    logger.info(f"Graded: {len(graded)}/{len(documents)} documents relevant")
    for detail in grading_details:
        logger.debug(detail)

    return {
        "graded_documents": graded,
        "retry_count": state.get("retry_count", 0),
        "reasoning_trace": state.get("reasoning_trace", []) + [f"Grader: {len(graded)}/{len(documents)} relevant"],
    }
