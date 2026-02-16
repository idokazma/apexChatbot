"""Domain routing node: classifies query into insurance domain(s)."""

from loguru import logger

from agent.state import AgentState
from config.domains import DOMAIN_NAMES
from config.prompts import ROUTING_PROMPT
from llm.ollama_client import OllamaClient


def router(state: AgentState, llm: OllamaClient) -> dict:
    """Route the query to one or more insurance domains."""
    query = state.get("rewritten_query") or state["query"]

    prompt = ROUTING_PROMPT.format(query=query)
    response = llm.generate(prompt, temperature=0.0, max_tokens=64)
    response = response.strip().lower()

    # Parse domain(s) from response
    detected = []
    for domain in DOMAIN_NAMES:
        if domain in response:
            detected.append(domain)

    # Check for off-topic
    if "off_topic" in response or not detected:
        logger.info(f"Router: off-topic or unrecognized -> fallback. Response: {response}")
        return {
            "detected_domains": [],
            "should_fallback": "off_topic" in response,
        }

    logger.info(f"Router: detected domains = {detected}")
    return {
        "detected_domains": detected,
        "should_fallback": False,
    }
