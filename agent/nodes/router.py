"""Domain routing node: keyword pre-classification + LLM fallback."""

import re

from loguru import logger

from agent.state import AgentState
from config.domains import DOMAIN_NAMES, DOMAIN_NAMES_HE
from config.prompts import ROUTING_PROMPT
from llm.ollama_client import OllamaClient

# Keyword patterns per domain (English + Hebrew)
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "car": [
        r"\bcar\b", r"\bvehicle\b", r"\bauto\b", r"\bdriv", r"\bmotor\b",
        r"\bרכב\b", r"\bמכונית\b", r"\bנהיגה\b", r"\bצד.?שלישי\b", r"\bחובה\b",
        r"\bמקיף\b", r"\bתאונת.?דרכים\b",
    ],
    "life": [
        r"\blife\b", r"\bdeath\b", r"\bbeneficiar",
        r"\bחיים\b", r"\bמוות\b", r"\bמוטב", r"\bפטירה\b",
    ],
    "travel": [
        r"\btravel\b", r"\bflight\b", r"\babroad\b", r"\bvacation\b", r"\bluggage\b",
        r"\bנסיע", r"\bטיס", r"\bחו\"?ל\b", r"\bמזוודה\b", r"\bחופש", r"\bביטול.?טיסה",
    ],
    "health": [
        r"\bhealth\b", r"\bmedical\b", r"\bhospital\b", r"\bsurger", r"\bdoctor\b",
        r"\bבריאות\b", r"\bרפואי\b", r"\bבית.?חולים\b", r"\bניתוח\b", r"\bרופא\b",
        r"\bתרופ", r"\bמשלים\b",
    ],
    "dental": [
        r"\bdental\b", r"\btooth\b", r"\bteeth\b", r"\borthodont",
        r"\bשיניים\b", r"\bשן\b", r"\bאורתודנט", r"\bהשתלת.?שיניים",
    ],
    "mortgage": [
        r"\bmortgage\b", r"\bhome.?loan\b",
        r"\bמשכנתא\b", r"\bהלוואת.?דיור\b",
    ],
    "business": [
        r"\bbusiness\b", r"\bliabilit", r"\bprofessional\b", r"\bcommercial\b",
        r"\bעסק", r"\bאחריות.?מקצועית\b", r"\bעסקי\b", r"\bמסחרי\b",
    ],
    "apartment": [
        r"\bapartment\b", r"\bhome\b", r"\bhouse\b", r"\bproperty\b", r"\bcontents?\b",
        r"\bדירה\b", r"\bבית\b", r"\bמבנה\b", r"\bתכולה\b", r"\bרעידת.?אדמה\b",
        r"\bצנרת\b", r"\bנזקי.?מים\b",
    ],
}


def _keyword_route(query: str) -> list[str]:
    """Fast keyword-based domain detection (no LLM call)."""
    query_lower = query.lower()
    hits: dict[str, int] = {}
    for domain, patterns in _DOMAIN_KEYWORDS.items():
        count = sum(1 for p in patterns if re.search(p, query_lower))
        if count:
            hits[domain] = count

    if not hits:
        # Try Hebrew domain names
        for he_name, en_name in DOMAIN_NAMES_HE.items():
            if he_name in query:
                hits[en_name] = 1

    # Return domains sorted by hit count (most specific first)
    return sorted(hits, key=hits.get, reverse=True)


def router(state: AgentState, llm: OllamaClient) -> dict:
    """Route the query to insurance domain(s) using keywords + LLM fallback."""
    query = state.get("rewritten_query") or state["query"]

    # Phase 1: keyword pre-classification (fast, deterministic)
    keyword_domains = _keyword_route(query)

    if keyword_domains:
        reasoning = f"Keyword match: {keyword_domains}"
        logger.info(f"Router (keyword): detected domains = {keyword_domains}")
        return {
            "detected_domains": keyword_domains,
            "should_fallback": False,
            "reasoning_trace": state.get("reasoning_trace", []) + [f"Router: {reasoning}"],
        }

    # Phase 2: LLM classification (for ambiguous queries)
    prompt = ROUTING_PROMPT.format(query=query)
    response = llm.generate(prompt, max_tokens=8192)
    response = response.strip().lower()

    detected = []
    for domain in DOMAIN_NAMES:
        if domain in response:
            detected.append(domain)

    if "off_topic" in response or not detected:
        reasoning = f"LLM fallback: off-topic or unrecognized. Raw response: {response}"
        logger.info(f"Router (LLM): off-topic or unrecognized. Response: {response}")
        return {
            "detected_domains": [],
            "should_fallback": "off_topic" in response,
            "reasoning_trace": state.get("reasoning_trace", []) + [f"Router: {reasoning}"],
        }

    reasoning = f"LLM fallback: detected {detected}. Raw response: {response}"
    logger.info(f"Router (LLM): detected domains = {detected}")
    return {
        "detected_domains": detected,
        "should_fallback": False,
        "reasoning_trace": state.get("reasoning_trace", []) + [f"Router: {reasoning}"],
    }
