"""Quality checking node: verifies answer quality and can reroute or rephrase.

Replaces the binary hallucination checker with a self-correcting quality gate:
- PASS: Answer is grounded and addresses the question
- REROUTE: Wrong domain detected, try a different one
- REPHRASE: Answer is weak, rephrase the question for better retrieval
"""

import re

from loguru import logger

from agent.state import AgentState
from llm.ollama_client import OllamaClient

_QUALITY_CHECK_PROMPT = """You are a quality checker for an insurance customer support chatbot.

Given the customer's question, the detected insurance domain, and the generated answer,
determine if the answer is good enough to send to the customer.

Customer question: {query}
Detected domain: {domain}
Generated answer:
{answer}

Source documents used:
{sources}

Evaluate and respond with EXACTLY one of these (include the reasoning):

1. If the answer correctly addresses the question and is grounded in sources:
   PASS
   Reasoning: <why it's good>

2. If the answer seems to be about the wrong insurance domain:
   REROUTE: <correct_domain>
   Reasoning: <why the domain is wrong>
   (domains: car, life, travel, health, dental, mortgage, business, apartment)

3. If the answer is weak or doesn't fully address the question, suggest a better search query:
   REPHRASE: <improved_question_for_retrieval>
   Reasoning: <what's missing>

Respond with the action on the first line, then reasoning."""


def quality_checker(state: AgentState, llm: OllamaClient) -> dict:
    """Check answer quality and determine corrective action if needed."""
    answer = state.get("generation", "")
    query = state.get("query", "")
    documents = state.get("graded_documents", [])
    domains = state.get("detected_domains", [])

    if not answer or not documents:
        return {
            "is_grounded": False,
            "quality_action": "fail",
            "quality_reasoning": "No answer or documents",
        }

    # Format sources summary
    sources = "\n".join(
        f"- [{doc.get('source_doc_title', 'Unknown')}] {doc['content'][:300]}..."
        for doc in documents[:5]
    )

    domain_str = ", ".join(domains) if domains else "unknown"
    prompt = _QUALITY_CHECK_PROMPT.format(
        query=query,
        domain=domain_str,
        answer=answer,
        sources=sources,
    )

    response = llm.generate(prompt, max_tokens=8192)
    response_text = response.strip()
    first_line = response_text.split("\n")[0].strip().upper()

    # Parse reasoning (everything after the first line)
    reasoning_match = re.search(r"(?:Reasoning|reasoning)[:\s]*(.*)", response_text, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Parse action
    if first_line.startswith("PASS"):
        logger.info(f"Quality check: PASS. {reasoning[:80]}")
        return {
            "is_grounded": True,
            "quality_action": "pass",
            "quality_reasoning": reasoning,
        }

    reroute_match = re.match(r"REROUTE[:\s]+(\w+)", first_line)
    if reroute_match:
        new_domain = reroute_match.group(1).lower()
        logger.info(f"Quality check: REROUTE to '{new_domain}'. {reasoning[:80]}")
        return {
            "is_grounded": False,
            "quality_action": "reroute",
            "quality_reasoning": reasoning,
            "detected_domains": [new_domain],
        }

    rephrase_match = re.match(r"REPHRASE[:\s]+(.*)", first_line, re.IGNORECASE)
    if rephrase_match:
        new_query = rephrase_match.group(1).strip()
        # Also check second line if first line was just "REPHRASE:"
        if not new_query and len(response_text.split("\n")) > 1:
            new_query = response_text.split("\n")[1].strip()
        logger.info(f"Quality check: REPHRASE to '{new_query[:60]}'. {reasoning[:80]}")
        return {
            "is_grounded": False,
            "quality_action": "rephrase",
            "quality_reasoning": reasoning,
            "rewritten_query": new_query or query,
        }

    # Default: treat as not grounded
    logger.info(f"Quality check: FAIL (unparseable). Raw: {first_line[:60]}")
    return {
        "is_grounded": False,
        "quality_action": "fail",
        "quality_reasoning": f"Unparseable response: {first_line[:100]}",
    }
