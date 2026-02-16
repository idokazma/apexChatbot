"""Prompt templates for domain routing."""

ROUTING_PROMPT = """Classify the following customer question into one or more insurance domains.

Available domains: car, life, travel, health, dental, mortgage, business, apartment

Rules:
- Return ONLY the domain name(s), comma-separated if multiple
- If the question is not about insurance, return "off_topic"
- If unclear which domain, return the most likely one(s)

Question: {query}

Domain(s):"""

QUERY_REWRITE_PROMPT = """Rewrite the following customer question to improve retrieval from an insurance knowledge base.

Rules:
- Expand abbreviations
- Add relevant insurance terminology
- Keep the original language (Hebrew or English)
- Make the query more specific if it's vague
- Preserve the original intent

Original question: {query}
Conversation context: {context}

Rewritten question:"""
