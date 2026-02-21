"""Prompts for each step of the 3-level hierarchical navigation."""

DOMAIN_SELECTION_PROMPT = """You are a librarian at an insurance company knowledge base.
A customer asks: "{query}"

Here is the catalog of our library — each domain covers a different type of insurance:

{catalog_text}

Which domain(s) should I search in to answer this question?
Think step by step about what type of insurance this is about.

Respond with ONLY a JSON list of domain names, e.g. ["car", "health"].
Pick 1-2 domains maximum. If the question is clearly off-topic (not about insurance), respond with [].
"""

DOCUMENT_SELECTION_PROMPT = """A customer asks: "{query}"

I'm in the "{domain}" ({domain_he}) insurance section.
Here are the available documents with their summaries:

{documents_text}

Which documents might contain the answer?
Read each document summary carefully — they describe exactly what information each document contains.
Consider: is this about policy terms? coverage details? claims? pricing? FAQ?

Respond with ONLY a JSON list of doc_id values, e.g. ["abc123", "def456"].
Pick 1-3 most relevant documents.
"""
