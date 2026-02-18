"""Prompts for each step of the hierarchical navigation."""

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
Here are the available documents:

{documents_text}

Which documents might contain the answer?
Consider: is this about policy terms? coverage details? claims? pricing?

Respond with ONLY a JSON list of doc_id values, e.g. ["abc123", "def456"].
Pick 1-3 most relevant documents.
"""

SECTION_SELECTION_PROMPT = """A customer asks: "{query}"

I'm looking at document: "{doc_title}"
Here is the table of contents:

{toc_text}

Which sections should I read to find the answer?

Respond with ONLY a JSON list of section_id values, e.g. ["aaa111", "bbb222"].
Pick 1-4 most relevant sections.
"""

CHUNK_SELECTION_PROMPT = """A customer asks: "{query}"

I'm in section: "{section_path}" of document "{doc_title}".
Here are summaries of the individual text chunks in this section:

{chunks_text}

Which chunks contain the specific information needed to answer the question?

Respond with ONLY a JSON list of chunk_id values, e.g. ["c001", "c002"].
Pick all chunks that are relevant — typically 1-5.
"""
