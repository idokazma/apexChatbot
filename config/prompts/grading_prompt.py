"""Prompt templates for document relevance grading and hallucination checking."""

RELEVANCE_GRADING_PROMPT = """You are a relevance grader. Assess whether the following document is relevant \
to answering the customer's question about insurance.

Question: {query}

Document:
{document}

Is this document relevant to answering the question? Answer ONLY "yes" or "no"."""

HALLUCINATION_CHECK_PROMPT = """You are a fact-checker. Verify that every claim in the answer is supported \
by the source documents.

Source documents:
{sources}

Generated answer:
{answer}

Check each factual claim in the answer. Is every claim supported by the sources?
Answer ONLY "grounded" if all claims are supported, or "not_grounded" if any claim lacks support."""

GENERATION_PROMPT = """Answer the customer's question using ONLY the information in the provided documents.

Rules:
- Answer ONLY based on the provided documents
- Cite every factual claim using the document number in brackets, e.g. [1], [2]
- You MUST include at least one citation [N] for every factual statement
- If the documents don't contain enough information, say so clearly
- Respond in the same language as the question
- Be precise about coverage amounts, conditions, and exclusions
- Structure your answer with bullet points for complex questions

Documents:
{context}

Question: {query}

Answer (remember to cite with [1], [2], etc.):"""
