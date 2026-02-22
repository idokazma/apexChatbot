"""Prompt templates for document relevance grading and hallucination checking."""

RELEVANCE_GRADING_PROMPT = """You are a relevance grader. Assess whether the following document is relevant \
to answering the customer's question about insurance.

Question: {query}

Document:
{document}

Is this document relevant to answering the question? Answer ONLY "yes" or "no"."""

HALLUCINATION_CHECK_PROMPT = """You are a fact-checker. Verify that every claim in the answer is supported \
by the reference information.

Reference information:
{sources}

Answer to verify:
{answer}

Check each factual claim in the answer. Is every claim supported by the reference information?
Answer ONLY "grounded" if all claims are supported, or "not_grounded" if any claim lacks support."""

GENERATION_PROMPT = """Answer the customer's insurance question based on Harel's policy information below.

Guidelines:
- Write naturally, as if you're a knowledgeable support agent having a conversation
- Be precise about coverage amounts, conditions, and exclusions
- If you don't have enough information to fully answer, let the customer know honestly
- Respond in the same language as the question
- For complex questions, use bullet points to keep things clear
- At the end of your answer, add brief references like [1] or [2] pointing to the relevant policy info

Policy information:
{context}

Customer's question: {query}

Your answer:"""
