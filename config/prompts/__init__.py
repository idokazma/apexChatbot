"""Prompt templates for the chatbot agent."""

from config.prompts.grading_prompt import (
    GENERATION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    RELEVANCE_GRADING_PROMPT,
)
from config.prompts.routing_prompt import QUERY_REWRITE_PROMPT, ROUTING_PROMPT
from config.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_HE

__all__ = [
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_HE",
    "ROUTING_PROMPT",
    "QUERY_REWRITE_PROMPT",
    "RELEVANCE_GRADING_PROMPT",
    "HALLUCINATION_CHECK_PROMPT",
    "GENERATION_PROMPT",
]
