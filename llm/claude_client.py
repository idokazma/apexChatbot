"""Claude API client for preprocessing tasks."""

import time

import anthropic
from loguru import logger

from config.settings import settings
from llm.trace import record_call


class ClaudeClient:
    """Wrapper for Claude API used in preprocessing (chunking, metadata enrichment)."""

    def __init__(self, api_key: str = settings.anthropic_api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from Claude.

        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum response length.

        Returns:
            Generated text response.
        """
        t0 = time.time()

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        result = response.content[0].text

        record_call(
            prompt=prompt,
            system_prompt=system_prompt,
            response=result,
            temperature=temperature,
            max_tokens=max_tokens,
            duration_ms=(time.time() - t0) * 1000,
        )

        return result
