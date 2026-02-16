"""Claude API client for preprocessing tasks."""

import anthropic
from loguru import logger

from config.settings import settings


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
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text
