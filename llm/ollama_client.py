"""Ollama client for local LLM inference with Gemma."""

from loguru import logger
from ollama import Client

from config.settings import settings


class OllamaClient:
    """Wrapper for Ollama API for Gemma inference."""

    def __init__(
        self,
        host: str = settings.ollama_host,
        model: str = settings.ollama_model,
    ):
        self.client = Client(host=host)
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the local model.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum response length.

        Returns:
            Generated text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        return response["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """Stream a response from the local model.

        Yields:
            Text chunks as they are generated.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        for chunk in stream:
            yield chunk["message"]["content"]

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models = self.client.list()
            available = [m["name"] for m in models.get("models", [])]
            return any(self.model in name for name in available)
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
