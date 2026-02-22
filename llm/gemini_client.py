"""Google Gemini client for fast cloud inference."""

import time

from google import genai
from google.genai import errors as genai_errors
from loguru import logger

from config.settings import settings
from llm.trace import record_call

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds (delays: 2s, 4s, 8s)


class GeminiClient:
    """Wrapper for Google Gemini API."""

    def __init__(
        self,
        api_key: str = settings.google_api_key,
        model: str = settings.gemini_model,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Generate a response from Gemini with retry on rate limits.

        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum response length.

        Returns:
            Generated text response.
        """
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if kwargs.get("response_mime_type"):
            config["response_mime_type"] = kwargs["response_mime_type"]
        if system_prompt:
            config["system_instruction"] = system_prompt

        t0 = time.time()

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"Gemini call (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"model={self.model}, prompt={len(prompt)} chars"
                )
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                result = response.text
                logger.info(
                    f"Gemini response: {len(result)} chars in {(time.time() - t0):.1f}s"
                )

                record_call(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    response=result,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    duration_ms=(time.time() - t0) * 1000,
                )

                return result
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                error_str = str(e)
                if "429" in error_str or "503" in error_str:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Gemini rate limited (attempt {attempt + 1}/{MAX_RETRIES}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"Gemini failed after {MAX_RETRIES} retries")

    def is_available(self) -> bool:
        """Check if Gemini API is reachable."""
        try:
            self.generate("Say hi", max_tokens=10)
            return True
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")
            return False
