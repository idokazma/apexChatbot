"""HTTP client for sending questions to the chatbot API."""

import time

import httpx
from loguru import logger


class ChatbotAPIClient:
    """Client that sends questions to the chatbot's /chat endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        try:
            resp = self._client.get(f"{self.base_url}/health")
            data = resp.json()
            return data.get("status") in ("healthy", "degraded")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def ask(self, question: str, language: str | None = None) -> dict:
        """Send a question to the chatbot and return the full response.

        Args:
            question: The customer question text.
            language: Optional language hint ("he" or "en").

        Returns:
            Dict with keys: answer, citations, domain, confidence,
            conversation_id, language, latency_s, success.
        """
        payload: dict = {"message": question}
        if language:
            payload["language"] = language

        start = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat",
                    json=payload,
                )
                latency = time.time() - start

                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "answer": data.get("answer", ""),
                        "citations": data.get("citations", []),
                        "domain": data.get("domain"),
                        "confidence": data.get("confidence", 0.0),
                        "conversation_id": data.get("conversation_id", ""),
                        "language": data.get("language", ""),
                        "latency_s": round(latency, 2),
                        "success": True,
                    }
                else:
                    logger.warning(
                        f"API returned {resp.status_code} on attempt {attempt + 1}: "
                        f"{resp.text[:200]}"
                    )

            except httpx.TimeoutException:
                latency = time.time() - start
                logger.warning(
                    f"Request timed out after {latency:.1f}s (attempt {attempt + 1})"
                )
            except httpx.ConnectError as e:
                logger.error(f"Connection failed (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))

        latency = time.time() - start
        return {
            "answer": "",
            "citations": [],
            "domain": None,
            "confidence": 0.0,
            "conversation_id": "",
            "language": "",
            "latency_s": round(latency, 2),
            "success": False,
        }

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
