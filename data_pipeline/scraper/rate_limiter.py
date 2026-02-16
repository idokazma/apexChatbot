"""Rate limiting utilities for polite web scraping."""

import asyncio
import time


class RateLimiter:
    """Simple rate limiter with configurable delay between requests."""

    def __init__(self, delay_seconds: float = 1.5):
        self.delay = delay_seconds
        self._last_request_time = 0.0

    async def wait(self) -> None:
        """Wait until enough time has passed since the last request."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = time.monotonic()
