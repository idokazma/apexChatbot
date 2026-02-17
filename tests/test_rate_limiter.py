"""Tests for data_pipeline.scraper.rate_limiter."""

import asyncio
import time

import pytest

from data_pipeline.scraper.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_init_delay(self):
        limiter = RateLimiter(delay_seconds=2.0)
        assert limiter.delay == 2.0

    def test_default_delay(self):
        limiter = RateLimiter()
        assert limiter.delay == 1.5

    @pytest.mark.asyncio
    async def test_first_call_no_wait(self):
        limiter = RateLimiter(delay_seconds=0.5)
        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start
        # First call should barely wait (only if _last_request_time is 0)
        # Since monotonic starts from system boot, first call should not wait
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_second_call_waits(self):
        limiter = RateLimiter(delay_seconds=0.2)
        await limiter.wait()
        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start
        # Second call should wait approximately delay_seconds
        assert elapsed >= 0.15  # Allow small margin

    @pytest.mark.asyncio
    async def test_no_wait_after_delay_elapsed(self):
        limiter = RateLimiter(delay_seconds=0.1)
        await limiter.wait()
        await asyncio.sleep(0.15)  # Wait longer than delay
        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # Should not need to wait
