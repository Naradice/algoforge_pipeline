"""Rate limiting with jitter, header inspection, and maximum wait limits."""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Dict, Any

from research_pipeline.collectors.models import RateLimitExceededError


class RateLimiter:
    def __init__(self, max_wait_seconds: float = 120.0):
        self.max_wait_seconds = max_wait_seconds
        self._last_request_time: Dict[str, float] = {}
        self._min_intervals: Dict[str, float] = {}

    def set_min_interval(self, domain: str, interval_seconds: float) -> None:
        self._min_intervals[domain] = interval_seconds

    async def throttle(self, domain: str) -> None:
        """Enforce polite minimum interval with jitter."""
        if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("FORCE_RATE_LIMIT"):
            return

        min_interval = self._min_intervals.get(domain, 1.0)
        last_time = self._last_request_time.get(domain, 0.0)
        elapsed = time.time() - last_time

        if elapsed < min_interval:
            jitter = random.uniform(0.1, 0.5)
            wait = (min_interval - elapsed) + jitter
            await asyncio.sleep(wait)

        self._last_request_time[domain] = time.time()

    async def handle_rate_limit_headers(self, domain: str, headers: Any, status_code: int) -> None:
        """Inspect headers for rate limit constraints or retry-after."""
        # 1. Check Retry-After header
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                wait_sec = float(retry_after)
            except ValueError:
                wait_sec = 60.0

            if wait_sec > self.max_wait_seconds:
                raise RateLimitExceededError(
                    f"Retry-After ({wait_sec}s) exceeds maximum allowed wait ({self.max_wait_seconds}s)",
                    wait_seconds=wait_sec,
                )
            if not (os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("FORCE_RATE_LIMIT")):
                await asyncio.sleep(wait_sec)
            return

        # 2. Check GitHub rate limit headers
        remaining = headers.get("x-ratelimit-remaining")
        reset_time = headers.get("x-ratelimit-reset")
        if remaining is not None and int(remaining) == 0 and reset_time is not None:
            now = time.time()
            wait_sec = max(0.0, float(reset_time) - now)
            if wait_sec > self.max_wait_seconds:
                raise RateLimitExceededError(
                    f"GitHub Rate Limit exhausted. Reset wait ({wait_sec:.1f}s) exceeds maximum allowed wait ({self.max_wait_seconds}s)",
                    wait_seconds=wait_sec,
                )
            if not (os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("FORCE_RATE_LIMIT")):
                await asyncio.sleep(wait_sec)
