"""Rate limiting — sliding window counter with Redis-like in-memory store."""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    burst_limit: int = 10
    ban_threshold: int = 200  # ban after this many requests in window


class SlidingWindowCounter:
    """Track request counts per client using a sliding time window."""

    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def record(self, client_id: str) -> int:
        """Record a request and return the count in the current window."""
        now = time.monotonic()
        timestamps = self._requests[client_id]
        # Prune expired timestamps
        cutoff = now - self.window
        self._requests[client_id] = [t for t in timestamps if t > cutoff]
        self._requests[client_id].append(now)
        return len(self._requests[client_id])


@dataclass
class RateLimiter:
    config: RateLimitConfig = field(default_factory=RateLimitConfig)
    _counter: SlidingWindowCounter = field(default_factory=SlidingWindowCounter)
    _banned: set[str] = field(default_factory=set)

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id should be allowed."""
        if client_id in self._banned:
            return False
        count = self._counter.record(client_id)
        if count > self.config.ban_threshold:
            self._banned.add(client_id)
            return False
        return count <= self.config.requests_per_minute

    def reset(self, client_id: str) -> None:
        """Remove rate limit state for a client."""
        self._banned.discard(client_id)
        self._counter._requests.pop(client_id, None)


def handle_rate_limit(client_ip: str, limiter: RateLimiter) -> dict:
    """Process a rate-limited request. Returns response dict."""
    if not limiter.is_allowed(client_ip):
        return {"status": 429, "error": "Too Many Requests", "retry_after": 60}
    return {"status": 200, "allowed": True}
