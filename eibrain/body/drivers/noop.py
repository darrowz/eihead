"""No-op driver implementation."""

from __future__ import annotations

import time

from .base import DriverResult


class NoopDriver:
    def heartbeat(self) -> DriverResult:
        started = time.perf_counter()
        return DriverResult(
            status="healthy",
            details={"driver": "noop", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
        )

    def invoke(self, operation: str, payload: dict[str, object]) -> DriverResult:
        started = time.perf_counter()
        return DriverResult(
            status="ok",
            details={
                "operation": operation,
                "payload": dict(payload),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
