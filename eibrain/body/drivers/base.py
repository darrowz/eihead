"""Driver contracts for the embodied execution boundary.

Drivers are the low-level boundary objects consumed by organs.
The execution layer should only depend on this contract and remain agnostic to
provider implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class DriverResult:
    """Structured result returned by driver calls."""

    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)


class DriverAdapter(Protocol):
    """Protocol for runtime driver adapters.\n\n    status values are interpreted by the organ layer's health normalizer."""

    def heartbeat(self) -> str | DriverResult: ...

    def invoke(self, operation: str, payload: dict[str, Any]) -> DriverResult: ...
