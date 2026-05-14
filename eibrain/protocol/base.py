"""Shared protocol base models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProtocolMessage:
    """Base message for all protocol payloads."""

    ts: float
    source: str
    session_id: str | None = None
    actor_id: str | None = None
    target_id: str | None = None
    kind: str = field(init=False, default="protocol_message")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
