"""Envelope wrappers for kernel bus transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import ProtocolMessage


@dataclass(slots=True)
class Envelope:
    channel: str
    payload: dict[str, Any]

    @classmethod
    def wrap(cls, channel: str, payload: ProtocolMessage) -> "Envelope":
        return cls(channel=channel, payload=payload.to_dict())
