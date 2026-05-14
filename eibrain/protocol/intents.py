"""Intent contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProtocolMessage


@dataclass(slots=True)
class Intent(ProtocolMessage):
    reason: str = ""
    priority: int = 0
    kind: str = field(init=False, default="intent")


@dataclass(slots=True)
class SpeakIntent(Intent):
    text: str = ""
    kind: str = field(init=False, default="speak_intent")


@dataclass(slots=True)
class PauseIntent(Intent):
    kind: str = field(init=False, default="pause_intent")


@dataclass(slots=True)
class OrientIntent(Intent):
    target_name: str = ""
    target_x: float | None = None
    kind: str = field(init=False, default="orient_intent")
