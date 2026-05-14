"""Outcome contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ProtocolMessage


@dataclass(slots=True)
class Outcome(ProtocolMessage):
    status: str = "ok"
    kind: str = field(init=False, default="outcome")


@dataclass(slots=True)
class SpeechPlaybackCompleted(Outcome):
    kind: str = field(init=False, default="speech_playback_completed")


@dataclass(slots=True)
class ActionExecuted(Outcome):
    action_kind: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="action_executed")
