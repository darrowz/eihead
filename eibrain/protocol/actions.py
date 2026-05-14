"""Action contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProtocolMessage


@dataclass(slots=True)
class Action(ProtocolMessage):
    kind: str = field(init=False, default="action")


@dataclass(slots=True)
class PlaySpeechAction(Action):
    text: str = ""
    kind: str = field(init=False, default="play_speech_action")


@dataclass(slots=True)
class StopSpeechAction(Action):
    reason: str = ""
    details: dict[str, object] = field(default_factory=dict)
    kind: str = field(init=False, default="stop_speech_action")


@dataclass(slots=True)
class MoveHeadAction(Action):
    target_name: str = ""
    target_x: float | None = None
    target_angle: int | None = None
    kind: str = field(init=False, default="move_head_action")
