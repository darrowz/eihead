"""Unified cognitive event models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eibrain.state.embodied import EmbodiedState

from .base import ProtocolMessage


@dataclass(slots=True)
class ObservationEvent(ProtocolMessage):
    modality: str = "text"
    text: str = ""
    image_url: str = ""
    target_x: float | None = None
    confidence: float = 1.0
    status: str = "ok"
    payload: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="observation_event")


@dataclass(slots=True)
class Moment:
    """Bound view of the current embodied situation."""

    state: EmbodiedState
    session_id: str | None = None
    actor_id: str | None = None
    transcript: str = ""
    visual_summary: str = ""
    visual_target: str = ""
    target_x: float | None = None
    engagement_phase: str = "idle"
    modalities: tuple[str, ...] = ()
    body_capabilities: dict[str, bool] = field(default_factory=dict)

    @property
    def query_text(self) -> str:
        parts = [self.transcript.strip(), self.visual_summary.strip()]
        return " | ".join(part for part in parts if part)


@dataclass(slots=True)
class SalienceDecision:
    score: float
    reason: str
    should_recall: bool = True
    should_reply: bool = False
    should_orient: bool = False
    should_writeback: bool = False
    priority: int = 0


@dataclass(slots=True)
class CognitiveDecision:
    decision_type: str
    reason: str
    should_recall: bool = True
    should_reply: bool = False
    should_orient: bool = False
    should_writeback: bool = False
    priority: int = 0
    active_policy: dict[str, Any] = field(default_factory=dict)
