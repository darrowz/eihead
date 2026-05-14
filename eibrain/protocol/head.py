"""eihead observation, action, and outcome contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ProtocolMessage
from .capabilities import HeadHealth


def _without_kind(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned.pop("kind", None)
    return cleaned


@dataclass(slots=True)
class HeadMessage(ProtocolMessage):
    """Base payload for messages crossing the eihead/eibrain boundary."""

    trace_id: str = ""
    target: str = ""
    timestamp_ms: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="head_message")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadMessage":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        return cls(**payload)


@dataclass(slots=True)
class HeadObservation(HeadMessage):
    """Generic observation emitted by eihead sensors or local perception."""

    observation_type: str = ""
    modality: str = ""
    device_id: str = ""
    confidence: float = 1.0
    status: str = "ok"
    kind: str = field(init=False, default="head_observation")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadObservation":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        return cls(**payload)


@dataclass(slots=True)
class AudioTurn(HeadObservation):
    """Final or partial ASR turn emitted by eihead."""

    text: str = ""
    language: str = "und"
    is_final: bool = True
    start_ms: int | None = None
    end_ms: int | None = None
    audio_level: float | None = None
    wake_word: str = ""
    observation_type: str = "audio_turn"
    modality: str = "audio"
    kind: str = field(init=False, default="audio_turn")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioTurn":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        return cls(**payload)


@dataclass(slots=True)
class VisionObservation(HeadObservation):
    """Vision frame summary emitted by eihead local perception."""

    frame_id: str = ""
    image_url: str = ""
    width: int | None = None
    height: int | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    tracked_target: dict[str, Any] = field(default_factory=dict)
    observation_type: str = "vision_observation"
    modality: str = "vision"
    kind: str = field(init=False, default="vision_observation")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionObservation":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        payload["detections"] = [dict(item) for item in payload.get("detections", [])]
        payload["tracked_target"] = dict(payload.get("tracked_target", {}))
        return cls(**payload)


@dataclass(slots=True)
class DeviceStatus(HeadObservation):
    """Runtime device health and metrics snapshot from eihead."""

    device_id: str = ""
    device_kind: str = ""
    health: HeadHealth = field(default_factory=HeadHealth)
    metrics: dict[str, Any] = field(default_factory=dict)
    observation_type: str = "device_status"
    modality: str = "device"
    kind: str = field(init=False, default="device_status")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeviceStatus":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        payload["health"] = (
            payload["health"]
            if isinstance(payload.get("health"), HeadHealth)
            else HeadHealth.from_dict(payload.get("health"))
        )
        payload["metrics"] = dict(payload.get("metrics", {}))
        return cls(**payload)


@dataclass(slots=True)
class HeadAction(HeadMessage):
    """Command from eibrain to eihead."""

    action_id: str = ""
    action_type: str = ""
    device_id: str = ""
    priority: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="head_action")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadAction":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        payload["params"] = dict(payload.get("params", {}))
        return cls(**payload)


@dataclass(slots=True)
class ExecutionOutcome(HeadMessage):
    """Result of a head action or local eihead execution step."""

    action_id: str = ""
    action_type: str = ""
    device_id: str = ""
    status: str = "ok"
    success: bool = True
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="execution_outcome")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionOutcome":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        payload["details"] = dict(payload.get("details", {}))
        return cls(**payload)


@dataclass(slots=True)
class UserFeedback(HeadMessage):
    """User-visible feedback about the last interaction or execution."""

    feedback_type: str = ""
    value: str = ""
    score: float | None = None
    text: str = ""
    related_trace_id: str = ""
    kind: str = field(init=False, default="user_feedback")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserFeedback":
        payload = _without_kind(data)
        payload["payload"] = dict(payload.get("payload", {}))
        return cls(**payload)
