"""Local eihead observation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ProtocolMessage


VISION_STATIC_COMPAT_MODE = "compat/static"
VISION_REALTIME_MODE = "realtime"
EYE_REALTIME_CHANNEL = "eye.realtime"
VISION_REALTIME_ALIAS = "vision.realtime"


@dataclass(slots=True)
class AudioTranscriptFinal(ProtocolMessage):
    kind: str = field(init=False, default="audio_transcript_final")
    text: str = ""
    language: str = "zh"


@dataclass(slots=True)
class HeadObservation(ProtocolMessage):
    target: str = ""
    timestamp_ms: int | None = None
    trace_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="head_observation")

    @property
    def observation_type(self) -> str:
        return self.kind

    @property
    def modality(self) -> str:
        return "head"


@dataclass(slots=True)
class VisionObservation(HeadObservation):
    frame_id: str = ""
    width: int | None = None
    height: int | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    tracked_target: dict[str, Any] = field(default_factory=dict)
    mode: str = VISION_STATIC_COMPAT_MODE
    primary_mode: bool = False
    kind: str = field(init=False, default="vision_observation")

    @property
    def modality(self) -> str:
        return "vision"


@dataclass(slots=True)
class VoiceAudioFrameObservation(HeadObservation):
    stream_id: str = ""
    chunk_index: int | None = None
    audio_base64: str = ""
    sample_rate_hz: int | None = None
    channels: int | None = None
    latency_ms: float | None = None
    kind: str = field(init=False, default="voice_audio_frame_observation")

    @property
    def modality(self) -> str:
        return "audio"


@dataclass(slots=True)
class RealtimeVisionObservation(HeadObservation):
    stream_id: str = ""
    camera_id: str = ""
    status: str = "unknown"
    frame_id: str = ""
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    latency_ms: float | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    tracked_target: dict[str, Any] = field(default_factory=dict)
    stream: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    channel: str = EYE_REALTIME_CHANNEL
    aliases: list[str] = field(default_factory=lambda: [VISION_REALTIME_ALIAS])
    mode: str = VISION_REALTIME_MODE
    primary_mode: bool = True
    kind: str = field(init=False, default="realtime_vision_observation")

    @property
    def modality(self) -> str:
        return VISION_REALTIME_ALIAS


@dataclass(slots=True)
class VisionTrackingObservation(HeadObservation):
    frame_id: str = ""
    tracked_target: dict[str, Any] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float | None = None
    status: str = "unknown"
    kind: str = field(init=False, default="vision_tracking_observation")

    @property
    def modality(self) -> str:
        return "vision"


__all__ = [
    "AudioTranscriptFinal",
    "EYE_REALTIME_CHANNEL",
    "HeadObservation",
    "RealtimeVisionObservation",
    "VISION_REALTIME_ALIAS",
    "VISION_REALTIME_MODE",
    "VISION_STATIC_COMPAT_MODE",
    "VisionTrackingObservation",
    "VisionObservation",
    "VoiceAudioFrameObservation",
]
