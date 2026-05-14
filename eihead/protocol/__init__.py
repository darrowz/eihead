"""Minimal local protocol models for the eihead runtime."""

from __future__ import annotations

from .actions import Action, MoveHeadAction, PlaySpeechAction, StopSpeechAction
from .base import ProtocolMessage, message_payload, serialize_message
from .observations import (
    AudioTranscriptFinal,
    EYE_REALTIME_CHANNEL,
    HeadObservation,
    RealtimeVisionObservation,
    VISION_REALTIME_ALIAS,
    VISION_REALTIME_MODE,
    VISION_STATIC_COMPAT_MODE,
    VisionObservation,
)
from .outcomes import ActionExecuted, Outcome, SpeechPlaybackCompleted

__all__ = [
    "ActionExecuted",
    "Action",
    "AudioTranscriptFinal",
    "EYE_REALTIME_CHANNEL",
    "HeadObservation",
    "MoveHeadAction",
    "Outcome",
    "PlaySpeechAction",
    "ProtocolMessage",
    "RealtimeVisionObservation",
    "SpeechPlaybackCompleted",
    "StopSpeechAction",
    "VISION_REALTIME_ALIAS",
    "VISION_REALTIME_MODE",
    "VISION_STATIC_COMPAT_MODE",
    "VisionObservation",
    "message_payload",
    "serialize_message",
]
