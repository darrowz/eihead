"""Minimal eibrain.protocol compatibility exports for transitional eihead."""

from .actions import MoveHeadAction, PlaySpeechAction, StopSpeechAction
from .observations import AudioTranscriptFinal
from .outcomes import ActionExecuted, SpeechPlaybackCompleted

__all__ = [
    "ActionExecuted",
    "AudioTranscriptFinal",
    "MoveHeadAction",
    "PlaySpeechAction",
    "SpeechPlaybackCompleted",
    "StopSpeechAction",
]
