"""Native eihead mouth playback status contracts."""

from .playback import (
    MouthTtsConfig,
    PRIMARY_TTS_PROVIDER,
    SpeechPlaybackStatus,
    SpeakActionSummary,
    StopSpeechSummary,
    build_mouth_status,
    summarize_speak_action,
    summarize_stop_speech_result,
)

__all__ = [
    "MouthTtsConfig",
    "PRIMARY_TTS_PROVIDER",
    "SpeechPlaybackStatus",
    "SpeakActionSummary",
    "StopSpeechSummary",
    "build_mouth_status",
    "summarize_speak_action",
    "summarize_stop_speech_result",
]
