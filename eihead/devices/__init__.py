"""Device adapters for eihead-native modules."""

from .audio import (
    AcousticFrontendReadiness,
    AudioDeviceCandidate,
    AudioRoutePlan,
    PlaybackInterruptionPlan,
    build_aplay_command,
    build_loopback_readiness,
    build_playback_stop_plan,
    build_arecord_command,
    choose_audio_routes,
    evaluate_audio_frontend_readiness,
    parse_aplay_devices,
    parse_arecord_devices,
    parse_pactl_sources,
    select_preferred_input,
)
from .neck_servo import NeckServoCommandAdapter

__all__ = [
    "AcousticFrontendReadiness",
    "AudioDeviceCandidate",
    "AudioRoutePlan",
    "NeckServoCommandAdapter",
    "PlaybackInterruptionPlan",
    "build_aplay_command",
    "build_loopback_readiness",
    "build_playback_stop_plan",
    "build_arecord_command",
    "choose_audio_routes",
    "evaluate_audio_frontend_readiness",
    "parse_aplay_devices",
    "parse_arecord_devices",
    "parse_pactl_sources",
    "select_preferred_input",
]
