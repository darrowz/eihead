"""eihead-native realtime eye primitives."""

from .adapters import (
    AdapterRuntimeError,
    AdapterReadiness,
    GStreamerHailoDetector,
    GStreamerHailoFrameSource,
    GStreamerHailoRealtimeAdapter,
    GStreamerHailoRealtimeConfig,
    normalize_hailo_detection,
)
from .gstreamer import GStreamerAppSinkFrameReader
from .hailo_metadata import HailoMetadataParseError, parse_hailo_detections
from .realtime import (
    CompatStaticFrameSource,
    RealtimeDetection,
    RealtimeEyePipeline,
    RealtimeEyeStatus,
    RealtimeVisionFrame,
)
from .service import RealtimeEyeService
from .scene import RealtimeVisionSceneBridge
from .tracking import DEFAULT_TRACKING_LABELS, TrackingTarget, select_tracking_target

__all__ = [
    "AdapterReadiness",
    "AdapterRuntimeError",
    "CompatStaticFrameSource",
    "DEFAULT_TRACKING_LABELS",
    "GStreamerHailoDetector",
    "GStreamerHailoFrameSource",
    "GStreamerHailoRealtimeAdapter",
    "GStreamerHailoRealtimeConfig",
    "GStreamerAppSinkFrameReader",
    "HailoMetadataParseError",
    "RealtimeDetection",
    "RealtimeEyePipeline",
    "RealtimeEyeService",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
    "RealtimeVisionSceneBridge",
    "TrackingTarget",
    "normalize_hailo_detection",
    "parse_hailo_detections",
    "select_tracking_target",
]
