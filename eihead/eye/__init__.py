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
from .identity import (
    FaceEmbeddingProvider,
    FaceEvidence,
    FaceIdentityMatcher,
    IdentityObservation,
    JsonIdentityRegistry,
    OnnxFaceEmbeddingProvider,
    StaticFaceEmbeddingProvider,
    UnavailableFaceEmbeddingProvider,
)
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
    "FaceEmbeddingProvider",
    "FaceEvidence",
    "FaceIdentityMatcher",
    "GStreamerHailoDetector",
    "GStreamerHailoFrameSource",
    "GStreamerHailoRealtimeAdapter",
    "GStreamerHailoRealtimeConfig",
    "GStreamerAppSinkFrameReader",
    "HailoMetadataParseError",
    "IdentityObservation",
    "JsonIdentityRegistry",
    "OnnxFaceEmbeddingProvider",
    "RealtimeDetection",
    "RealtimeEyePipeline",
    "RealtimeEyeService",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
    "RealtimeVisionSceneBridge",
    "StaticFaceEmbeddingProvider",
    "TrackingTarget",
    "UnavailableFaceEmbeddingProvider",
    "normalize_hailo_detection",
    "parse_hailo_detections",
    "select_tracking_target",
]
