"""Realtime hardware adapter scaffold for the eihead eye pipeline.

The classes here define a testable boundary for a future
``/dev/video0`` + ``/dev/hailo0`` GStreamer/Hailo implementation.  They do
not import GStreamer at module import time and they do not fall back to static
images as the primary path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import time
from typing import Any

from .realtime import (
    REALTIME_STREAM_MODE,
    RealtimeDetection,
    RealtimeEyePipeline,
    RealtimeEyeStatus,
    RealtimeVisionFrame,
)


DeviceExists = Callable[[str], bool]
GstAvailable = Callable[[], bool]
FrameReader = Callable[[], RealtimeVisionFrame | Mapping[str, Any] | None]
DetectionReader = Callable[[RealtimeVisionFrame], Iterable[Any]]


@dataclass(frozen=True, slots=True)
class GStreamerHailoRealtimeConfig:
    """Configuration for the future realtime GStreamer/Hailo path."""

    camera_device: str = "/dev/video0"
    hailo_device: str = "/dev/hailo0"
    hailo_device_id: str = ""
    width: int = 640
    height: int = 480
    framerate: int = 30
    inference_width: int = 640
    inference_height: int = 640
    inference_format: str = "RGB"
    hef_path: str = ""
    postprocess_so_path: str = ""
    postprocess_config_path: str = ""
    postprocess_function: str = "filter"
    score_threshold: float = 0.3
    labels: tuple[str, ...] = ("person", "face")
    strict_metadata: bool = False
    model_id: str = "hailo"
    backend: str = "gstreamer_hailo"
    source_name: str = "gstreamer_hailo"
    mode: str = REALTIME_STREAM_MODE
    appsink_name: str = "eihead_realtime_sink"
    max_frame_age_s: float | None = 2.0

    @property
    def device_paths(self) -> tuple[str, str]:
        return (self.camera_device, self.hailo_device)

    def pipeline_fields(self) -> dict[str, str]:
        """Return a deterministic pipeline description without importing Gst."""

        inference_parts = ["hailonet"]
        if self.hailo_device_id:
            inference_parts.append(f"device-id={self.hailo_device_id}")
        if self.hef_path:
            inference_parts.append(f"hef-path={self.hef_path}")
        postprocess_parts = ["hailofilter", "qos=false"]
        if self.postprocess_so_path:
            postprocess_parts.append(f"so-path={self.postprocess_so_path}")
        if self.postprocess_config_path:
            postprocess_parts.append(f"config-path={self.postprocess_config_path}")
        if self.postprocess_function:
            postprocess_parts.append(f"function-name={self.postprocess_function}")
        return {
            "mode": self.mode,
            "backend": self.backend,
            "camera_device": self.camera_device,
            "hailo_device": self.hailo_device,
            "source": f"v4l2src device={self.camera_device} do-timestamp=true",
            "caps": f"video/x-raw,width={int(self.width)},height={int(self.height)},framerate={int(self.framerate)}/1",
            "convert": "videoconvert",
            "scale": "videoscale",
            "inference_caps": (
                f"video/x-raw,format={self.inference_format},"
                f"width={int(self.inference_width)},height={int(self.inference_height)}"
            ),
            "inference": " ".join(inference_parts),
            "postprocess": " ".join(postprocess_parts),
            "sink": (
                f"appsink name={self.appsink_name} emit-signals=true "
                "sync=false max-buffers=1 drop=true"
            ),
        }

    def build_pipeline_description(self) -> str:
        fields = self.pipeline_fields()
        return " ! ".join(
            [
                fields["source"],
                fields["caps"],
                fields["convert"],
                fields["scale"],
                fields["inference_caps"],
                fields["inference"],
                fields["postprocess"],
                fields["sink"],
            ]
        )


@dataclass(frozen=True, slots=True)
class AdapterReadiness:
    ready: bool
    status: str
    message: str
    missing_devices: tuple[str, ...] = ()


class GStreamerHailoFrameSource:
    """Realtime frame source boundary for the future GStreamer appsink."""

    mode = REALTIME_STREAM_MODE

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        frame_reader: FrameReader | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or GStreamerHailoRealtimeConfig()
        self.source_name = self.config.source_name
        self.backend = self.config.backend
        self._device_exists = device_exists or _default_device_exists
        self._gst_available = gst_available or _default_gst_available
        self._frame_reader = frame_reader
        self._clock = clock

    @property
    def placeholder(self) -> bool:
        return not self.readiness().ready

    @property
    def not_wired(self) -> bool:
        return not self.readiness().ready

    def readiness(self) -> AdapterReadiness:
        readiness = _readiness(self.config, device_exists=self._device_exists, gst_available=self._gst_available)
        if not readiness.ready:
            return readiness
        if self._frame_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime frame reader is not wired",
            )
        return readiness

    def pipeline_fields(self) -> dict[str, str]:
        return self.config.pipeline_fields()

    def build_pipeline_description(self) -> str:
        return self.config.build_pipeline_description()

    def next_frame(self) -> RealtimeVisionFrame | None:
        if not self.readiness().ready or self._frame_reader is None:
            return None
        try:
            raw_frame = self._frame_reader()
        except Exception as exc:
            raise AdapterRuntimeError(
                f"realtime frame reader failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        if raw_frame is None:
            return None
        try:
            return _coerce_realtime_frame(raw_frame, config=self.config, clock=self._clock)
        except Exception as exc:
            raise AdapterRuntimeError(
                f"realtime frame conversion failed: {exc.__class__.__name__}: {exc}"
            ) from exc


class GStreamerHailoDetector:
    """Detector boundary that normalizes future Hailo results."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        detection_reader: DetectionReader | None = None,
    ) -> None:
        self.config = config or GStreamerHailoRealtimeConfig()
        self.backend = self.config.backend
        self._device_exists = device_exists or _default_device_exists
        self._gst_available = gst_available or _default_gst_available
        self._detection_reader = detection_reader

    @property
    def placeholder(self) -> bool:
        return not self.readiness().ready

    @property
    def not_wired(self) -> bool:
        return not self.readiness().ready

    def readiness(self) -> AdapterReadiness:
        readiness = _readiness(self.config, device_exists=self._device_exists, gst_available=self._gst_available)
        if not readiness.ready:
            return readiness
        if self._detection_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime detection reader is not wired",
            )
        return readiness

    def detect(self, frame: RealtimeVisionFrame) -> list[RealtimeDetection | Mapping[str, Any]]:
        if frame is None or not self.readiness().ready or self._detection_reader is None:
            return []
        try:
            raw_detections = list(self._detection_reader(frame))
        except TypeError as exc:
            raise AdapterRuntimeError(
                f"realtime detection reader did not return iterable: {exc.__class__.__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise AdapterRuntimeError(
                f"realtime detection reader failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        normalized: list[RealtimeDetection | Mapping[str, Any]] = []
        for raw_detection in raw_detections:
            normalized.append(
                normalize_hailo_detection(
                    raw_detection,
                    frame=frame,
                    config=self.config,
                )
            )
        return normalized


class GStreamerHailoRealtimeAdapter:
    """Small composition wrapper for source, detector, and realtime status."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        frame_reader: FrameReader | None = None,
        detection_reader: DetectionReader | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or GStreamerHailoRealtimeConfig()
        self._device_exists = device_exists or _default_device_exists
        self._gst_available = gst_available or _default_gst_available
        self._frame_reader = frame_reader
        self._detection_reader = detection_reader
        self._parser_state: dict[str, Any] = {"parse_error_count": 0, "errors": []}
        self._native_frame_reader = None
        self.frame_source = GStreamerHailoFrameSource(
            self.config,
            device_exists=self._device_exists,
            gst_available=self._gst_available,
            frame_reader=frame_reader,
            clock=clock,
        )
        self.detector = GStreamerHailoDetector(
            self.config,
            device_exists=self._device_exists,
            gst_available=self._gst_available,
            detection_reader=detection_reader,
        )
        self._pipeline = RealtimeEyePipeline(
            frame_source=self.frame_source,
            detector=self.detector,
            backend=self.config.backend,
            mode=self.config.mode,
            clock=clock,
            max_frame_age_s=self.config.max_frame_age_s,
        )
        self._last_status = self._initial_status()

    @classmethod
    def from_native_gstreamer(
        cls,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        clock: Callable[[], float] = time.time,
        frame_reader_factory: Callable[..., Any] | None = None,
        hailo_module_loader: Callable[[], Any] | None = None,
    ) -> "GStreamerHailoRealtimeAdapter":
        """Build the native `/dev/video0` + `/dev/hailo0` appsink adapter."""

        adapter_config = config or GStreamerHailoRealtimeConfig()
        if frame_reader_factory is None:
            from .gstreamer import GStreamerAppSinkFrameReader

            frame_reader_factory = GStreamerAppSinkFrameReader
        if hailo_module_loader is None:
            from .gstreamer import _load_gstreamer_modules

            hailo_module_loader = lambda: _load_gstreamer_modules()[3]

        native_reader = frame_reader_factory(
            camera_device=adapter_config.camera_device,
            hailo_device=adapter_config.hailo_device,
            hailo_device_id=adapter_config.hailo_device_id,
            width=adapter_config.width,
            height=adapter_config.height,
            framerate=adapter_config.framerate,
            inference_width=adapter_config.inference_width,
            inference_height=adapter_config.inference_height,
            inference_format=adapter_config.inference_format,
            backend=adapter_config.backend,
            hef_path=adapter_config.hef_path,
            appsink_name=adapter_config.appsink_name,
            hailofilter_so_path=adapter_config.postprocess_so_path,
            hailofilter_config_path=adapter_config.postprocess_config_path,
            hailofilter_function=adapter_config.postprocess_function,
            clock=clock,
            pipeline_metadata={"adapter": "eihead.eye.adapters"},
        )
        parser_state: dict[str, Any] = {"parse_error_count": 0, "errors": []}

        def read_frame() -> RealtimeVisionFrame | Mapping[str, Any] | None:
            if hasattr(native_reader, "start"):
                native_reader.start()
            frame = native_reader.read_frame()
            if isinstance(frame, Mapping) and frame.get("error"):
                raise AdapterRuntimeError(f"native GStreamer frame read failed: {frame.get('error')}")
            return frame

        def read_detections(frame: RealtimeVisionFrame) -> Iterable[Mapping[str, Any]]:
            from .hailo_metadata import parse_hailo_detections

            sample = frame.payload
            if sample is None or not hasattr(sample, "get_buffer"):
                return []
            try:
                buffer = sample.get_buffer()
            except Exception as exc:
                raise AdapterRuntimeError(
                    f"native GStreamer sample buffer unavailable: {exc.__class__.__name__}: {exc}"
                ) from exc
            parsed = parse_hailo_detections(
                buffer=buffer,
                hailo_module=hailo_module_loader(),
                model_id=adapter_config.model_id,
                score_threshold=adapter_config.score_threshold,
                labels=adapter_config.labels,
                strict=adapter_config.strict_metadata,
            )
            parser_state["parse_error_count"] = _safe_int(parsed.get("parse_error_count")) or 0
            parser_state["errors"] = parsed.get("errors", [])
            detections = parsed.get("detections", [])
            return detections if isinstance(detections, list) else []

        adapter = cls(
            adapter_config,
            device_exists=device_exists,
            gst_available=gst_available,
            frame_reader=read_frame,
            detection_reader=read_detections,
            clock=clock,
        )
        adapter._native_frame_reader = native_reader
        adapter._parser_state = parser_state
        return adapter

    def readiness(self) -> AdapterReadiness:
        readiness = _readiness(self.config, device_exists=self._device_exists, gst_available=self._gst_available)
        if not readiness.ready:
            return readiness
        if self._frame_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime frame reader is not wired",
            )
        if self._detection_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime detection reader is not wired",
            )
        return readiness

    def pipeline_fields(self) -> dict[str, str]:
        return self.config.pipeline_fields()

    def build_pipeline_description(self) -> str:
        return self.config.build_pipeline_description()

    def status(self) -> RealtimeEyeStatus:
        readiness = self.readiness()
        if not readiness.ready:
            self._last_status = _not_wired_status(
                self.config,
                readiness.message,
                parser_state=self._parser_state,
            )
        return self._last_status

    def poll(self) -> RealtimeEyeStatus:
        readiness = self.readiness()
        if not readiness.ready:
            self._last_status = _not_wired_status(
                self.config,
                readiness.message,
                parser_state=self._parser_state,
            )
            return self._last_status

        try:
            payload = self._pipeline.process_next()
        except AdapterRuntimeError as exc:
            self._last_status = _degraded_status(
                self.config,
                str(exc),
                status_reason=_adapter_error_reason(str(exc)),
                readiness_message=readiness.message,
                parser_state=self._parser_state,
            )
            return self._last_status
        except Exception as exc:  # pragma: no cover - defensive guard for hardware backends.
            self._last_status = _degraded_status(
                self.config,
                f"realtime adapter poll failed: {exc.__class__.__name__}: {exc}",
                status_reason="adapter_poll_failed",
                readiness_message=readiness.message,
                parser_state=self._parser_state,
            )
            return self._last_status
        if isinstance(payload, Mapping):
            self._last_status = _status_from_payload(
                {
                    **dict(payload),
                    **_adapter_diagnostics(
                        self.config,
                        readiness_message=readiness.message,
                        parser_state=self._parser_state,
                    ),
                }
            )
            return self._last_status
        self._last_status = _not_wired_status(
            self.config,
            f"invalid realtime payload from pipeline: {type(payload).__name__}",
            parser_state=self._parser_state,
        )
        return self._last_status

    def _initial_status(self) -> RealtimeEyeStatus:
        readiness = self.readiness()
        if not readiness.ready:
            return _not_wired_status(
                self.config,
                readiness.message,
                parser_state=self._parser_state,
            )
        return RealtimeEyeStatus(
            mode=self.config.mode,
            status="waiting_for_frame",
            backend=self.config.backend,
            source="eihead.eye.adapters",
            placeholder=False,
            not_wired=False,
            stream_ready=False,
            status_reason="waiting_for_frame",
            message="realtime adapter is wired; waiting for frames",
            **_adapter_diagnostics(
                self.config,
                readiness_message=readiness.message,
                parser_state=self._parser_state,
            ),
        )


def normalize_hailo_detection(
    raw_detection: Any,
    *,
    frame: RealtimeVisionFrame,
    config: GStreamerHailoRealtimeConfig | None = None,
) -> dict[str, Any]:
    """Convert a raw Hailo/GStreamer detection payload to normalized detection fields.

    The adapter always emits a dict for downstream normalization into
    ``RealtimeDetection``. This keeps parser output shape stable across mapping
    and dataclass sources.
    """

    adapter_config = config or GStreamerHailoRealtimeConfig()
    frame_ts = _safe_float(frame.timestamp, default=0.0)
    if isinstance(raw_detection, RealtimeDetection):
        return {
            "label": str(raw_detection.label),
            "score": float(raw_detection.confidence),
            "bbox": _bbox_dict(raw_detection.bbox),
            "class_id": raw_detection.class_id,
            "track_id": raw_detection.track_id,
            "source": raw_detection.source,
            "model_id": raw_detection.model_id,
            "ts": frame_ts,
            "raw": raw_detection.to_dict(),
            "attributes": raw_detection.attributes,
        }

    raw_mapping = raw_detection if isinstance(raw_detection, Mapping) else {}
    if isinstance(raw_detection, (list, tuple)) and len(raw_detection) == 4:
        raw_mapping = {"bbox": raw_detection}
    raw_payload = raw_detection if isinstance(raw_detection, Mapping) else {"value": raw_detection}
    raw_bbox = _raw_bbox(raw_mapping)
    normalized_bbox = _normalize_bbox(raw_bbox, frame=frame)
    class_id = _safe_int(raw_mapping.get("class_id"))
    label = _first_text(raw_mapping, ("label", "class_name", "name")) or (
        f"class_{class_id}" if class_id is not None else "unknown"
    )
    score = _safe_float(
        _first_value(raw_mapping, ("score", "confidence", "probability")),
        default=0.0,
    )
    ts = _safe_float(
        _first_value(
            raw_mapping,
            ("ts", "timestamp", "captured_at_ts", "frame_ts", "frame_timestamp"),
        ),
        default=frame_ts,
    )
    if ts is None:
        ts = frame_ts
    attributes = raw_mapping.get("attributes") if isinstance(raw_mapping.get("attributes"), Mapping) else {}
    return {
        "label": label,
        "score": score,
        "confidence": score,
        "bbox": normalized_bbox,
        "class_id": class_id,
        "track_id": raw_mapping.get("track_id"),
        "source": str(raw_mapping.get("source", adapter_config.source_name)),
        "model_id": str(raw_mapping.get("model_id", adapter_config.model_id)),
        "attributes": attributes if isinstance(attributes, Mapping) else {},
        "ts": ts,
        "raw": raw_payload,
    }


def _readiness(
    config: GStreamerHailoRealtimeConfig,
    *,
    device_exists: DeviceExists,
    gst_available: GstAvailable,
) -> AdapterReadiness:
    missing_devices = tuple(path for path in config.device_paths if not device_exists(path))
    if missing_devices:
        return AdapterReadiness(
            ready=False,
            status="not_wired",
            missing_devices=missing_devices,
            message=f"missing realtime devices: {', '.join(missing_devices)}",
        )
    if not gst_available():
        return AdapterReadiness(
            ready=False,
            status="not_wired",
            message="GStreamer backend is not installed",
        )
    return AdapterReadiness(ready=True, status="ready", message="realtime adapter is wired")


def _not_wired_status(
    config: GStreamerHailoRealtimeConfig,
    message: str,
    *,
    parser_state: Mapping[str, Any] | None = None,
) -> RealtimeEyeStatus:
    return RealtimeEyeStatus(
        mode=config.mode,
        status="not_wired",
        backend=config.backend,
        source="eihead.eye.adapters",
        placeholder=True,
        not_wired=True,
        stream_ready=False,
        status_reason="not_wired",
        not_wired_reason=message,
        message=message,
        **_adapter_diagnostics(config, readiness_message=message, parser_state=parser_state),
    )


def _degraded_status(
    config: GStreamerHailoRealtimeConfig,
    message: str,
    *,
    status_reason: str,
    readiness_message: str,
    parser_state: Mapping[str, Any] | None = None,
) -> RealtimeEyeStatus:
    return RealtimeEyeStatus(
        mode=config.mode,
        status="degraded",
        backend=config.backend,
        source="eihead.eye.adapters",
        placeholder=False,
        not_wired=False,
        stream_ready=False,
        degraded=True,
        status_reason=status_reason,
        degraded_reason=message,
        message=message,
        **_adapter_diagnostics(config, readiness_message=readiness_message, parser_state=parser_state),
    )


def _adapter_diagnostics(
    config: GStreamerHailoRealtimeConfig,
    *,
    readiness_message: str,
    parser_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parser_state = parser_state or {}
    return {
        "pipeline": config.pipeline_fields(),
        "devices": {
            "camera": config.camera_device,
            "hailo": config.hailo_device,
        },
        "readiness_message": readiness_message,
        "parse_error_count": _safe_int(parser_state.get("parse_error_count")) or 0,
        "parse_errors": list(parser_state.get("errors", [])) if isinstance(parser_state.get("errors"), list) else [],
    }


def _coerce_realtime_frame(
    raw_frame: RealtimeVisionFrame | Mapping[str, Any],
    *,
    config: GStreamerHailoRealtimeConfig,
    clock: Callable[[], float],
) -> RealtimeVisionFrame:
    if isinstance(raw_frame, RealtimeVisionFrame):
        if raw_frame.is_compat_static:
            raise ValueError("GStreamerHailoFrameSource only accepts realtime frames")
        return raw_frame

    metadata = raw_frame.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw_timestamp = raw_frame.get("timestamp", raw_frame.get("captured_at_ts"))
    if raw_timestamp is None:
        timestamp = float(clock())
    else:
        try:
            timestamp = float(raw_timestamp)
        except (TypeError, ValueError):
            timestamp = float(clock())
    return RealtimeVisionFrame(
        frame_id=str(raw_frame.get("frame_id") or f"gstreamer-hailo-{int(clock() * 1000)}"),
        timestamp=timestamp,
        width=_safe_int(raw_frame.get("width")) or config.width,
        height=_safe_int(raw_frame.get("height")) or config.height,
        source=config.source_name,
        mode=config.mode,
        payload=raw_frame.get("payload"),
        metadata={
            **dict(metadata),
            "backend": config.backend,
            "camera_device": config.camera_device,
            "hailo_device": config.hailo_device,
        },
    )


def _status_from_payload(payload: Mapping[str, Any]) -> RealtimeEyeStatus:
    detections = [_detection_from_payload(item) for item in payload.get("detections", [])]
    top_detection_payload = payload.get("top_detection")
    top_detection = (
        _detection_from_payload(top_detection_payload)
        if isinstance(top_detection_payload, Mapping)
        else (detections[0] if detections else None)
    )
    status = str(payload.get("status", "waiting_for_frame"))
    compatibility_mode = _coerce_bool(payload.get("compatibility_mode"), default=False)
    stale = _coerce_bool(payload.get("stale"), default=status == "stale")
    degraded = _coerce_bool(payload.get("degraded"), default=status == "degraded")
    not_wired = _coerce_bool(payload.get("not_wired"), default=False)
    placeholder = _coerce_bool(payload.get("placeholder"), default=False)
    stream_ready = _coerce_bool(
        payload.get("stream_ready"),
        default=(
            status in {"ok", "tracking"}
            and not compatibility_mode
            and not stale
            and not degraded
            and not not_wired
            and not placeholder
        ),
    )
    status_reason = str(payload.get("status_reason", "") or "")
    if not status_reason:
        status_reason = _status_reason_from_payload(
            status=status,
            stream_ready=stream_ready,
            stale=stale,
            degraded=degraded,
            not_wired=not_wired or placeholder,
            compatibility_mode=compatibility_mode,
        )
    return RealtimeEyeStatus(
        schema=str(payload.get("schema", "eihead.eye.realtime_status.v1")),
        mode=str(payload.get("mode", REALTIME_STREAM_MODE)),
        status=status,
        backend=str(payload.get("backend", "gstreamer_hailo")),
        frame_count=int(payload.get("frame_count", 0) or 0),
        detection_count=int(payload.get("detection_count", 0) or 0),
        fps=float(payload.get("fps", 0.0) or 0.0),
        last_frame_id=str(payload.get("last_frame_id", "") or ""),
        last_frame_age=payload.get("last_frame_age"),
        last_frame_captured_at_ts=payload.get("last_frame_captured_at_ts"),
        top_detection=top_detection,
        detections=detections,
        source=str(payload.get("source", "eihead.eye.adapters")),
        placeholder=placeholder,
        not_wired=not_wired,
        stream_ready=stream_ready,
        stale=stale,
        degraded=degraded,
        compatibility_mode=compatibility_mode,
        status_reason=status_reason,
        not_wired_reason=str(payload.get("not_wired_reason", "") or ""),
        stale_reason=str(payload.get("stale_reason", "") or ""),
        degraded_reason=str(payload.get("degraded_reason", "") or ""),
        message=str(payload.get("message", "") or ""),
        detection_boxes=_boxes_from_payload(payload.get("detection_boxes"), detections),
        detection_scores=_scores_from_payload(payload.get("detection_scores"), detections),
        readiness=_mapping_or_empty(payload.get("readiness")),
        compatibility_static_image=_mapping_or_empty(payload.get("compatibility_static_image")),
        pipeline=dict(payload["pipeline"]) if isinstance(payload.get("pipeline"), Mapping) else None,
        devices=dict(payload["devices"]) if isinstance(payload.get("devices"), Mapping) else None,
        readiness_message=str(payload.get("readiness_message", "") or ""),
        parse_error_count=_safe_int(payload.get("parse_error_count")),
        parse_errors=list(payload.get("parse_errors", [])) if isinstance(payload.get("parse_errors"), list) else [],
    )


def _adapter_error_reason(message: str) -> str:
    if "detection reader" in message or "sample buffer unavailable" in message:
        return "detection_reader_failed"
    if "frame reader" in message or "frame conversion" in message or "frame read" in message:
        return "frame_reader_failed"
    return "adapter_runtime_failed"


def _status_reason_from_payload(
    *,
    status: str,
    stream_ready: bool,
    stale: bool,
    degraded: bool,
    not_wired: bool,
    compatibility_mode: bool,
) -> str:
    if compatibility_mode:
        return "compat_static_frame_test_only"
    if stale:
        return "last_frame_stale"
    if degraded:
        return "degraded"
    if not_wired:
        return "not_wired"
    if stream_ready:
        return "realtime_stream_ready"
    return status


def _boxes_from_payload(raw_boxes: Any, detections: list[RealtimeDetection]) -> list[dict[str, float]]:
    if isinstance(raw_boxes, list):
        boxes = [_box_from_mapping(item) for item in raw_boxes]
        return [box for box in boxes if box is not None]
    boxes = [_box_from_mapping(detection.bbox) for detection in detections]
    return [box for box in boxes if box is not None]


def _scores_from_payload(raw_scores: Any, detections: list[RealtimeDetection]) -> list[float]:
    if isinstance(raw_scores, list):
        scores = [_safe_float(item, default=None) for item in raw_scores]
        return [round(float(score), 6) for score in scores if score is not None]
    return [round(float(detection.score), 6) for detection in detections]


def _box_from_mapping(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "x_min": _safe_float(value.get("x_min"), default=0.0),
        "y_min": _safe_float(value.get("y_min"), default=0.0),
        "x_max": _safe_float(value.get("x_max"), default=0.0),
        "y_max": _safe_float(value.get("y_max"), default=0.0),
    }


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"", "0", "false", "no", "off", "none", "null"}:
            return False
    return bool(value)


def _detection_from_payload(payload: Any) -> RealtimeDetection:
    if isinstance(payload, RealtimeDetection):
        return payload
    if not isinstance(payload, Mapping):
        return RealtimeDetection(label="unknown", confidence=0.0, bbox=None)
    attributes: dict[str, Any] = {}
    if isinstance(payload.get("attributes"), Mapping):
        attributes = dict(payload["attributes"])
    if "ts" in payload and "ts" not in attributes:
        attributes["ts"] = payload["ts"]
    if "raw" in payload and "raw" not in attributes:
        attributes["raw"] = payload["raw"]
    return RealtimeDetection(
        label=str(payload.get("label", "unknown")),
        confidence=_safe_float(payload.get("confidence", payload.get("score")), default=0.0),
        bbox=payload.get("bbox") if isinstance(payload.get("bbox"), Mapping) else None,
        class_id=_safe_int(payload.get("class_id")),
        track_id=payload.get("track_id"),
        source=str(payload.get("source", "gstreamer_hailo")),
        model_id=str(payload.get("model_id", "")),
        attributes=attributes,
    )


def _bbox_dict(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        normalized = {}
        for key in ("x_min", "y_min", "x_max", "y_max"):
            normalized[key] = _safe_float(value.get(key), default=0.0)
        return normalized
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x_min, y_min, x_max, y_max = value
        return {
            "x_min": _safe_float(x_min, default=0.0),
            "y_min": _safe_float(y_min, default=0.0),
            "x_max": _safe_float(x_max, default=0.0),
            "y_max": _safe_float(y_max, default=0.0),
        }
    return None


def _raw_bbox(raw_detection: Mapping[str, Any]) -> Any:
    return _first_value(raw_detection, ("bbox", "box", "bounds"))


def _normalize_bbox(raw_bbox: Any, *, frame: RealtimeVisionFrame) -> dict[str, float] | None:
    if raw_bbox is None:
        return None

    if isinstance(raw_bbox, Mapping):
        x_min = _first_value(raw_bbox, ("x_min", "xmin", "x1", "left"))
        y_min = _first_value(raw_bbox, ("y_min", "ymin", "y1", "top"))
        x_max = _first_value(raw_bbox, ("x_max", "xmax", "x2", "right"))
        y_max = _first_value(raw_bbox, ("y_max", "ymax", "y2", "bottom"))
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x_min, y_min, x_max, y_max = raw_bbox
    else:
        return None

    bbox = {
        "x_min": _safe_float(x_min, default=0.0),
        "y_min": _safe_float(y_min, default=0.0),
        "x_max": _safe_float(x_max, default=0.0),
        "y_max": _safe_float(y_max, default=0.0),
    }
    width = _safe_float(frame.width, default=0.0)
    height = _safe_float(frame.height, default=0.0)
    if width > 0 and max(abs(bbox["x_min"]), abs(bbox["x_max"])) > 1.0:
        bbox["x_min"] = bbox["x_min"] / width
        bbox["x_max"] = bbox["x_max"] / width
    if height > 0 and max(abs(bbox["y_min"]), abs(bbox["y_max"])) > 1.0:
        bbox["y_min"] = bbox["y_min"] / height
        bbox["y_max"] = bbox["y_max"] / height
    return bbox


def _first_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_value(mapping, keys)
    return str(value) if value is not None else ""


def _first_value(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _safe_float(value: Any, *, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_device_exists(path: str) -> bool:
    return Path(path).exists()


def _default_gst_available() -> bool:
    if importlib.util.find_spec("gi") is None:
        return False
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        Gst = importlib.import_module("gi.repository.Gst")
        Gst.init(None)
        element_factory = getattr(Gst, "ElementFactory", None)
        if element_factory is None:
            return False
        return all(
            element_factory.find(element_name) is not None
            for element_name in ("v4l2src", "videoconvert", "appsink", "hailonet", "hailofilter")
        )
    except Exception:
        return False


class AdapterRuntimeError(RuntimeError):
    """Raised when a realtime adapter callback fails during polling."""


__all__ = [
    "AdapterRuntimeError",
    "AdapterReadiness",
    "GStreamerHailoDetector",
    "GStreamerHailoFrameSource",
    "GStreamerHailoRealtimeAdapter",
    "GStreamerHailoRealtimeConfig",
    "normalize_hailo_detection",
    "RealtimeDetection",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
]
