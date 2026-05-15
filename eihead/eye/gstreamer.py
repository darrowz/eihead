"""Realtime GStreamer appsink reader for eye vision frames."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import time
from typing import Any

from .realtime import REALTIME_STREAM_MODE, RealtimeVisionFrame


class GStreamerAppSinkFrameReader:
    """Read realtime frames from a GStreamer appsink branch.

    The class keeps all heavy imports lazy so importing this module never
    touches GStreamer/Hailo runtime dependencies.
    """

    mode = REALTIME_STREAM_MODE
    source_name = "gstreamer_hailo"

    def __init__(
        self,
        *,
        camera_device: str = "/dev/video0",
        hailo_device: str = "/dev/hailo0",
        hailo_device_id: str = "",
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        backend: str = "gstreamer_hailo",
        hef_path: str = "",
        appsink_name: str = "vision_sink",
        hailofilter_so_path: str = "",
        hailofilter_config_path: str = "",
        hailofilter_function: str = "",
        clock: Callable[[], float] = time.time,
        pipeline_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.camera_device = camera_device
        self.hailo_device = hailo_device
        self.hailo_device_id = str(hailo_device_id)
        self.width = int(width)
        self.height = int(height)
        self.framerate = int(framerate)
        self.backend = backend
        self.hef_path = str(hef_path)
        self.appsink_name = appsink_name
        self.hailofilter_so_path = str(hailofilter_so_path)
        self.hailofilter_config_path = str(hailofilter_config_path)
        self.hailofilter_function = str(hailofilter_function)
        self._clock = clock
        self._pipeline = None
        self._appsink = None
        self._frame_counter = 0
        self._pipeline_metadata = dict(pipeline_metadata or {})

    def start(self) -> None:
        if self._pipeline is not None:
            return

        gi, Gst, _GLib, _hailo = _load_gstreamer_modules()
        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        Gst.init(None)

        pipeline = Gst.parse_launch(self.pipeline_text())
        appsink = pipeline.get_by_name(self.appsink_name)
        if appsink is None:
            raise RuntimeError(f"appsink {self.appsink_name!r} not found in pipeline")

        start_state = pipeline.set_state(Gst.State.PLAYING)
        if start_state == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("failed to start GStreamer appsink pipeline")

        self._pipeline = pipeline
        self._appsink = appsink

    def stop(self) -> None:
        if self._pipeline is None:
            return

        _gi, Gst, _GLib, _hailo = _load_gstreamer_modules()
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsink = None

    def read_frame(self) -> RealtimeVisionFrame | dict[str, Any] | None:
        if self._appsink is None:
            return None
        sample = self._appsink.emit("pull-sample")
        if sample is None:
            return None

        try:
            self._frame_counter += 1
            buffer = sample.get_buffer()
            buffer_timestamp_s = self._read_buffer_timestamp(buffer=buffer)
            metadata = self._build_metadata(buffer_timestamp_s=buffer_timestamp_s)
            width, height = self._read_frame_size(sample=sample)
            return RealtimeVisionFrame(
                frame_id=self._read_frame_id(sample=sample, buffer=buffer),
                timestamp=float(self._clock()),
                width=width,
                height=height,
                source=self.source_name,
                mode=REALTIME_STREAM_MODE,
                payload=sample,
                metadata=metadata,
            )
        except Exception as exc:
            return {
                "backend": self.backend,
                "pipeline": self.pipeline_text(),
                "error": f"{exc.__class__.__name__}: {exc}",
                "frame_counter": self._frame_counter,
                "camera_device": self.camera_device,
                "hailo_device": self.hailo_device,
            }

    def pipeline_text(self) -> str:
        hailo_text = ["hailonet"]
        if self.hailo_device_id:
            hailo_text.append(f"device-id={self.hailo_device_id}")
        if self.hef_path:
            hailo_text.append(f"hef-path={self.hef_path}")
        hailo_clause = " ".join(hailo_text)

        filter_text = ["hailofilter", "qos=false"]
        if self.hailofilter_so_path:
            filter_text.append(f"so-path={self.hailofilter_so_path}")
        if self.hailofilter_config_path:
            filter_text.append(f"config-path={self.hailofilter_config_path}")
        if self.hailofilter_function:
            filter_text.append(f"function-name={self.hailofilter_function}")
        filter_clause = " ".join(filter_text)

        return (
            f"v4l2src device={self.camera_device} io-mode=mmap ! "
            f"video/x-raw,width={self.width},height={self.height},framerate={self.framerate}/1 ! "
            "videoconvert ! "
            "video/x-raw,format=RGB ! "
            f"{hailo_clause} ! "
            f"{filter_clause} ! "
            f"appsink name={self.appsink_name} emit-signals=true sync=false max-buffers=1 drop=true"
        )

    def _read_buffer_timestamp(self, *, buffer: Any) -> float | None:
        for attribute in ("pts", "dts", "timestamp"):
            raw_value = getattr(buffer, attribute, None)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            if value > 1_000_000:
                # GStreamer timestamps are usually in nanoseconds.
                if value > 1_000_000_000:
                    return value / 1_000_000_000
                return value / 1_000_000
            return value
        return None

    def _read_frame_size(self, *, sample: Any) -> tuple[int, int]:
        caps = sample.get_caps()
        if caps is None:
            return self.width, self.height

        structure = None
        if hasattr(caps, "get_structure"):
            try:
                structure = caps.get_structure(0)
            except Exception:
                structure = None
        if structure is None and isinstance(caps, dict):
            structure = caps

        width = _read_structure_size(structure, "width")
        height = _read_structure_size(structure, "height")
        return (
            self.width if width is None else width,
            self.height if height is None else height,
        )

    def _read_frame_id(self, *, sample: Any, buffer: Any) -> str:
        for getter in (
            lambda: sample.get_frame_id(),  # type: ignore[misc]
            lambda: sample.frame_id,
            lambda: sample.frame_meta.get("frame_id"),  # type: ignore[union-attr]
            lambda: buffer.offset,
        ):
            try:
                value = getter()
            except Exception:
                continue
            if value is None:
                continue
            return str(value)
        return f"{self.source_name}-{int(self._clock() * 1000)}"

    def _build_metadata(self, *, buffer_timestamp_s: float | None = None) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "pipeline": self.pipeline_text(),
            "source_name": self.source_name,
            "camera_device": self.camera_device,
            "hailo_device": self.hailo_device,
            "frame_index": self._frame_counter,
            "gst_buffer_timestamp_s": buffer_timestamp_s,
            **self._pipeline_metadata,
        }


def _load_gstreamer_modules():
    gi = importlib.import_module("gi")  # noqa: E402
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    Gst = importlib.import_module("gi.repository.Gst")  # noqa: E402
    GLib = importlib.import_module("gi.repository.GLib")  # noqa: E402
    hailo = importlib.import_module("hailo")  # noqa: E402
    return gi, Gst, GLib, hailo


def _read_structure_size(structure: Any, key: str) -> int | None:
    if structure is None:
        return None
    if hasattr(structure, "get_int"):
        value = structure.get_int(key)
        if isinstance(value, tuple) and len(value) == 2:
            ok, number = value
            if ok:
                return int(number)
        candidate = _safe_int(value)
        if candidate is not None:
            return candidate
    if hasattr(structure, "get_value"):
        value = structure.get_value(key)
        return _safe_int(value)
    if isinstance(structure, dict):
        return _safe_int(structure.get(key))
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
