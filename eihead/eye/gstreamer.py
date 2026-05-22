"""Realtime GStreamer appsink reader for eye vision frames."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
import importlib
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .realtime import REALTIME_STREAM_MODE, RealtimeDetection, RealtimeVisionFrame


DEFAULT_EVIDENCE_DIR = Path("/tmp/eibrain-vision/evidence")


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
        inference_width: int = 640,
        inference_height: int = 640,
        inference_format: str = "RGB",
        backend: str = "gstreamer_hailo",
        hef_path: str = "",
        appsink_name: str = "vision_sink",
        hailofilter_so_path: str = "",
        hailofilter_config_path: str = "",
        hailofilter_function: str = "",
        sample_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.time,
        pipeline_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.camera_device = camera_device
        self.hailo_device = hailo_device
        self.hailo_device_id = str(hailo_device_id)
        self.width = int(width)
        self.height = int(height)
        self.framerate = int(framerate)
        self.inference_width = int(inference_width)
        self.inference_height = int(inference_height)
        self.inference_format = str(inference_format)
        self.backend = backend
        self.hef_path = str(hef_path)
        self.appsink_name = appsink_name
        self.hailofilter_so_path = str(hailofilter_so_path)
        self.hailofilter_config_path = str(hailofilter_config_path)
        self.hailofilter_function = str(hailofilter_function)
        self.sample_timeout_s = float(sample_timeout_s)
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
        sample = self._appsink.emit("try-pull-sample", self._sample_timeout_ns())
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
            "videoscale ! "
            f"video/x-raw,format={self.inference_format},width={self.inference_width},height={self.inference_height} ! "
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
            "sample_timeout_s": self.sample_timeout_s,
            **self._pipeline_metadata,
        }

    def _sample_timeout_ns(self) -> int:
        return int(max(0.0, self.sample_timeout_s) * 1_000_000_000)


class GStreamerEvidenceWriter:
    """Persist evidence images from the current appsink sample.

    The writer first accepts samples that expose JPEG bytes via a small Python
    method interface, which keeps unit tests hardware-free. Native GStreamer
    samples are handled by mapping the raw video buffer and encoding it as JPEG.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path = DEFAULT_EVIDENCE_DIR,
        min_interval_s: float = 1.0,
        max_files: int = 240,
        max_age_s: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.max_files = max(0, int(max_files))
        self.max_age_s = max(0.0, float(max_age_s))
        self._clock = clock
        self._last_export_at = 0.0
        self._last_frame_evidence: dict[str, Any] = {}

    def export(
        self,
        frame: RealtimeVisionFrame,
        *,
        detections: list[RealtimeDetection | Mapping[str, Any]] | tuple[RealtimeDetection | Mapping[str, Any], ...] = (),
        include_face_crops: bool = True,
    ) -> dict[str, Any]:
        sample = frame.payload
        if sample is None:
            return {}
        now_ts = float(self._clock())
        if self.min_interval_s and self._last_export_at and now_ts - self._last_export_at < self.min_interval_s:
            return (
                {"frame": dict(self._last_frame_evidence), "face_crops": [], "throttled": True}
                if self._last_frame_evidence
                else {}
            )

        evidence: dict[str, Any] = {}
        safe_frame_id = _safe_path_fragment(frame.frame_id)
        frame_bytes = _sample_jpeg_bytes(sample)
        if frame_bytes:
            frame_path = self.output_dir / f"{safe_frame_id}-frame.jpg"
            self._write_bytes(frame_path, frame_bytes, now_ts=now_ts)
            evidence["frame"] = {
                "path": str(frame_path),
                "frame_id": frame.frame_id,
                "captured_at_ts": frame.timestamp,
                "written_at_ts": now_ts,
                "source": frame.source,
                "sample_source": "appsink_payload",
                "mime_type": "image/jpeg",
                "width": frame.width,
                "height": frame.height,
            }

        face_crops: list[dict[str, Any]] = []
        if include_face_crops:
            for index, detection in enumerate(_face_detections(detections)):
                bbox = _detection_bbox(detection)
                crop_bytes = _sample_crop_jpeg_bytes(
                    sample,
                    bbox=bbox,
                    frame_width=frame.width,
                    frame_height=frame.height,
                )
                if not crop_bytes:
                    continue
                crop_path = self.output_dir / f"{safe_frame_id}-face-{index}.jpg"
                self._write_bytes(crop_path, crop_bytes, now_ts=now_ts)
                face_crops.append(
                    {
                        "path": str(crop_path),
                        "frame_id": frame.frame_id,
                        "label": _detection_label(detection),
                        "score": _detection_score(detection),
                        "bbox": bbox,
                        "mime_type": "image/jpeg",
                    }
                )
        evidence["face_crops"] = face_crops
        if "frame" in evidence or face_crops:
            self._last_export_at = now_ts
            if "frame" in evidence:
                self._last_frame_evidence = dict(evidence["frame"])
            self._cleanup_output_dir(now_ts=now_ts)
            return evidence
        return {}

    def _write_bytes(self, path: Path, payload: bytes, *, now_ts: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{int(now_ts * 1000)}.tmp")
        tmp_path.write_bytes(payload)
        tmp_path.replace(path)

    def _cleanup_output_dir(self, *, now_ts: float) -> None:
        try:
            files = [path for path in self.output_dir.glob("*.jpg") if path.is_file()]
        except OSError:
            return
        if self.max_age_s > 0:
            cutoff = now_ts - self.max_age_s
            for path in files:
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue
            try:
                files = [path for path in self.output_dir.glob("*.jpg") if path.is_file()]
            except OSError:
                return
        if self.max_files <= 0 or len(files) <= self.max_files:
            return
        files.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
        for path in files[: max(0, len(files) - self.max_files)]:
            try:
                path.unlink()
            except OSError:
                continue


def _load_gstreamer_modules():
    gi = importlib.import_module("gi")  # noqa: E402
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    Gst = importlib.import_module("gi.repository.Gst")  # noqa: E402
    GLib = importlib.import_module("gi.repository.GLib")  # noqa: E402
    hailo = importlib.import_module("hailo")  # noqa: E402
    return gi, Gst, GLib, hailo


def _sample_jpeg_bytes(sample: Any) -> bytes | None:
    for getter_name in ("get_jpeg_bytes", "to_jpeg_bytes"):
        getter = getattr(sample, getter_name, None)
        if callable(getter):
            payload = getter()
            return payload if isinstance(payload, bytes) and payload else None
    payload = getattr(sample, "jpeg_bytes", None)
    if isinstance(payload, bytes) and payload:
        return payload
    frame = _sample_raw_frame(sample)
    return _encode_jpeg(frame) if frame is not None else None


def _sample_crop_jpeg_bytes(
    sample: Any,
    *,
    bbox: dict[str, float],
    frame_width: int | None,
    frame_height: int | None,
) -> bytes | None:
    for getter_name in ("crop_jpeg_bytes", "get_crop_jpeg_bytes"):
        getter = getattr(sample, getter_name, None)
        if callable(getter):
            payload = getter(bbox=bbox, frame_width=frame_width, frame_height=frame_height)
            return payload if isinstance(payload, bytes) and payload else None
    frame = _sample_raw_frame(sample)
    if frame is None:
        return None
    crop = _crop_raw_frame(frame, bbox=bbox, frame_width=frame_width, frame_height=frame_height)
    return _encode_jpeg(crop) if crop is not None else None


def _sample_raw_frame(sample: Any) -> Any | None:
    """Return a numpy image array from a raw GStreamer sample, if possible."""

    get_buffer = getattr(sample, "get_buffer", None)
    if not callable(get_buffer):
        return None
    try:
        buffer = get_buffer()
    except Exception:
        return None
    if buffer is None:
        return None
    mapped = _map_buffer(buffer)
    if mapped is None:
        return None
    map_info, raw_bytes = mapped
    try:
        width, height = _sample_dimensions(sample)
        if width is None or height is None:
            return None
        pixel_format = _sample_format(sample).upper()
        channel_count = _pixel_channel_count(pixel_format)
        if channel_count is None:
            return None
        expected_size = int(width) * int(height) * channel_count
        if len(raw_bytes) < expected_size:
            return None
        try:
            import numpy as np
        except Exception:
            return None
        array = np.frombuffer(raw_bytes[:expected_size], dtype=np.uint8)
        if channel_count == 1:
            return array.reshape((int(height), int(width)))
        image = array.reshape((int(height), int(width), channel_count))
        if pixel_format == "BGR":
            return image[..., ::-1].copy()
        if pixel_format == "BGRA":
            return image[..., [2, 1, 0, 3]].copy()
        return image
    finally:
        _unmap_buffer(buffer, map_info)


def _sample_dimensions(sample: Any) -> tuple[int | None, int | None]:
    caps = _sample_caps_structure(sample)
    return (_read_structure_size(caps, "width"), _read_structure_size(caps, "height"))


def _sample_format(sample: Any) -> str:
    caps = _sample_caps_structure(sample)
    value = _read_structure_text(caps, "format")
    return value or "RGB"


def _sample_caps_structure(sample: Any) -> Any:
    get_caps = getattr(sample, "get_caps", None)
    if not callable(get_caps):
        return None
    try:
        caps = get_caps()
    except Exception:
        return None
    if caps is None:
        return None
    if hasattr(caps, "get_structure"):
        try:
            return caps.get_structure(0)
        except Exception:
            return None
    if isinstance(caps, Mapping):
        return caps
    return None


def _map_buffer(buffer: Any) -> tuple[Any, bytes] | None:
    map_func = getattr(buffer, "map", None)
    if not callable(map_func):
        return None
    try:
        mapped = map_func(_gst_map_read_flag())
    except TypeError:
        mapped = map_func(None)
    except Exception:
        return None
    if isinstance(mapped, tuple) and len(mapped) == 2:
        ok, map_info = mapped
        if not ok:
            return None
    else:
        map_info = mapped
    data = getattr(map_info, "data", None)
    if data is None:
        return None
    try:
        return map_info, bytes(data)
    except Exception:
        return None


def _unmap_buffer(buffer: Any, map_info: Any) -> None:
    unmap = getattr(buffer, "unmap", None)
    if callable(unmap):
        try:
            unmap(map_info)
        except Exception:
            return None


def _gst_map_read_flag() -> Any:
    try:
        Gst = importlib.import_module("gi.repository.Gst")
        return Gst.MapFlags.READ
    except Exception:
        return 1


def _pixel_channel_count(pixel_format: str) -> int | None:
    if pixel_format in {"RGB", "BGR"}:
        return 3
    if pixel_format in {"RGBA", "BGRA"}:
        return 4
    if pixel_format in {"GRAY8", "GREY"}:
        return 1
    return None


def _crop_raw_frame(
    frame: Any,
    *,
    bbox: dict[str, float],
    frame_width: int | None,
    frame_height: int | None,
) -> Any | None:
    bounds = _bbox_pixel_bounds(
        bbox,
        frame_width=frame_width or _frame_width(frame),
        frame_height=frame_height or _frame_height(frame),
    )
    if bounds is None:
        return None
    x_min, y_min, x_max, y_max = bounds
    crop = frame[y_min:y_max, x_min:x_max]
    if getattr(crop, "size", 0) == 0:
        return None
    return crop


def _bbox_pixel_bounds(
    bbox: dict[str, float],
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> tuple[int, int, int, int] | None:
    if not bbox or not frame_width or not frame_height:
        return None
    raw_x_min = bbox.get("x_min")
    raw_y_min = bbox.get("y_min")
    raw_x_max = bbox.get("x_max")
    raw_y_max = bbox.get("y_max")
    if None in {raw_x_min, raw_y_min, raw_x_max, raw_y_max}:
        return None
    width = int(frame_width)
    height = int(frame_height)
    x_min = _axis_to_pixel(float(raw_x_min), width)
    y_min = _axis_to_pixel(float(raw_y_min), height)
    x_max = _axis_to_pixel(float(raw_x_max), width)
    y_max = _axis_to_pixel(float(raw_y_max), height)
    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(x_min + 1, min(width, x_max))
    y_max = max(y_min + 1, min(height, y_max))
    return x_min, y_min, x_max, y_max


def _axis_to_pixel(value: float, size: int) -> int:
    return int(round(value * size)) if 0.0 <= value <= 1.0 else int(round(value))


def _frame_width(frame: Any) -> int | None:
    shape = getattr(frame, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        return int(shape[1])
    return None


def _frame_height(frame: Any) -> int | None:
    shape = getattr(frame, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        return int(shape[0])
    return None


def _encode_jpeg(frame: Any) -> bytes | None:
    if frame is None:
        return None
    try:
        import cv2
    except Exception:
        return _encode_jpeg_with_pillow(frame)
    try:
        success, encoded = cv2.imencode(".jpg", _opencv_frame(frame))
    except Exception:
        return None
    if not success:
        return None
    try:
        return encoded.tobytes()
    except Exception:
        return None


def _opencv_frame(frame: Any) -> Any:
    shape = getattr(frame, "shape", ())
    if not isinstance(shape, tuple) or len(shape) < 3:
        return frame
    channels = shape[2]
    try:
        import cv2
    except Exception:
        return frame
    if channels == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return frame


def _encode_jpeg_with_pillow(frame: Any) -> bytes | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        image = Image.fromarray(frame)
        output = BytesIO()
        image.save(output, format="JPEG")
        return output.getvalue()
    except Exception:
        return None


def _face_detections(
    detections: list[RealtimeDetection | Mapping[str, Any]] | tuple[RealtimeDetection | Mapping[str, Any], ...],
) -> list[RealtimeDetection | Mapping[str, Any]]:
    return [detection for detection in detections if _detection_label(detection).lower() == "face"]


def _detection_label(detection: RealtimeDetection | Mapping[str, Any]) -> str:
    if isinstance(detection, RealtimeDetection):
        return detection.label
    return str(detection.get("label", ""))


def _detection_score(detection: RealtimeDetection | Mapping[str, Any]) -> float:
    if isinstance(detection, RealtimeDetection):
        return round(float(detection.score), 6)
    value = detection.get("score", detection.get("confidence", 0.0))
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _detection_bbox(detection: RealtimeDetection | Mapping[str, Any]) -> dict[str, float]:
    raw_bbox: Any
    if isinstance(detection, RealtimeDetection):
        raw_bbox = detection.bbox
    else:
        raw_bbox = detection.get("bbox", {})
    if isinstance(raw_bbox, Mapping):
        return {str(key): float(value) for key, value in raw_bbox.items()}
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x_min, y_min, x_max, y_max = raw_bbox
        return {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max),
        }
    return {}


def _safe_path_fragment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return safe or "frame"


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


def _read_structure_text(structure: Any, key: str) -> str:
    if structure is None:
        return ""
    if hasattr(structure, "get_string"):
        try:
            value = structure.get_string(key)
        except Exception:
            value = None
        if value not in (None, ""):
            return str(value)
    if hasattr(structure, "get_value"):
        try:
            value = structure.get_value(key)
        except Exception:
            value = None
        if value not in (None, ""):
            return str(value)
    if isinstance(structure, dict):
        value = structure.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
