"""Standalone Hailo vision service for honjia.

The service owns /dev/video0 and /dev/hailo0, then publishes a small
state.json contract consumed by EyeOrgan. This keeps monitoring and tracking
from synchronously grabbing frames.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import signal
import time
from typing import Any, Mapping

from eibrain.body.runtime_linux import capture_frame, run_hailo_frame_inference
from apps.body_runtime.engagement_state import DEFAULT_ENGAGEMENT_STATE_PATH
from apps.body_runtime.engagement_state import EngagementStateReader
from eibrain.body.vision_state import DEFAULT_VISION_FRAME_PATH
from eibrain.body.vision_state import DEFAULT_VISION_STATE_PATH
from eibrain.body.vision_state import VisionStateWriter
from eibrain.body.vision_state import build_vision_state
from eibrain.cognition.vision_realtime import RealtimeVisionSimulator
from eibrain.cognition.vision_realtime import to_eiprotocol_event_contents
from eibrain.cognition.vision_realtime import to_eiprotocol_scene_content
from eibrain.infra.config import EIBrainConfig, load_config


COCO80_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

MAX_VISION_TARGET_FPS = 15.0
MIN_VISION_INTERVAL_S = 1.0 / MAX_VISION_TARGET_FPS


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    kind: str
    hef_path: str
    postprocess_so_path: str
    postprocess_config_path: str
    postprocess_function: str
    labels: list[str]
    score_threshold: float
    cadence_hz: float
    priority: int
    output_kind: str
    enabled: bool = True


class SingleFrameHailoDetector:
    """Compatibility detector used as a fallback when GStreamer is unavailable."""

    def __init__(
        self,
        *,
        camera_device: str,
        frame_path: str | Path,
        hef_path: str,
        labels: list[str],
        score_threshold: float,
        model_id: str = "personface",
        input_format: str = "mjpeg",
        video_size: str = "640x480",
        timeout_s: float = 5.0,
    ) -> None:
        self.camera_device = camera_device
        self.frame_path = Path(frame_path)
        self.hef_path = hef_path
        self.model_id = model_id
        self.labels = labels
        self.score_threshold = score_threshold
        self.input_format = input_format
        self.video_size = video_size
        self.timeout_s = timeout_s

    def detect_once(self) -> dict[str, Any]:
        captured_at = time.time()
        capture_result = capture_frame(
            device=self.camera_device,
            output_path=self.frame_path,
            input_format=self.input_format,
            video_size=self.video_size,
            timeout_s=self.timeout_s,
        )
        if capture_result.get("status") != "ok":
            return build_vision_state(
                detections=[],
                frame_path=self.frame_path,
                status="capture_failed",
                frame_captured_at_ts=captured_at,
                backend="hailort_single_frame",
                details={"capture_result": capture_result},
            )
        inference = run_hailo_frame_inference(
            image_path=self.frame_path,
            hef_path=self.hef_path,
            labels=self.labels,
            score_threshold=self.score_threshold,
        )
        details = dict(inference.get("details", {})) if isinstance(inference.get("details"), dict) else {}
        detections = details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        return build_vision_state(
            detections=[item for item in detections if isinstance(item, dict)],
            frame_path=self.frame_path,
            status="ok" if inference.get("status") == "ok" else "inference_failed",
            frame_captured_at_ts=captured_at,
            backend="hailort_single_frame",
            pipeline={"model_id": self.model_id, "output_kind": "detections"},
            details={"capture_result": capture_result, "inference_details": details},
        )


class GStreamerHailoDetector:
    """GStreamer/Hailo detector that reads Hailo metadata from appsink."""

    def __init__(
        self,
        *,
        camera_device: str,
        frame_path: str | Path,
        hef_path: str,
        model_id: str,
        postprocess_so_path: str,
        postprocess_config_path: str,
        postprocess_function: str,
        labels: list[str],
        score_threshold: float,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
    ) -> None:
        self.camera_device = camera_device
        self.frame_path = Path(frame_path)
        self.hef_path = hef_path
        self.model_id = model_id
        self.postprocess_so_path = postprocess_so_path
        self.postprocess_config_path = postprocess_config_path
        self.postprocess_function = postprocess_function
        self.labels = labels
        self.score_threshold = score_threshold
        self.width = width
        self.height = height
        self.framerate = framerate
        self._pipeline = None
        self._appsink = None
        self._latest_state: dict[str, Any] | None = None

    def start(self) -> None:
        gi, Gst, _GLib, _hailo = _load_gstreamer_modules()
        gi.require_version("Gst", "1.0")
        Gst.init(None)
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_text = self._pipeline_text()
        self._pipeline = Gst.parse_launch(pipeline_text)
        self._appsink = self._pipeline.get_by_name("metadata_sink")
        self._appsink.connect("new-sample", self._on_sample)
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("failed to start Hailo GStreamer pipeline")

    def stop(self) -> None:
        if self._pipeline is None:
            return
        _gi, Gst, _GLib, _hailo = _load_gstreamer_modules()
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None

    def detect_once(self) -> dict[str, Any]:
        if self._pipeline is None:
            self.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._latest_state is not None:
                state = dict(self._latest_state)
                self._latest_state = None
                return state
            time.sleep(0.02)
        return build_vision_state(
            detections=[],
            frame_path=self.frame_path,
            status="no_metadata",
            backend="gstreamer_hailo",
            details={"pipeline": self._pipeline_text()},
        )

    def _on_sample(self, appsink) -> object:
        _gi, Gst, _GLib, hailo = _load_gstreamer_modules()
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        detections: list[dict[str, Any]] = []
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
                confidence = float(detection.get_confidence())
                if confidence < self.score_threshold:
                    continue
                bbox = detection.get_bbox()
                item: dict[str, Any] = {
                    "label": str(detection.get_label()),
                    "class_id": int(detection.get_class_id()),
                    "score": confidence,
                    "source": "hailo",
                    "model_id": self.model_id,
                    "bbox": {
                        "x_min": float(bbox.xmin()),
                        "y_min": float(bbox.ymin()),
                        "x_max": float(bbox.xmax()),
                        "y_max": float(bbox.ymax()),
                    },
                }
                track_id = _read_track_id(detection)
                if track_id is not None:
                    item["track_id"] = track_id
                detections.append(item)
        except Exception as exc:
            self._latest_state = build_vision_state(
                detections=[],
                frame_path=self.frame_path,
                status="metadata_parse_failed",
                backend="gstreamer_hailo",
                details={"error": str(exc)},
            )
            return Gst.FlowReturn.OK
        self._latest_state = build_vision_state(
            detections=detections,
            objects=detections,
            frame_path=self.frame_path,
            status="ok",
            backend="gstreamer_hailo",
            pipeline={
                "model_id": self.model_id,
                "hef_path": self.hef_path,
                "postprocess_function": self.postprocess_function,
                "label_count": len(self.labels),
            },
            details={"pipeline": "appsink_metadata"},
        )
        return Gst.FlowReturn.OK

    def _pipeline_text(self) -> str:
        frame_location = str(self.frame_path)
        return (
            f"v4l2src device={self.camera_device} io-mode=mmap ! "
            f"image/jpeg,width={self.width},height={self.height},framerate={self.framerate}/1 ! "
            "jpegdec ! videoconvert ! videoscale ! "
            "video/x-raw,format=RGB,width=640,height=640 ! "
            "tee name=t "
            "t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"hailonet hef-path={self.hef_path} scheduling-algorithm=0 is-active=true force-writable=true ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            f"{self._hailofilter_text()} ! "
            "hailotracker class-id=-1 ! "
            "appsink name=metadata_sink emit-signals=true sync=false max-buffers=1 drop=true "
            "t. ! queue max-size-buffers=1 leaky=downstream ! "
            f"videoconvert ! jpegenc ! multifilesink location={frame_location} max-files=1"
        )

    def _hailofilter_text(self) -> str:
        parts = [
            "hailofilter",
            f"so-path={self.postprocess_so_path}",
            f"function-name={self.postprocess_function}",
            "qos=false",
        ]
        if self.postprocess_config_path:
            parts.insert(2, f"config-path={self.postprocess_config_path}")
        return " ".join(parts)


class VisionHailoService:
    def __init__(
        self,
        *,
        detector,
        writer: VisionStateWriter,
        interval_s: float = 0.2,
        engagement_reader: EngagementStateReader | None = None,
        sleeping_interval_s: float = 2.0,
        clock=time.time,
        realtime_simulator: RealtimeVisionSimulator | None = None,
    ) -> None:
        self.detector = detector
        self.writer = writer
        self.configured_interval_s = max(0.0, float(interval_s))
        self.interval_s = max(self.configured_interval_s, MIN_VISION_INTERVAL_S)
        self.engagement_reader = engagement_reader
        self.sleeping_interval_s = max(float(sleeping_interval_s), self.interval_s)
        self.clock = clock
        self.realtime_simulator = realtime_simulator or RealtimeVisionSimulator()
        self._last_frame_ts: float | None = None
        self._running = False

    def run_forever(self) -> None:
        self._running = True
        while self._running:
            started = time.monotonic()
            result = self.process_once()
            interval_s = self.sleeping_interval_s if isinstance(result, float) else self.interval_s
            sleep_s = max(0.0, interval_s - (time.monotonic() - started))
            if sleep_s:
                time.sleep(sleep_s)

    def process_once(self) -> float | dict[str, Any]:
        if self._should_run_vision():
            loop_started = time.monotonic()
            detected_state = self.detector.detect_once()
            loop_elapsed_ms = _round_float((time.monotonic() - loop_started) * 1000.0)
            state = self._enrich_realtime_state(detected_state, loop_elapsed_ms=loop_elapsed_ms)
            self.writer.write(state)
            return state

        self._stop_detector_pipeline()
        state = self._with_timing_metadata(
            build_vision_state(
                detections=[],
                frame_path=None,
                status="sleeping",
                backend="vision_sleep_gate",
                pipeline={"mode": "sleeping", "reason": "conversation_inactive"},
                details={"engagement": self.engagement_reader.read() if self.engagement_reader else {}},
            )
        )
        self.writer.write(state)
        return self.sleeping_interval_s

    def stop(self) -> None:
        self._running = False
        stop = getattr(self.detector, "stop", None)
        if callable(stop):
            stop()

    def _should_run_vision(self) -> bool:
        if self.engagement_reader is None:
            return True
        return self.engagement_reader.should_run_vision()

    def _stop_detector_pipeline(self) -> None:
        stop = getattr(self.detector, "stop", None)
        if callable(stop):
            stop()

    def _enrich_realtime_state(self, state: dict[str, Any], *, loop_elapsed_ms: float | None = None) -> dict[str, Any]:
        frame_ts = _float_or_none(state.get("frame_captured_at_ts"))
        observed_at = str(state.get("observed_at") or state.get("observedAt") or frame_ts or self.clock())
        frame_id = str(state.get("frame_id") or state.get("frameId") or f"frame-{int((frame_ts or self.clock()) * 1000)}")
        detections = state.get("detections")
        snapshot = self.realtime_simulator.update(
            frame_id=frame_id,
            observed_at=observed_at,
            detections=[dict(item) for item in detections] if isinstance(detections, list) else [],
        )
        scene_content = to_eiprotocol_scene_content(snapshot)
        event_contents = to_eiprotocol_event_contents(snapshot)
        enriched = dict(state)
        details = enriched.get("details")
        details_map = dict(details) if isinstance(details, dict) else {}
        backend = str(enriched.get("backend") or "hailo")
        track_count = len(scene_content.get("objects", [])) if isinstance(scene_content.get("objects"), list) else 0
        latency_ms = _latency_ms(details_map)
        enriched.update(
            {
                "scene": scene_content,
                "spatial": {"relations": list(scene_content.get("relationships", []))},
                "events": event_contents,
                "scene_id": scene_content.get("sceneId"),
                "track_count": track_count,
                "event_count": len(event_contents),
                "fps": self._fps(frame_ts),
                "latency": {"ms": latency_ms},
                "freshness": {
                    "source": backend,
                    "age_s": _round_float(max(0.0, float(self.clock()) - frame_ts)) if frame_ts is not None else None,
                },
                "source": {"backend": backend, "mode": "realtime_simulated"},
                "last_detection_summary": str(snapshot.get("sceneGraphSummary") or scene_content.get("summary") or ""),
            }
        )
        enriched = self._with_timing_metadata(enriched, loop_elapsed_ms=loop_elapsed_ms)
        if frame_ts is not None:
            self._last_frame_ts = frame_ts
        return enriched

    def _with_timing_metadata(
        self,
        state: dict[str, Any],
        *,
        loop_elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        enriched = dict(state)
        telemetry = self._timing_telemetry(enriched, loop_elapsed_ms=loop_elapsed_ms)
        pipeline = enriched.get("pipeline")
        pipeline_map = dict(pipeline) if isinstance(pipeline, dict) else {}
        pipeline_map.update(
            {
                "configured_interval_s": telemetry["configured_interval_s"],
                "interval_s": telemetry["interval_s"],
                "target_fps": telemetry["target_fps"],
            }
        )
        existing_telemetry = enriched.get("telemetry")
        telemetry_map = dict(existing_telemetry) if isinstance(existing_telemetry, dict) else {}
        telemetry_map.update(telemetry)
        enriched["pipeline"] = pipeline_map
        enriched["telemetry"] = telemetry_map
        return enriched

    def _timing_telemetry(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        loop_elapsed_ms: float | None = None,
    ) -> dict[str, Any]:
        raw_interval_s = max(0.0, float(self.interval_s))
        interval_s = _round_float(raw_interval_s)
        telemetry: dict[str, Any] = {
            "configured_interval_s": _round_float(self.configured_interval_s),
            "interval_s": interval_s,
            "target_fps": _round_float(1.0 / raw_interval_s) if raw_interval_s > 0 else 0.0,
        }
        if state is not None:
            frame_ts = _float_or_none(state.get("frame_captured_at_ts"))
            if frame_ts is not None:
                telemetry["frame_age_ms"] = _round_float(max(0.0, float(self.clock()) - frame_ts) * 1000.0)
            telemetry["dropped_frames"] = max(0, int(_float_or_none(state.get("dropped_frames")) or 0.0))
            service_state = state.get("service_state") or state.get("status")
            if service_state is not None:
                telemetry["service_state"] = str(service_state)
        if loop_elapsed_ms is not None:
            telemetry["loop_elapsed_ms"] = _round_float(loop_elapsed_ms)
        return telemetry

    def _fps(self, frame_ts: float | None) -> float:
        if frame_ts is None or self._last_frame_ts is None:
            return 0.0
        delta = frame_ts - self._last_frame_ts
        if delta <= 0:
            return 0.0
        return _round_float(1.0 / delta)


def detector_from_config(config: EIBrainConfig, *, backend: str) -> object:
    eye = config.body.organs.get("eye")
    camera = eye.subfunctions.get("camera") if eye is not None else None
    detection = eye.subfunctions.get("detection") if eye is not None else None
    camera_extra = camera.driver.extra if camera is not None else {}
    detection_extra = detection.driver.extra if detection is not None else {}
    frame_path = Path(str(detection_extra.get("frame_path", camera_extra.get("frame_path", DEFAULT_VISION_FRAME_PATH))))
    model = _select_model_config(detection_extra)
    if backend == "single_frame":
        return SingleFrameHailoDetector(
            camera_device=str(camera_extra.get("device", "/dev/video0")),
            frame_path=frame_path,
            hef_path=model.hef_path,
            model_id=model.id,
            labels=model.labels,
            score_threshold=model.score_threshold,
            input_format=str(camera_extra.get("input_format", "mjpeg") or "mjpeg"),
            video_size=str(camera_extra.get("video_size", "640x480") or "640x480"),
            timeout_s=float(camera_extra.get("timeout_s", 5.0)),
        )
    return GStreamerHailoDetector(
        camera_device=str(camera_extra.get("device", "/dev/video0")),
        frame_path=frame_path,
        hef_path=model.hef_path,
        model_id=model.id,
        postprocess_so_path=model.postprocess_so_path,
        postprocess_config_path=model.postprocess_config_path,
        postprocess_function=model.postprocess_function,
        labels=model.labels,
        score_threshold=model.score_threshold,
        width=int(camera_extra.get("pipeline_width", 640)),
        height=int(camera_extra.get("pipeline_height", 480)),
        framerate=int(camera_extra.get("pipeline_framerate", 30)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eibrain Hailo vision service")
    parser.add_argument("--config", default="config/eibrain.yaml")
    parser.add_argument("--backend", choices=("gstreamer", "single_frame"), default="gstreamer")
    parser.add_argument("--state-path", default="")
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--engagement-state-path", default=str(DEFAULT_ENGAGEMENT_STATE_PATH))
    parser.add_argument("--sleeping-interval-s", type=float, default=2.0)
    parser.add_argument("--security-vision-always-on", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    state_path = Path(args.state_path) if args.state_path else _state_path_from_config(config)
    detector = detector_from_config(config, backend=args.backend)
    engagement_reader = EngagementStateReader(
        args.engagement_state_path,
        security_mode=args.security_vision_always_on,
    )
    service = VisionHailoService(
        detector=detector,
        writer=VisionStateWriter(state_path),
        interval_s=args.interval_s,
        engagement_reader=engagement_reader,
        sleeping_interval_s=args.sleeping_interval_s,
    )

    def _stop(_signum, _frame) -> None:
        service.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(f"eibrain vision service writing {state_path} via {args.backend}", flush=True)
    service.run_forever()


def _state_path_from_config(config: EIBrainConfig) -> Path:
    eye = config.body.organs.get("eye")
    if eye is None:
        return DEFAULT_VISION_STATE_PATH
    for name in ("detection", "camera", "identity"):
        subfunction = eye.subfunctions.get(name)
        if subfunction is None:
            continue
        state_path = subfunction.driver.extra.get("state_path")
        if state_path:
            return Path(str(state_path))
    return DEFAULT_VISION_STATE_PATH


def _load_gstreamer_modules():
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import GLib, Gst  # type: ignore
    import hailo  # type: ignore

    return gi, Gst, GLib, hailo


def _read_track_id(detection) -> object | None:
    try:
        for unique_id in detection.get_objects_typed("HAILO_UNIQUE_ID"):
            return unique_id.get_id()
    except Exception:
        return None
    return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latency_ms(details: dict[str, Any]) -> float | None:
    for key in ("latency_ms", "latencyMs"):
        value = _float_or_none(details.get(key))
        if value is not None:
            return value
    timings = details.get("timings_ms")
    if isinstance(timings, dict):
        value = _float_or_none(timings.get("total"))
        if value is not None:
            return value
    return None


def _round_float(value: float) -> float:
    return round(float(value), 3)


def _select_model_config(detection_extra: dict[str, Any]) -> ModelConfig:
    models = detection_extra.get("models")
    if isinstance(models, list):
        parsed_models = [_model_config_from_mapping(item, detection_extra) for item in models if isinstance(item, dict)]
        enabled_models = [model for model in parsed_models if model.enabled]
        if enabled_models:
            return max(enabled_models, key=lambda model: model.priority)
    return _legacy_model_config(detection_extra)


def _legacy_model_config(detection_extra: dict[str, Any]) -> ModelConfig:
    return _model_config_from_mapping({}, detection_extra)


def _model_config_from_mapping(raw: dict[str, Any], detection_extra: dict[str, Any]) -> ModelConfig:
    model_id = str(raw.get("id", detection_extra.get("model_id", "personface")))
    return ModelConfig(
        id=model_id,
        kind=str(raw.get("kind", detection_extra.get("kind", "yolo"))),
        hef_path=str(raw.get("hef_path", detection_extra.get("hef_path", "/usr/share/hailo-models/yolov5s_personface_h8l.hef"))),
        postprocess_so_path=str(
            raw.get(
                "postprocess_so_path",
                detection_extra.get(
                    "postprocess_so_path",
                    "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
                ),
            )
        ),
        postprocess_config_path=str(
            raw.get(
                "postprocess_config_path",
                detection_extra.get("postprocess_config_path", "/usr/share/hailo-models/yolov5_personface.json"),
            )
        ),
        postprocess_function=str(raw.get("postprocess_function", detection_extra.get("postprocess_function", "filter"))),
        labels=_read_labels(raw.get("labels", detection_extra.get("labels")), label_set=raw.get("label_set", detection_extra.get("label_set")), default=["person", "face"]),
        score_threshold=float(raw.get("score_threshold", detection_extra.get("score_threshold", 0.3))),
        cadence_hz=float(raw.get("cadence_hz", detection_extra.get("cadence_hz", 0.0))),
        priority=int(raw.get("priority", detection_extra.get("priority", 0))),
        output_kind=str(raw.get("output_kind", detection_extra.get("output_kind", "detections"))),
        enabled=bool(raw.get("enabled", True)),
    )


def _read_labels(value: object, *, label_set: object = None, default: list[str]) -> list[str]:
    if str(label_set).lower() == "coco80":
        return list(COCO80_LABELS)
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default)


if __name__ == "__main__":
    if shutil.which("gst-launch-1.0") is None:
        pass
    main()
