"""Service wrapper for the realtime eihead eye loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import threading
import time
from typing import Any, Protocol

from .realtime import REALTIME_STREAM_MODE


class RealtimeEyeAdapter(Protocol):
    def status(self) -> object:
        """Return the adapter's latest realtime status."""


class RealtimeEyeService:
    """Owns a realtime eye adapter and keeps the latest status/observation."""

    def __init__(
        self,
        *,
        adapter: RealtimeEyeAdapter | None = None,
        scene_bridge: Any | None = None,
        enable_scene_bridge: bool = True,
        poll_interval_s: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.adapter = adapter or self._default_adapter()
        self.scene_bridge = scene_bridge if scene_bridge is not None else self._default_scene_bridge(enable_scene_bridge)
        self.poll_interval_s = float(poll_interval_s)
        self._sleep = sleep
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_status: dict[str, Any] = self._coerce_status(self.adapter.status())
        self._latest_scene_key = ""
        self._latest_scene_result: dict[str, Any] | None = None

    @property
    def latest_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest_status)

    def status(self) -> dict[str, Any]:
        """Refresh and return the adapter's current status without forcing a frame poll."""

        status = self._coerce_status(self.adapter.status())
        with self._lock:
            self._latest_status = status
            return dict(self._latest_status)

    def poll_once(self) -> dict[str, Any]:
        """Poll one realtime frame from the adapter and cache the resulting status."""

        status = self._coerce_status(self._poll_adapter_once())
        with self._lock:
            self._latest_status = status
            return dict(self._latest_status)

    def poll(self, *, max_polls: int | None = None, interval_s: float | None = None) -> dict[str, Any]:
        """Run a bounded polling loop and return the latest status.

        ``max_polls`` defaults to one so direct calls cannot accidentally block
        forever. Use ``start()`` for the long-running background loop.
        """

        poll_count = 1 if max_polls is None else max(0, int(max_polls))
        delay_s = self.poll_interval_s if interval_s is None else float(interval_s)
        for index in range(poll_count):
            self.poll_once()
            if index < poll_count - 1 and delay_s > 0:
                self._sleep(delay_s)
        return self.latest_status

    def start(self) -> None:
        """Start the optional background polling loop."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_background_loop,
                name="eihead-realtime-eye-service",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout_s: float | None = 1.0) -> dict[str, Any]:
        """Stop the optional background polling loop and return the latest status."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        return self.latest_status

    def latest_observation(self) -> dict[str, Any]:
        """Return the latest realtime vision observation as a JSON-like dict."""

        status = self.latest_status
        detections = _json_list(status.get("detections"))
        boxes = _json_list(status.get("detection_boxes", status.get("boxes")))
        scores = list(status.get("detection_scores", status.get("scores", [])) or [])
        top_detection = _json_mapping(status.get("top_detection")) or (detections[0] if detections else None)
        captured_at_ts = _first_present(
            status,
            "last_frame_captured_at_ts",
            "captured_at_ts",
            "timestamp",
            "ts",
        )
        observation = {
            "kind": "realtime_vision_observation",
            "mode": status.get("mode") or REALTIME_STREAM_MODE,
            "frame_id": status.get("last_frame_id") or status.get("frame_id") or "",
            "width": _first_present(status, "width", "frame_width", "last_frame_width"),
            "height": _first_present(status, "height", "frame_height", "last_frame_height"),
            "detections": detections,
            "boxes": boxes,
            "scores": scores,
            "tracked_target": _tracked_target(top_detection),
            "captured_at_ts": captured_at_ts,
            "fps": status.get("fps", 0.0),
            "last_frame_age": status.get("last_frame_age", status.get("last_frame_age_s")),
            "status": status.get("status") or "unknown",
            "stream_ready": bool(status.get("stream_ready", False)),
            "placeholder": bool(status.get("placeholder", False)),
            "not_wired": bool(status.get("not_wired", False)),
            "compatibility_mode": bool(status.get("compatibility_mode", False)),
            "degraded": bool(status.get("degraded", False)),
            "stale": bool(status.get("stale", False)),
            "backend": status.get("backend") or "",
            "status_reason": status.get("status_reason") or "",
        }
        return self._attach_scene_bridge(observation)

    def _run_background_loop(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(max(0.0, self.poll_interval_s))

    def _poll_adapter_once(self) -> object:
        poll = getattr(self.adapter, "poll", None)
        if callable(poll):
            return poll()
        process_next = getattr(self.adapter, "process_next", None)
        if callable(process_next):
            return process_next()
        return self.adapter.status()

    @staticmethod
    def _default_adapter() -> RealtimeEyeAdapter:
        from .adapters import GStreamerHailoRealtimeAdapter

        return GStreamerHailoRealtimeAdapter()

    @staticmethod
    def _default_scene_bridge(enabled: bool) -> Any | None:
        if not enabled:
            return None
        from .scene import RealtimeVisionSceneBridge

        return RealtimeVisionSceneBridge()

    @staticmethod
    def _coerce_status(raw_status: object) -> dict[str, Any]:
        if hasattr(raw_status, "to_dict"):
            payload = raw_status.to_dict()
        elif isinstance(raw_status, Mapping):
            payload = raw_status
        else:
            payload = {"status": str(raw_status)}
        status = dict(payload)
        status.setdefault("mode", REALTIME_STREAM_MODE)
        status.setdefault("status", "unknown")
        status.setdefault("detections", [])
        status.setdefault("detection_boxes", [])
        status.setdefault("detection_scores", [])
        return status

    def _attach_scene_bridge(self, observation: dict[str, Any]) -> dict[str, Any]:
        bridge = self.scene_bridge
        if bridge is None:
            return observation
        cache_key = _scene_cache_key(observation)
        with self._lock:
            if cache_key == self._latest_scene_key and self._latest_scene_result is not None:
                scene_result = dict(self._latest_scene_result)
            else:
                scene_result = dict(bridge.update(observation))
                self._latest_scene_key = cache_key
                self._latest_scene_result = dict(scene_result)
        scene = _json_mapping(scene_result.get("scene_snapshot")) or {}
        events = _json_list(scene_result.get("event_contents") or scene_result.get("events"))
        tracks = _json_list(scene.get("objects"))
        diagnostics = _json_mapping(scene_result.get("diagnostics")) or {}
        stable_target = _json_mapping(scene_result.get("stable_target")) or _json_mapping(diagnostics.get("stable_target"))
        last_event = _json_mapping(scene_result.get("last_event")) or (events[-1] if events else None)
        observation.update(
            {
                "scene": scene,
                "sceneSnapshot": scene,
                "scene_id": scene_result.get("latest_scene_id") or scene.get("sceneId") or "",
                "scene_summary": scene_result.get("sceneGraphSummary") or scene.get("summary") or "",
                "sceneGraphSummary": scene_result.get("sceneGraphSummary") or scene.get("summary") or "",
                "event_summary": scene_result.get("event_summary") or scene.get("eventSummary") or "",
                "events": events,
                "tracks": tracks,
                "stable_target": stable_target,
                "scene_bridge": {
                    "kind": scene_result.get("kind"),
                    "live": scene_result.get("live"),
                    "reason": scene_result.get("reason"),
                    "object_count": scene_result.get("object_count", len(tracks)),
                    "track_count": scene_result.get("track_count", len(tracks)),
                    "fps": diagnostics.get("fps", observation.get("fps", 0.0)),
                    "frame_age": diagnostics.get("frame_age", observation.get("last_frame_age")),
                    "frame_age_s": diagnostics.get("frame_age_s", observation.get("last_frame_age")),
                    "stable_target": stable_target,
                    "event_count": scene_result.get("event_count", len(events)),
                    "last_event": last_event,
                },
            }
        )
        return observation


def _json_list(value: object) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [_json_value(item) for item in value]


def _json_mapping(value: object) -> dict[str, Any] | None:
    json_value = _json_value(value)
    return json_value if isinstance(json_value, dict) else None


def _json_value(value: object) -> Any:
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _tracked_target(top_detection: object) -> dict[str, Any] | None:
    detection = _json_mapping(top_detection)
    if not detection:
        return None
    target: dict[str, Any] = {}
    for key in ("label", "score", "confidence", "bbox", "track_id", "trackId", "class_id", "id"):
        if key in detection:
            target[key] = detection[key]
    if "score" not in target and "confidence" in target:
        target["score"] = target["confidence"]
    if "track_id" not in target and "trackId" in target:
        target["track_id"] = target["trackId"]
    return target or None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _scene_cache_key(observation: Mapping[str, Any]) -> str:
    payload = {
        "frame_id": observation.get("frame_id"),
        "captured_at_ts": observation.get("captured_at_ts"),
        "status": observation.get("status"),
        "mode": observation.get("mode"),
        "stream_ready": observation.get("stream_ready"),
        "placeholder": observation.get("placeholder"),
        "not_wired": observation.get("not_wired"),
        "compatibility_mode": observation.get("compatibility_mode"),
        "stale": observation.get("stale"),
        "backend": observation.get("backend"),
        "detections": observation.get("detections"),
    }
    return json.dumps(_json_value(payload), sort_keys=True, default=str)
