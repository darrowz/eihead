"""Eye organ implementation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.body.organs.base import BaseOrgan
from eibrain.body.runtime_linux import capture_frame, run_hailo_frame_inference
from eibrain.body.vision_state import DEFAULT_VISION_STATE_PATH, VisionStateReader
from eibrain.body.vision_state import summarize_detections


class EyeOrgan(BaseOrgan):
    name = "eye"
    subfunction_names = ("camera", "detection", "identity")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._cache_ttl_s = self._read_float_config("detection", "refresh_interval_s", default=0.5)
        frame_dir = Path(tempfile.gettempdir()) / "eibrain-eye"
        frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_path = frame_dir / "latest.jpg"
        self._cached_heartbeat: OrganHealth | None = None
        self._cached_heartbeat_at = 0.0
        self._vision_state_reader = self._build_vision_state_reader()

    @property
    def latest_frame_path(self) -> str | None:
        state_path = self._latest_state_frame_path()
        if state_path:
            return state_path
        if self._frame_path.exists():
            return str(self._frame_path)
        return None

    def passive_heartbeat(self) -> OrganHealth:
        if self._vision_state_reader is not None:
            return self.heartbeat()
        if self._cached_heartbeat is not None:
            return self._cached_heartbeat
        subfunctions = {
            name: SubfunctionHealth(
                name=name,
                health="healthy",
                details={"driver": self._driver_kind(name), "status": "live_probe_skipped"},
            )
            for name in self.subfunction_names
        }
        return OrganHealth(organ=self.name, health="healthy", subfunctions=subfunctions)

    def heartbeat(self) -> OrganHealth:
        if not self._visual_runtime_enabled():
            return super().heartbeat()
        now_ts = time.time()
        if self._cached_heartbeat is not None and now_ts - self._cached_heartbeat_at < self._cache_ttl_s:
            return self._cached_heartbeat
        if self._vision_state_reader is not None:
            heartbeat = self._heartbeat_from_vision_state(now_ts=now_ts)
            self._cached_heartbeat = heartbeat
            self._cached_heartbeat_at = now_ts
            return heartbeat

        camera_state = self._camera_health(now_ts=now_ts)
        detection_state = self._detection_health(camera_state=camera_state, now_ts=now_ts)
        identity_state = self._identity_health(detection_state=detection_state, now_ts=now_ts)
        subfunctions = {
            "camera": camera_state,
            "detection": detection_state,
            "identity": identity_state,
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        self._cached_heartbeat = OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)
        self._cached_heartbeat_at = now_ts
        return self._cached_heartbeat

    def _visual_runtime_enabled(self) -> bool:
        return any(
            self.config.subfunctions.get(name) is not None
            and self.config.subfunctions[name].driver.kind != "noop"
            for name in ("camera", "detection", "identity")
        )

    def read_visual_tracking_snapshot(self) -> dict[str, object]:
        """Read detections from vision state without probing camera/Hailo drivers."""
        if self._vision_state_reader is None:
            return {}
        try:
            snapshot = self._vision_state_reader.read()
        except Exception as exc:
            return {
                "detections": [],
                "detection_count": 0,
                "top_detection": None,
                "status": "state_unavailable",
                "error": str(exc),
                "source": "vision_state",
                "state_path": str(self._vision_state_reader.state_path),
            }
        return self._vision_state_detection_details(snapshot=snapshot, elapsed_ms=0.0)

    def _build_vision_state_reader(self) -> VisionStateReader | None:
        config = self._vision_state_config()
        if config is None:
            return None
        state_path = str(config.get("state_path", DEFAULT_VISION_STATE_PATH))
        stale_after_s = self._read_float_from_extra(config, "stale_after_s", default=3.0)
        return VisionStateReader(state_path, stale_after_s=stale_after_s)

    def _vision_state_config(self) -> dict[str, object] | None:
        for name in ("camera", "detection", "identity"):
            subfunction = self.config.subfunctions.get(name)
            if subfunction is None:
                continue
            extra = subfunction.driver.extra
            provider = str(extra.get("provider", "") or "")
            if provider in {"vision_state", "hailo8_service"} or extra.get("state_path"):
                return extra
        return None

    def _heartbeat_from_vision_state(self, *, now_ts: float) -> OrganHealth:
        started = time.perf_counter()
        try:
            snapshot = self._vision_state_reader.read(now_ts=now_ts) if self._vision_state_reader is not None else None
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            subfunctions = self._vision_state_unavailable_subfunctions(
                elapsed_ms=elapsed_ms,
                status="state_unavailable",
                error=str(exc),
            )
            return OrganHealth(organ=self.name, health="unavailable", subfunctions=subfunctions)
        if snapshot is None:
            subfunctions = self._vision_state_unavailable_subfunctions(
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                status="state_unavailable",
                error="vision_state_reader_not_configured",
            )
            return OrganHealth(organ=self.name, health="unavailable", subfunctions=subfunctions)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._vision_state_detection_details(snapshot=snapshot, elapsed_ms=elapsed_ms)
        status = str(details.get("service_status", "ok") or "ok")
        is_sleeping = status == "sleeping"
        camera_health = "degraded" if snapshot.stale or (status != "ok" and not is_sleeping) else "healthy"
        detection_health = camera_health
        camera_state = SubfunctionHealth(
            name="camera",
            health=camera_health,
            details={**details, "capture_result": {"status": status}},
        )
        detection_state = SubfunctionHealth(
            name="detection",
            health=detection_health,
            details=details,
        )
        identity_state = self._identity_from_detection_state(detection_state=detection_state, now_ts=now_ts)
        subfunctions = {"camera": camera_state, "detection": detection_state, "identity": identity_state}
        statuses = [state.health for state in subfunctions.values()]
        health = "healthy" if all(item == "healthy" for item in statuses) else "degraded"
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def _vision_state_detection_details(self, *, snapshot, elapsed_ms: float) -> dict[str, object]:
        payload = snapshot.payload
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        detections = [item for item in detections if isinstance(item, dict)]
        top_detection = payload.get("top_detection")
        if top_detection is not None and not isinstance(top_detection, dict):
            top_detection = None
        frame_path = payload.get("frame_path") or self._vision_state_frame_path()
        frame_captured_at_ts = payload.get("frame_captured_at_ts") or payload.get("updated_at_ts")
        status = str(payload.get("status", "ok") or "ok")
        details = {
            "driver": "vision_state",
            "elapsed_ms": elapsed_ms,
            "status": "stale" if snapshot.stale else ("live" if status == "ok" else status),
            "service_status": status,
            "source": "vision_state",
            "state_path": str(self._vision_state_reader.state_path),
            "state_age_s": snapshot.age_s,
            "state_updated_at_ts": payload.get("updated_at_ts"),
            "stale_after_s": self._vision_state_reader.stale_after_s,
            "frame_path": str(frame_path) if frame_path else None,
            "frame_captured_at_ts": frame_captured_at_ts,
            "frame_updated_at_ts": frame_captured_at_ts,
            "backend": payload.get("backend", "hailo8_service"),
            "details": dict(payload.get("details", {})) if isinstance(payload.get("details"), dict) else {},
            "fps": payload.get("fps"),
            "pipeline": dict(payload.get("pipeline", {})) if isinstance(payload.get("pipeline"), dict) else {},
            "telemetry": dict(payload.get("telemetry", {})) if isinstance(payload.get("telemetry"), dict) else {},
            "detections": detections,
            "detection_count": len(detections),
            "top_detection": top_detection or (detections[0] if detections else None),
            "scene_labels": payload.get("scene_labels")
            if isinstance(payload.get("scene_labels"), list)
            else sorted({str(item.get("label", "unknown")) for item in detections}),
            "scene_summary": str(payload.get("scene_summary") or summarize_detections(detections)),
        }
        if snapshot.stale:
            details["error"] = "vision_state_stale"
        elif status not in {"ok", "sleeping"}:
            details["error"] = payload.get("error", status)
        return details

    def _vision_state_unavailable_subfunctions(
        self,
        *,
        elapsed_ms: float,
        status: str,
        error: str,
    ) -> dict[str, SubfunctionHealth]:
        state_path = str(self._vision_state_reader.state_path) if self._vision_state_reader is not None else ""
        common = {
            "driver": "vision_state",
            "elapsed_ms": elapsed_ms,
            "status": status,
            "state_path": state_path,
            "frame_path": None,
            "frame_captured_at_ts": None,
            "error": error,
            "details": {},
        }
        return {
            "camera": SubfunctionHealth(name="camera", health="unavailable", details=dict(common)),
            "detection": SubfunctionHealth(
                name="detection",
                health="unavailable",
                details={
                    **common,
                    "detections": [],
                    "detection_count": 0,
                    "top_detection": None,
                    "scene_summary": "vision state unavailable",
                },
            ),
            "identity": SubfunctionHealth(
                name="identity",
                health="unavailable",
                details={
                    **common,
                    "identity_candidates": [],
                    "face_candidate_count": 0,
                    "identity_summary": "identity chain blocked by detection",
                },
            ),
        }

    def _latest_state_frame_path(self) -> str | None:
        if self._vision_state_reader is None:
            return None
        frame_path = self._vision_state_frame_path()
        return frame_path if frame_path and Path(frame_path).exists() else None

    def _vision_state_frame_path(self) -> str | None:
        config = self._vision_state_config()
        if config is None:
            return None
        raw = config.get("frame_path")
        return str(raw) if raw else None

    def _camera_health(self, *, now_ts: float) -> SubfunctionHealth:
        if self._driver_kind("camera") == "noop":
            return self._subfunction_health("camera")
        config = self.config.subfunctions.get("camera")
        device = str(config.driver.extra.get("device", "/dev/video0")) if config is not None else "/dev/video0"
        input_format = str(config.driver.extra.get("input_format", "") or "") if config is not None else ""
        video_size = str(config.driver.extra.get("video_size", "") or "") if config is not None else ""
        timeout_s = self._read_float_config("camera", "timeout_s", default=5.0)
        started = time.perf_counter()
        probe = self.drivers["camera"].heartbeat()
        capture_result = capture_frame(
            device=device,
            output_path=self._frame_path,
            input_format=input_format,
            video_size=video_size,
            timeout_s=timeout_s,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._merge_probe_details(
            probe=probe.details,
            elapsed_ms=elapsed_ms,
            status="healthy" if capture_result.get("status") == "ok" else "capture_failed",
        )
        details.update(
            {
                "device": device,
                "input_format": input_format,
                "video_size": video_size,
                "timeout_s": timeout_s,
                "frame_path": str(self._frame_path),
                "frame_captured_at_ts": now_ts,
                "capture_result": dict(capture_result.get("details", {})),
            }
        )
        if capture_result.get("status") == "ok":
            health = "healthy"
        else:
            health = "unavailable" if probe.status == "unavailable" else "degraded"
            capture_details = capture_result.get("details", {})
            if isinstance(capture_details, dict):
                details["error"] = capture_details.get("stderr") or capture_details.get("stdout") or "capture_failed"
        return SubfunctionHealth(name="camera", health=health, details=details)

    def _detection_health(
        self,
        *,
        camera_state: SubfunctionHealth,
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("detection") == "noop":
            return self._subfunction_health("detection")
        config = self.config.subfunctions.get("detection")
        probe = self.drivers["detection"].heartbeat()
        probe_details = dict(probe.details)
        started = time.perf_counter()
        if camera_state.health != "healthy":
            details = self._merge_probe_details(
                probe=probe_details,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                status="camera_unavailable",
            )
            details.update(
                {
                    "frame_path": self.latest_frame_path,
                    "frame_captured_at_ts": now_ts,
                    "detections": [],
                    "detection_count": 0,
                    "scene_summary": "camera unavailable",
                    "error": "camera_unavailable",
                }
            )
            return SubfunctionHealth(name="detection", health="unavailable", details=details)

        hef_path = str(config.driver.extra.get("hef_path", "/usr/share/hailo-models/yolov5s_personface_h8l.hef")) if config is not None else "/usr/share/hailo-models/yolov5s_personface_h8l.hef"
        score_threshold = self._read_float_config("detection", "score_threshold", default=0.3)
        labels = self._read_label_config("detection", default=["person", "face"])
        inference = run_hailo_frame_inference(
            image_path=self._frame_path,
            hef_path=hef_path,
            labels=labels,
            score_threshold=score_threshold,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        inference_details = dict(inference.get("details", {}))
        detections = inference_details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        details = self._merge_probe_details(
            probe=probe_details,
            elapsed_ms=elapsed_ms,
            status="healthy" if inference.get("status") == "ok" else "inference_failed",
        )
        details.update(
            {
                "frame_path": str(self._frame_path),
                "frame_captured_at_ts": now_ts,
                "hef_path": hef_path,
                "score_threshold": score_threshold,
                "labels": labels,
                "detections": detections,
                "detection_count": len(detections),
                "top_detection": detections[0] if detections else None,
                "scene_labels": sorted({str(item.get("label", "unknown")) for item in detections}),
                "scene_summary": self._summarize_detections(detections),
                "inference_details": inference_details,
            }
        )
        if inference.get("status") == "ok":
            return SubfunctionHealth(name="detection", health="healthy", details=details)
        details["error"] = inference_details.get("error") or inference_details.get("stderr") or inference_details.get("reason")
        health = "unavailable" if probe.status == "unavailable" else "degraded"
        return SubfunctionHealth(name="detection", health=health, details=details)

    def _identity_health(
        self,
        *,
        detection_state: SubfunctionHealth,
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("identity") == "noop":
            return self._subfunction_health("identity")
        probe = self.drivers["identity"].heartbeat()
        probe_details = dict(probe.details)
        started = time.perf_counter()
        detections = detection_state.details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        face_candidates = [item for item in detections if str(item.get("label")) == "face"]
        identity_candidates = [
            {
                "candidate_id": f"unknown-face-{index + 1}",
                "identity": "unknown",
                "score": candidate.get("score"),
                "bbox": candidate.get("bbox"),
            }
            for index, candidate in enumerate(face_candidates)
        ]
        if detection_state.health != "healthy":
            status = "detection_unavailable"
            health = "unavailable"
        elif identity_candidates:
            status = "observing_unknown_face"
            health = "healthy"
        else:
            status = "no_face_candidates"
            health = "healthy"
        details = self._merge_probe_details(
            probe=probe_details,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            status=status,
        )
        details.update(
            {
                "frame_path": self.latest_frame_path,
                "frame_captured_at_ts": now_ts,
                "identity_candidates": identity_candidates,
                "face_candidate_count": len(identity_candidates),
                "identity_summary": self._summarize_identity(identity_candidates, status=status),
            }
        )
        if health != "healthy":
            details["error"] = status
        return SubfunctionHealth(name="identity", health=health, details=details)

    def _identity_from_detection_state(
        self,
        *,
        detection_state: SubfunctionHealth,
        now_ts: float,
    ) -> SubfunctionHealth:
        detections = detection_state.details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        face_candidates = [item for item in detections if isinstance(item, dict) and str(item.get("label")) == "face"]
        identity_candidates = [
            {
                "candidate_id": f"unknown-face-{index + 1}",
                "identity": "unknown",
                "score": candidate.get("score"),
                "bbox": candidate.get("bbox"),
            }
            for index, candidate in enumerate(face_candidates)
        ]
        if detection_state.health != "healthy":
            status = "detection_unavailable"
            health = "unavailable"
        elif identity_candidates:
            status = "observing_unknown_face"
            health = "healthy"
        else:
            status = "no_face_candidates"
            health = "healthy"
        details = {
            "driver": "vision_state",
            "status": status,
            "source": "vision_state",
            "frame_path": self.latest_frame_path,
            "frame_captured_at_ts": now_ts,
            "identity_candidates": identity_candidates,
            "face_candidate_count": len(identity_candidates),
            "identity_summary": self._summarize_identity(identity_candidates, status=status),
        }
        if health != "healthy":
            details["error"] = status
        return SubfunctionHealth(name="identity", health=health, details=details)

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    def _read_float_config(self, subfunction_name: str, key: str, *, default: float) -> float:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return default
        value = config.driver.extra.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_label_config(self, subfunction_name: str, *, default: list[str]) -> list[str]:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return list(default)
        raw = config.driver.extra.get("labels", default)
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return list(default)

    @staticmethod
    def _read_float_from_extra(extra: dict[str, object], key: str, *, default: float) -> float:
        value = extra.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _merge_probe_details(
        *,
        probe: dict[str, object],
        elapsed_ms: float,
        status: str,
    ) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        merged["elapsed_ms"] = elapsed_ms
        merged["status"] = status
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged

    @staticmethod
    def _summarize_detections(detections: list[dict[str, object]]) -> str:
        return summarize_detections(detections)

    @staticmethod
    def _summarize_identity(identity_candidates: list[dict[str, object]], *, status: str) -> str:
        if identity_candidates:
            return f"{len(identity_candidates)} unknown face candidate(s)"
        if status == "detection_unavailable":
            return "identity chain blocked by detection"
        return "no recognizable face candidate in current frame"
