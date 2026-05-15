"""Realtime eye scene/event aggregation bridge."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import math
from typing import Any

RealtimeVisionSimulator = None  # type: ignore[assignment]
to_eiprotocol_event_contents = None  # type: ignore[assignment]
to_eiprotocol_scene_content = None  # type: ignore[assignment]

from .realtime import COMPAT_STATIC_FRAME_MODE, REALTIME_STREAM_MODE


_DEVICE_LABELS = {"phone", "mobile", "smartphone", "tablet", "laptop", "monitor", "screen", "device"}
_HAND_KEYPOINTS = {"left_wrist", "right_wrist", "left_hand", "right_hand", "hand", "wrist"}


class RealtimeVisionSceneBridge:
    """Convert realtime eye observations into scene snapshots and events."""

    def __init__(
        self,
        *,
        simulator: RealtimeVisionSimulator | None = None,
        match_distance: float = 0.35,
        move_threshold: float = 0.12,
        max_missing_frames: int = 1,
    ) -> None:
        self.simulator = simulator or _new_simulator(
            match_distance=match_distance,
            move_threshold=move_threshold,
            max_missing_frames=max_missing_frames,
        )
        self.latest_scene_id = ""

    def update(self, observation_or_status: Mapping[str, Any]) -> dict[str, Any]:
        """Aggregate one service observation/status dict into protocol content."""

        observation = dict(observation_or_status)
        frame_id = _frame_id(observation)
        observed_at = _observed_at(observation)
        live, reason = _live_state(observation)
        if not live:
            return self._non_live_result(frame_id=frame_id, observed_at=observed_at, reason=reason)

        detections = _detections(observation)
        snapshot = self.simulator.update(
            frame_id=frame_id,
            observed_at=observed_at,
            detections=detections,
        )
        scene_snapshot = _scene_content(snapshot)
        _augment_scene_with_detection_modalities(scene_snapshot, detections)
        _attach_tracking_diagnostics(scene_snapshot)
        _attach_observation_metadata(scene_snapshot, observation)
        event_contents = _event_contents(snapshot)
        event_contents.extend(_lightweight_event_contents(scene_snapshot, observed_at=observed_at, frame_id=frame_id))
        self.latest_scene_id = str(scene_snapshot.get("sceneId", ""))
        object_count = len(scene_snapshot.get("objects", []))
        stable_target = _stable_target_from_scene(scene_snapshot)
        last_event = dict(event_contents[-1]) if event_contents else None
        diagnostics = _diagnostics(
            observation,
            track_count=object_count,
            stable_target=stable_target,
            event_count=len(event_contents),
            last_event=last_event,
        )

        return {
            "kind": "realtime_vision_scene_bridge",
            "mode": REALTIME_STREAM_MODE,
            "status": "ok",
            "stream_ready": True,
            "not_wired": False,
            "stale": False,
            "live": True,
            "reason": "live",
            "frame_id": frame_id,
            "observed_at": observed_at,
            "latest_scene_id": self.latest_scene_id,
            "scene_id": self.latest_scene_id,
            "scene": scene_snapshot,
            "scene_snapshot": scene_snapshot,
            "scene_summary": str(snapshot.get("sceneGraphSummary", "")),
            "sceneGraphSummary": str(snapshot.get("sceneGraphSummary", "")),
            "event_summary": str(
                snapshot.get("eventSummary")
                or scene_snapshot.get("eventSummary")
                or _event_summary(event_contents)
            ),
            "event_contents": event_contents,
            "events": event_contents,
            "tracks": [dict(item) for item in scene_snapshot.get("objects", []) if isinstance(item, Mapping)],
            "target": _target_from_scene(scene_snapshot),
            "stable_target": stable_target,
            "object_count": object_count,
            "track_count": object_count,
            "event_count": len(event_contents),
            "last_event": last_event,
            "diagnostics": diagnostics,
        }

    def _non_live_result(self, *, frame_id: str, observed_at: str, reason: str) -> dict[str, Any]:
        scene_snapshot = {
            "sceneId": "",
            "observedAt": observed_at,
            "summary": f"Realtime vision observation is non-live: {reason}",
            "objects": [],
            "relationships": [],
            "environment": {"source": "eihead.eye.scene", "live": False},
            "imageUrl": "",
            "metadata": {
                "frameId": frame_id,
                "realtime": False,
                "reason": reason,
                "trackCount": 0,
            },
        }
        return {
            "kind": "realtime_vision_scene_bridge",
            "mode": REALTIME_STREAM_MODE,
            "status": reason,
            "stream_ready": False,
            "not_wired": reason in {"not_wired", "placeholder"},
            "stale": reason == "stale",
            "live": False,
            "reason": reason,
            "frame_id": frame_id,
            "observed_at": observed_at,
            "latest_scene_id": self.latest_scene_id,
            "scene_id": "",
            "scene": scene_snapshot,
            "scene_snapshot": scene_snapshot,
            "scene_summary": scene_snapshot["summary"],
            "sceneGraphSummary": scene_snapshot["summary"],
            "event_summary": "",
            "event_contents": [],
            "events": [],
            "tracks": [],
            "target": None,
            "stable_target": None,
            "object_count": 0,
            "track_count": 0,
            "event_count": 0,
            "last_event": None,
            "diagnostics": {
                "fps": 0.0,
                "frame_age": None,
                "frame_age_s": None,
                "track_count": 0,
                "stable_target": None,
                "event_count": 0,
                "last_event": None,
            },
        }


def _live_state(observation: Mapping[str, Any]) -> tuple[bool, str]:
    status = str(observation.get("status", "")).strip().lower()
    mode = str(observation.get("mode", "") or REALTIME_STREAM_MODE).strip().lower()
    backend = str(observation.get("backend", "")).strip().lower()

    if _truthy(observation.get("not_wired")) or status == "not_wired":
        return False, "not_wired"
    if _truthy(observation.get("placeholder")) or backend == "placeholder":
        return False, "placeholder"
    if (
        _truthy(observation.get("compatibility_mode"))
        or mode == COMPAT_STATIC_FRAME_MODE
        or status in {"compat_static", "compat_static_frame", "compat_static_frame_test_only"}
    ):
        return False, "compat_static"
    if status == "static":
        return False, "static"
    if _truthy(observation.get("stale")) or status == "stale":
        return False, "stale"
    return True, "live"


def _frame_id(observation: Mapping[str, Any]) -> str:
    return str(observation.get("frame_id") or observation.get("frameId") or observation.get("last_frame_id") or "")


def _observed_at(observation: Mapping[str, Any]) -> str:
    for key in ("observed_at", "observedAt", "captured_at", "capturedAt"):
        value = observation.get(key)
        if value:
            return str(value)
    for key in ("captured_at_ts", "last_frame_captured_at_ts", "timestamp", "ts"):
        value = observation.get(key)
        if value is None:
            continue
        try:
            return datetime.fromtimestamp(float(value), tz=UTC).isoformat(timespec="milliseconds")
        except (OSError, OverflowError, TypeError, ValueError):
            continue
    return ""


def _detections(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = observation.get("detections")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _target_from_scene(scene_snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    stable_target = _stable_target_from_scene(scene_snapshot)
    if stable_target:
        return stable_target
    attention = scene_snapshot.get("attention")
    if isinstance(attention, Mapping) and attention:
        track_id = str(attention.get("trackId") or "")
        for item in scene_snapshot.get("objects", []):
            if isinstance(item, Mapping) and str(item.get("trackId") or "") == track_id:
                return _target_from_object(item)
    for item in scene_snapshot.get("objects", []):
        if isinstance(item, Mapping):
            return _target_from_object(item)
    return None


def _stable_target_from_scene(scene_snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("stableTarget", "stable_target", "attention"):
        value = scene_snapshot.get(key)
        if not isinstance(value, Mapping) or not value:
            continue
        track_id = str(value.get("trackId") or value.get("track_id") or "")
        if track_id:
            for item in scene_snapshot.get("objects", []):
                if isinstance(item, Mapping) and str(item.get("trackId") or item.get("track_id") or "") == track_id:
                    return _target_from_object(item)
        return _target_from_object(value)
    return None


def _target_from_object(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "track_id": item.get("trackId"),
        "trackId": item.get("trackId"),
        "label": item.get("label"),
        "score": item.get("confidence"),
        "center": dict(item.get("center")) if isinstance(item.get("center"), Mapping) else None,
        "bbox": dict(item.get("bbox")) if isinstance(item.get("bbox"), Mapping) else None,
        "temporalState": item.get("temporalState"),
    }


def _diagnostics(
    observation: Mapping[str, Any],
    *,
    track_count: int,
    stable_target: Mapping[str, Any] | None,
    event_count: int,
    last_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    frame_age = _number_or_none(
        observation.get("last_frame_age", observation.get("last_frame_age_s", observation.get("frame_age")))
    )
    frame_age_ms = _number_or_none(observation.get("frame_age_ms"))
    if frame_age_ms is None and frame_age is not None:
        frame_age_ms = frame_age * 1000.0
    soak_summary = observation.get("soak_summary")
    soak_summary = dict(soak_summary) if isinstance(soak_summary, Mapping) else {}
    hailo_metadata = observation.get("hailo_metadata")
    hailo_metadata = dict(hailo_metadata) if isinstance(hailo_metadata, Mapping) else {}
    track_id_switch_count = int(_number_or_zero(soak_summary.get("track_id_switch_count", 0)))
    target_stability_ratio = _number_or_none(soak_summary.get("target_stability_ratio"))
    event_rate_hz = _number_or_none(soak_summary.get("event_rate_hz"))
    frame_drop_tolerance = int(_number_or_zero(soak_summary.get("frame_drop_tolerance", 0)))
    return {
        "fps": _number_or_zero(observation.get("fps")),
        "frame_age": frame_age,
        "frame_age_s": frame_age,
        "frame_age_ms": frame_age_ms,
        "p95_frame_age_ms": _number_or_none(soak_summary.get("p95_frame_age_ms")) or frame_age_ms,
        "track_count": int(track_count),
        "stable_target": dict(stable_target) if isinstance(stable_target, Mapping) else None,
        "event_count": int(event_count),
        "last_event": dict(last_event) if isinstance(last_event, Mapping) else None,
        "track_id_switch_count": track_id_switch_count,
        "target_stability_ratio": target_stability_ratio,
        "event_rate_hz": event_rate_hz,
        "frame_drop_tolerance": frame_drop_tolerance,
        "hailo_metadata": hailo_metadata,
        "soak_summary": soak_summary,
        "trace": {
            "kind": "vision_tracking_diagnostics",
            "metrics": {
                "track_id_switch_count": track_id_switch_count,
                "target_stability_ratio": target_stability_ratio,
                "event_rate_hz": event_rate_hz,
                "frame_drop_tolerance": frame_drop_tolerance,
                "p95_frame_age_ms": _number_or_none(soak_summary.get("p95_frame_age_ms")) or frame_age_ms,
            },
        },
    }


def _attach_observation_metadata(scene_snapshot: dict[str, Any], observation: Mapping[str, Any]) -> None:
    metadata = scene_snapshot.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        scene_snapshot["metadata"] = metadata
    hailo_metadata = observation.get("hailo_metadata")
    if isinstance(hailo_metadata, Mapping):
        metadata["hailo"] = dict(hailo_metadata)
    soak_summary = observation.get("soak_summary")
    if isinstance(soak_summary, Mapping):
        metadata["soak_summary"] = dict(soak_summary)


def _attach_tracking_diagnostics(scene_snapshot: dict[str, Any]) -> None:
    objects = [item for item in scene_snapshot.get("objects", []) if isinstance(item, Mapping)]
    for item in objects:
        diagnostics = item.get("trackingDiagnostics")
        if isinstance(diagnostics, Mapping):
            scene_snapshot.setdefault("trackingDiagnostics", dict(diagnostics))
            return


def _augment_scene_with_detection_modalities(scene_snapshot: dict[str, Any], detections: list[Mapping[str, Any]]) -> None:
    objects = [item for item in scene_snapshot.get("objects", []) if isinstance(item, dict)]
    used_detection_indexes: set[int] = set()
    for obj in objects:
        detection_index = _best_detection_index(obj, detections, used_detection_indexes)
        if detection_index is None:
            continue
        used_detection_indexes.add(detection_index)
        raw = detections[detection_index]
        source = _first_text(raw.get("source"), raw.get("provider"), obj.get("source"))
        model_id = _first_text(raw.get("model_id"), raw.get("modelId"), raw.get("model"), obj.get("model_id"))
        if source:
            obj["source"] = source
        if model_id:
            obj["model_id"] = model_id
        provenance = dict(raw.get("provenance")) if isinstance(raw.get("provenance"), Mapping) else {}
        if source:
            provenance.setdefault("source", source)
        if model_id:
            provenance.setdefault("model_id", model_id)
        if provenance:
            obj["provenance"] = provenance
        pose = _normalize_pose(_raw_value(raw, "pose"))
        if pose:
            obj["pose"] = pose
            obj["keypoints"] = list(pose["keypoints"])
        clip_labels = _normalize_label_annotations(_raw_value(raw, "clip_labels", "clipLabels"))
        if clip_labels:
            obj["clip_labels"] = clip_labels
        semantic_labels = _normalize_semantic_labels(_raw_value(raw, "semantic_labels", "semanticLabels"))
        if semantic_labels:
            obj["semantic_labels"] = semantic_labels
        tracking_diagnostics = _raw_value(raw, "tracking_diagnostics", "trackingDiagnostics")
        if isinstance(tracking_diagnostics, Mapping):
            obj["trackingDiagnostics"] = dict(tracking_diagnostics)
        depth_m = _structured_depth_m(raw) or _number_or_none(_raw_value(raw, "depth_m", "distance_m", "z_m"))
        if depth_m is not None:
            obj["depth_m"] = round(depth_m, 3)
        distance_band = _first_text(_raw_value(raw, "distance_band", "depth_band"))
        if not distance_band and depth_m is not None:
            distance_band = "near" if depth_m <= 1.0 else "far"
        if distance_band:
            obj["distance_band"] = distance_band
        looking_at_device = _optional_bool(_raw_value(raw, "looking_at_device", "lookingAtDevice"))
        if looking_at_device is not None:
            obj["looking_at_device"] = looking_at_device


def _best_detection_index(
    obj: Mapping[str, Any],
    detections: list[Mapping[str, Any]],
    used_indexes: set[int],
) -> int | None:
    label = _first_text(obj.get("label"))
    obj_track_id = _track_id(obj)
    obj_source_track_id = _first_text(obj.get("sourceTrackId"), obj.get("source_track_id"), obj.get("rawTrackId"))
    if obj_track_id:
        for index, raw in enumerate(detections):
            if index in used_indexes:
                continue
            raw_track_id = _track_id(raw)
            if raw_track_id == obj_track_id or (obj_source_track_id and raw_track_id == obj_source_track_id):
                return index
    bbox = obj.get("bbox")
    obj_center = _center(bbox) if isinstance(bbox, Mapping) else (0.0, 0.0)
    candidates: list[tuple[float, int]] = []
    for index, raw in enumerate(detections):
        if index in used_indexes or _first_text(raw.get("label"), raw.get("name"), raw.get("class")) != label:
            continue
        raw_bbox = _normalize_bbox(raw.get("bbox"), format_hint=_bbox_format(raw))
        if raw_bbox is None:
            continue
        candidates.append((_distance(obj_center, _center(raw_bbox)), index))
    if not candidates:
        return None
    return min(candidates)[1]


def _lightweight_event_contents(
    scene_snapshot: Mapping[str, Any],
    *,
    observed_at: str,
    frame_id: str,
) -> list[dict[str, Any]]:
    objects = [item for item in scene_snapshot.get("objects", []) if isinstance(item, Mapping)]
    scene_id = str(scene_snapshot.get("sceneId", ""))
    events: list[dict[str, Any]] = []
    for item in objects:
        if item.get("looking_at_device") is True:
            target = _nearest_device(item, objects)
            events.append(
                _lightweight_event_content(
                    event_type="looking_at_device",
                    scene_id=scene_id,
                    observed_at=observed_at,
                    frame_id=frame_id,
                    subject=item,
                    obj=target,
                    confidence=_number_or_zero(item.get("confidence")),
                )
            )
    for subject in objects:
        for obj in objects:
            if _track_id(subject) == _track_id(obj):
                continue
            if not _hand_near_object(subject, obj):
                continue
            events.append(
                _lightweight_event_content(
                    event_type="hand_near_object",
                    scene_id=scene_id,
                    observed_at=observed_at,
                    frame_id=frame_id,
                    subject=subject,
                    obj=obj,
                    confidence=min(_number_or_zero(subject.get("confidence")), _number_or_zero(obj.get("confidence"))),
                )
            )
    return sorted(events, key=lambda item: (str(item["eventType"]), str(item["subject"].get("trackId")), str(item["details"].get("objectId", ""))))


def _lightweight_event_content(
    *,
    event_type: str,
    scene_id: str,
    observed_at: str,
    frame_id: str,
    subject: Mapping[str, Any],
    obj: Mapping[str, Any] | None,
    confidence: float,
) -> dict[str, Any]:
    track_id = _track_id(subject)
    details: dict[str, Any] = {"frameId": frame_id}
    if obj is not None:
        details.update({"objectId": _track_id(obj), "objectLabel": _first_text(obj.get("label"))})
    metadata = dict(subject.get("provenance")) if isinstance(subject.get("provenance"), Mapping) else {}
    metadata["frameId"] = frame_id
    source = _first_text(subject.get("source"))
    model_id = _first_text(subject.get("model_id"))
    if source:
        metadata.setdefault("source", source)
    if model_id:
        metadata.setdefault("model_id", model_id)
    object_id = _track_id(obj) if obj is not None else ""
    object_suffix = f":{object_id}" if object_id else ""
    return {
        "eventId": f"{scene_id}:{event_type}:{track_id}{object_suffix}",
        "eventType": event_type,
        "observedAt": observed_at,
        "sceneId": scene_id,
        "subject": {"trackId": track_id, "label": subject.get("label")},
        "confidence": round(float(confidence), 3),
        "details": details,
        "metadata": metadata,
    }


def _normalize_pose(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("keypoints"), list):
        return {}
    keypoints: list[dict[str, Any]] = []
    for item in raw["keypoints"]:
        if not isinstance(item, Mapping):
            continue
        name = _first_text(item.get("name"), item.get("label"), item.get("part"))
        x = _number_or_none(item.get("x"))
        y = _number_or_none(item.get("y"))
        if not name or x is None or y is None:
            continue
        point: dict[str, Any] = {"name": name, "x": round(_clip01(x), 4), "y": round(_clip01(y), 4)}
        confidence = _number_or_none(item.get("confidence", item.get("score")))
        if confidence is not None:
            point["confidence"] = round(confidence, 3)
        keypoints.append(point)
    return {"keypoints": keypoints} if keypoints else {}


def _normalize_label_annotations(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    labels: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            label = _first_text(item.get("label"), item.get("name"), item.get("text"))
            if not label:
                continue
            normalized: dict[str, Any] = {"label": label}
            confidence = _number_or_none(item.get("confidence", item.get("score")))
            if confidence is not None:
                normalized["confidence"] = round(confidence, 3)
            source = _first_text(item.get("source"))
            if source:
                normalized["source"] = source
            labels.append(normalized)
        else:
            label = _first_text(item)
            if label:
                labels.append({"label": label})
    return labels


def _normalize_semantic_labels(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels: list[str] = []
    for item in raw:
        label = _first_text(item.get("label") if isinstance(item, Mapping) else item)
        if label and label not in labels:
            labels.append(label)
    return labels


def _structured_depth_m(raw: Mapping[str, Any]) -> float | None:
    depth = _raw_value(raw, "depth")
    if isinstance(depth, Mapping):
        for key in ("median", "subjectMedian", "subject_median", "meters", "m", "value"):
            value = _number_or_none(depth.get(key))
            if value is not None:
                return value
    distance = _raw_value(raw, "distance")
    if isinstance(distance, Mapping):
        for key in ("fromCameraM", "trackedTargetM", "nearestObjectM", "meters", "m", "value"):
            value = _number_or_none(distance.get(key))
            if value is not None:
                return value
    return None


def _hand_near_object(subject: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    bbox = obj.get("bbox")
    if not isinstance(bbox, Mapping):
        return False
    return any(_point_bbox_gap(point, bbox) <= 0.12 for point in _hand_points(subject))


def _hand_points(subject: Mapping[str, Any]) -> list[tuple[float, float]]:
    pose = subject.get("pose")
    keypoints = pose.get("keypoints") if isinstance(pose, Mapping) else subject.get("keypoints")
    if not isinstance(keypoints, list):
        return []
    points: list[tuple[float, float]] = []
    for item in keypoints:
        if not isinstance(item, Mapping):
            continue
        name = _first_text(item.get("name"), item.get("label"), item.get("part")).lower()
        if name in _HAND_KEYPOINTS:
            points.append((_number_or_zero(item.get("x")), _number_or_zero(item.get("y"))))
    return points


def _point_bbox_gap(point: tuple[float, float], bbox: Mapping[str, Any]) -> float:
    x, y = point
    horizontal_gap = max(_number_or_zero(bbox.get("x_min")) - x, x - _number_or_zero(bbox.get("x_max")), 0.0)
    vertical_gap = max(_number_or_zero(bbox.get("y_min")) - y, y - _number_or_zero(bbox.get("y_max")), 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _nearest_device(subject: Mapping[str, Any], objects: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    devices = [item for item in objects if _track_id(item) != _track_id(subject) and _is_device(item)]
    if not devices:
        return None
    subject_center = _object_center(subject)
    return min(devices, key=lambda item: _distance(subject_center, _object_center(item)))


def _is_device(item: Mapping[str, Any]) -> bool:
    labels = {_first_text(item.get("label")).lower()}
    labels.update(label.lower() for label in _normalize_semantic_labels(item.get("semantic_labels")))
    for clip_label in _normalize_label_annotations(item.get("clip_labels")):
        labels.add(_first_text(clip_label.get("label")).lower())
    return bool(labels & _DEVICE_LABELS)


def _object_center(item: Mapping[str, Any]) -> tuple[float, float]:
    center = item.get("center")
    if isinstance(center, Mapping):
        return (_number_or_zero(center.get("x")), _number_or_zero(center.get("y")))
    bbox = item.get("bbox")
    return _center(bbox) if isinstance(bbox, Mapping) else (0.0, 0.0)


def _track_id(item: Mapping[str, Any]) -> str:
    return _first_text(item.get("trackId"), item.get("track_id"), item.get("id"), item.get("stable_id"))


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _number_or_zero(value: Any) -> float:
    number = _number_or_none(value)
    return 0.0 if number is None else number


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _new_simulator(*, match_distance: float, move_threshold: float, max_missing_frames: int) -> Any:
    if RealtimeVisionSimulator is not None:
        return RealtimeVisionSimulator(
            match_distance=match_distance,
            move_threshold=move_threshold,
            max_missing_frames=max_missing_frames,
        )
    return _FallbackRealtimeVisionSimulator(
        match_distance=match_distance,
        move_threshold=move_threshold,
        max_missing_frames=max_missing_frames,
    )


def _scene_content(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if to_eiprotocol_scene_content is not None:
        return to_eiprotocol_scene_content(snapshot)
    scene = snapshot.get("sceneSnapshot")
    scene = scene if isinstance(scene, Mapping) else snapshot
    return {
        "sceneId": str(scene.get("sceneId", "")),
        "observedAt": str(scene.get("observedAt", "")),
        "summary": str(snapshot.get("sceneGraphSummary") or scene.get("summary") or ""),
        "objects": [dict(item) for item in _dict_list(scene.get("objects"))],
        "relationships": [dict(item) for item in _dict_list(scene.get("relationships"))],
        "environment": {"source": "eihead.eye.scene"},
        "imageUrl": "",
        "metadata": dict(scene.get("metadata")) if isinstance(scene.get("metadata"), Mapping) else {},
        "attention": dict(scene.get("attention")) if isinstance(scene.get("attention"), Mapping) else {},
        "stableTarget": dict(scene.get("stableTarget")) if isinstance(scene.get("stableTarget"), Mapping) else {},
        "eventSummary": str(snapshot.get("eventSummary") or scene.get("eventSummary") or ""),
    }


def _event_contents(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    if to_eiprotocol_event_contents is not None:
        return to_eiprotocol_event_contents(snapshot)
    contents: list[dict[str, Any]] = []
    for event in _dict_list(snapshot.get("events")):
        content = {
            "eventId": str(event.get("eventId", "")),
            "eventType": str(event.get("eventType", "")),
            "observedAt": str(event.get("observedAt", "")),
            "sceneId": str(event.get("sceneId", "")),
            "subject": dict(event.get("subject")) if isinstance(event.get("subject"), Mapping) else {},
            "confidence": event.get("confidence"),
            "details": dict(event.get("details")) if isinstance(event.get("details"), Mapping) else {},
            "metadata": dict(event.get("metadata")) if isinstance(event.get("metadata"), Mapping) else {},
        }
        tracking_diagnostics = event.get("trackingDiagnostics")
        if isinstance(tracking_diagnostics, Mapping):
            content["trackingDiagnostics"] = dict(tracking_diagnostics)
        contents.append(content)
    return contents


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


class _FallbackRealtimeVisionSimulator:
    """Small eihead-local tracker used when the full eibrain package is absent."""

    def __init__(self, *, match_distance: float, move_threshold: float, max_missing_frames: int) -> None:
        self.match_distance = float(match_distance)
        self.move_threshold = float(move_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks: dict[str, dict[str, Any]] = {}
        self._next_ids: dict[str, int] = {}

    def update(self, *, frame_id: str, observed_at: str, detections: list[Mapping[str, Any]]) -> dict[str, Any]:
        normalized = [item for item in (_normalize_detection(item) for item in detections) if item is not None]
        events: list[dict[str, Any]] = []
        matches = self._match(normalized)
        matched_detection_indexes = {detection_index for detection_index, _ in matches}
        matched_track_ids = {track_id for _, track_id in matches}

        for detection_index, track_id in matches:
            detection = normalized[detection_index]
            track = self._tracks[track_id]
            previous_bbox = dict(track["bbox"])
            previous_region = _region(_center(previous_bbox))
            distance = _distance(_center(previous_bbox), _center(detection["bbox"]))
            track.update(
                {
                    "bbox": dict(detection["bbox"]),
                    "confidence": detection["confidence"],
                    "extras": dict(detection.get("extras", {})),
                    "last_seen_frame": frame_id,
                    "last_observed_at": observed_at,
                    "missing_frames": 0,
                }
            )
            current_region = _region(_center(track["bbox"]))
            if distance >= self.move_threshold or previous_region != current_region:
                track["temporalState"] = "moving"
                events.append(
                    _fallback_event(
                        event_type="moved",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        from_region=previous_region,
                        to_region=current_region,
                        distance=distance,
                    )
                )
            else:
                track["temporalState"] = "stationary"

        for detection_index, detection in enumerate(normalized):
            if detection_index in matched_detection_indexes:
                continue
            track = self._new_track(detection, frame_id=frame_id, observed_at=observed_at)
            track["temporalState"] = "appeared"
            self._tracks[str(track["trackId"])] = track
            matched_track_ids.add(str(track["trackId"]))
            events.append(
                _fallback_event(
                    event_type="appeared",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=track,
                    from_region="",
                    to_region=_region(_center(track["bbox"])),
                    distance=0.0,
                )
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1
            if int(track["missing_frames"]) > self.max_missing_frames:
                events.append(
                    _fallback_event(
                        event_type="disappeared",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        from_region=_region(_center(track["bbox"])),
                        to_region="",
                        distance=0.0,
                    )
                )
                del self._tracks[track_id]

        active_tracks = [track for track in self._tracks.values() if int(track.get("missing_frames", 0)) == 0]
        attention = max(active_tracks, key=lambda item: (float(item["confidence"]), str(item["trackId"])), default=None)

        objects = [_track_object(track) for track in sorted(active_tracks, key=lambda item: str(item["trackId"]))]
        summary = _fallback_summary(objects, events)
        scene_id = _fallback_scene_id(frame_id, observed_at, objects)
        for event in events:
            event["sceneId"] = scene_id
            event["eventId"] = f"{scene_id}:{event['eventType']}:{event['subject']['trackId']}"
        return {
            "frameId": frame_id,
            "observedAt": observed_at,
            "events": events,
            "sceneSnapshot": {
                "sceneId": scene_id,
                "observedAt": observed_at,
                "frameId": frame_id,
                "objects": objects,
                "relationships": [],
                "attention": _track_object(attention) if attention else {},
                "summary": summary,
                "eventSummary": _event_summary(events),
                "metadata": {"frameId": frame_id, "realtime": True, "simulator": "eihead_local", "trackCount": len(objects)},
            },
            "sceneGraphSummary": summary,
        }

    def _match(self, detections: list[dict[str, Any]]) -> list[tuple[int, str]]:
        candidates: list[tuple[float, int, str]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if detection["label"] != track["label"]:
                    continue
                distance = _distance(_center(detection["bbox"]), _center(track["bbox"]))
                if distance <= self.match_distance:
                    candidates.append((distance, detection_index, track_id))
        matches: list[tuple[int, str]] = []
        used_detections: set[int] = set()
        used_tracks: set[str] = set()
        for _, detection_index, track_id in sorted(candidates):
            if detection_index in used_detections or track_id in used_tracks:
                continue
            matches.append((detection_index, track_id))
            used_detections.add(detection_index)
            used_tracks.add(track_id)
        return matches

    def _new_track(self, detection: dict[str, Any], *, frame_id: str, observed_at: str) -> dict[str, Any]:
        label = str(detection["label"])
        next_id = self._next_ids.get(label, 0) + 1
        self._next_ids[label] = next_id
        return {
            "trackId": f"{label}-{next_id:03d}",
            "label": label,
            "bbox": dict(detection["bbox"]),
            "confidence": float(detection["confidence"]),
            "first_seen_frame": frame_id,
            "last_seen_frame": frame_id,
            "last_observed_at": observed_at,
            "missing_frames": 0,
            "extras": dict(detection.get("extras", {})),
        }


def _normalize_detection(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    label = str(raw.get("label") or raw.get("name") or raw.get("class") or "").strip()
    bbox = _normalize_bbox(raw.get("bbox"), format_hint=_bbox_format(raw))
    if not label or bbox is None:
        return None
    return {
        "label": label,
        "bbox": bbox,
        "confidence": _coerce_float(raw.get("confidence", raw.get("score", 0.0))),
        "extras": _detection_extras(raw),
    }


def _normalize_bbox(raw: Any, *, format_hint: str = "") -> dict[str, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x_min, y_min, x_max, y_max = _sequence_bbox_values(raw, format_hint=format_hint)
        except (TypeError, ValueError):
            return None
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        return {
            "x_min": round(_clip01(x_min), 4),
            "y_min": round(_clip01(y_min), 4),
            "x_max": round(_clip01(x_max), 4),
            "y_max": round(_clip01(y_max), 4),
        }
    if not isinstance(raw, Mapping):
        return None
    try:
        if "x" in raw and "y" in raw and ("w" in raw or "width" in raw) and ("h" in raw or "height" in raw):
            x_min = _clip01(float(raw.get("x", 0.0)))
            y_min = _clip01(float(raw.get("y", 0.0)))
            x_max = _clip01(x_min + float(raw.get("w", raw.get("width", 0.0))))
            y_max = _clip01(y_min + float(raw.get("h", raw.get("height", 0.0))))
            return {
                "x_min": round(x_min, 4),
                "y_min": round(y_min, 4),
                "x_max": round(x_max, 4),
                "y_max": round(y_max, 4),
            }
        if "x1" in raw and "y1" in raw and "x2" in raw and "y2" in raw:
            x_min = _clip01(float(raw.get("x1", 0.0)))
            y_min = _clip01(float(raw.get("y1", 0.0)))
            x_max = _clip01(float(raw.get("x2", 0.0)))
            y_max = _clip01(float(raw.get("y2", 0.0)))
            if x_max < x_min:
                x_min, x_max = x_max, x_min
            if y_max < y_min:
                y_min, y_max = y_max, y_min
            return {
                "x_min": round(x_min, 4),
                "y_min": round(y_min, 4),
                "x_max": round(x_max, 4),
                "y_max": round(y_max, 4),
            }
        x_min = _clip01(float(raw.get("x_min", raw.get("xmin", raw.get("left", 0.0)))))
        y_min = _clip01(float(raw.get("y_min", raw.get("ymin", raw.get("top", 0.0)))))
        x_max = _clip01(float(raw.get("x_max", raw.get("xmax", raw.get("right", 0.0)))))
        y_max = _clip01(float(raw.get("y_max", raw.get("ymax", raw.get("bottom", 0.0)))))
    except (TypeError, ValueError):
        return None
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _sequence_bbox_values(raw: list[Any] | tuple[Any, ...], *, format_hint: str = "") -> tuple[float, float, float, float]:
    x_min = float(raw[0])
    y_min = float(raw[1])
    third = float(raw[2])
    fourth = float(raw[3])
    if format_hint == "xyxy":
        return (x_min, y_min, third, fourth)
    if format_hint == "xywh":
        return (x_min, y_min, x_min + third, y_min + fourth)
    if max(abs(x_min), abs(y_min), abs(third), abs(fourth)) <= 1.0:
        return (x_min, y_min, x_min + third, y_min + fourth)
    if third <= x_min or fourth <= y_min:
        return (x_min, y_min, x_min + third, y_min + fourth)
    return (x_min, y_min, third, fourth)


def _bbox_format(raw: Mapping[str, Any]) -> str:
    for key in ("bboxFormat", "bbox_format", "boxFormat", "box_format", "format"):
        value = raw.get(key)
        if value is not None:
            normalized = str(value).strip().lower().replace("-", "").replace("_", "")
            if normalized in {"xyxy", "x1y1x2y2"}:
                return "xyxy"
            if normalized in {"xywh", "ltwh"}:
                return "xywh"
    return ""


def _fallback_event(
    *,
    event_type: str,
    observed_at: str,
    frame_id: str,
    track: Mapping[str, Any],
    from_region: str,
    to_region: str,
    distance: float,
) -> dict[str, Any]:
    payload = {
        "eventId": "",
        "eventType": event_type,
        "observedAt": observed_at,
        "sceneId": "",
        "subject": {"trackId": track["trackId"], "label": track["label"]},
        "confidence": round(float(track["confidence"]), 3),
        "details": {"fromRegion": from_region, "toRegion": to_region, "distance": round(float(distance), 3)},
        "metadata": {"frameId": frame_id},
    }
    extras = track.get("extras")
    if isinstance(extras, Mapping):
        for key in ("pose", "clipLabels", "semanticLabels", "depth", "distance", "trackingDiagnostics"):
            value = extras.get(key)
            if value:
                payload[key] = value
    return payload


def _track_object(track: Mapping[str, Any]) -> dict[str, Any]:
    center = _center(track["bbox"])
    payload = {
        "trackId": track["trackId"],
        "label": track["label"],
        "confidence": round(float(track["confidence"]), 3),
        "bbox": dict(track["bbox"]),
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "region": _region(center),
        "missingFrames": int(track.get("missing_frames", 0)),
        "temporalState": str(track.get("temporalState") or "stationary"),
    }
    extras = track.get("extras")
    if isinstance(extras, Mapping):
        payload.update({key: value for key, value in extras.items() if value})
    return payload


def _detection_extras(raw: Mapping[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    source_track_id = _track_id(raw)
    if source_track_id:
        extras["sourceTrackId"] = source_track_id
    for source_key, target_key in (
        ("pose", "pose"),
        ("clipLabels", "clipLabels"),
        ("clip_labels", "clipLabels"),
        ("semanticLabels", "semanticLabels"),
        ("semantic_labels", "semanticLabels"),
        ("depth", "depth"),
        ("distance", "distance"),
        ("trackingDiagnostics", "trackingDiagnostics"),
        ("tracking_diagnostics", "trackingDiagnostics"),
    ):
        value = _raw_value(raw, source_key)
        if value:
            extras[target_key] = value
    return extras


def _raw_value(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    attributes = raw.get("attributes")
    if isinstance(attributes, Mapping):
        for key in keys:
            if key in attributes and attributes[key] not in (None, ""):
                return attributes[key]
    return None


def _fallback_summary(objects: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    observed = ", ".join(sorted({str(item.get("label")) for item in objects})) if objects else "empty scene"
    event_types = sorted({str(event.get("eventType")) for event in events if event.get("eventType")})
    return f"Observed {observed}; realtime events: {', '.join(event_types) if event_types else 'none'}"


def _event_summary(events: list[Mapping[str, Any]]) -> str:
    event_types = sorted({str(event.get("eventType")) for event in events if event.get("eventType")})
    return ", ".join(event_types)


def _fallback_scene_id(frame_id: str, observed_at: str, objects: list[dict[str, Any]]) -> str:
    payload = {"frameId": frame_id, "observedAt": observed_at, "objects": objects}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"scene_rt_{digest}"


def _center(bbox: Mapping[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0)


def _region(center: tuple[float, float]) -> str:
    x_name = "left" if center[0] < 0.33 else "center" if center[0] < 0.66 else "right"
    y_name = "top" if center[1] < 0.25 else "middle" if center[1] < 0.75 else "bottom"
    return f"{x_name}_{y_name}"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["RealtimeVisionSceneBridge"]
