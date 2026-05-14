"""Shared vision-state contract for Hailo-backed eye runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


DEFAULT_VISION_STATE_DIR = Path(tempfile.gettempdir()) / "eibrain-vision"
DEFAULT_VISION_STATE_PATH = DEFAULT_VISION_STATE_DIR / "state.json"
DEFAULT_VISION_FRAME_PATH = DEFAULT_VISION_STATE_DIR / "latest.jpg"
DETECTION_EXTENSION_FIELDS = (
    "category",
    "source",
    "model_id",
    "attributes",
    "spatial",
    "stable_id",
    "region",
    "canonical_label",
)


@dataclass(frozen=True, slots=True)
class VisionStateSnapshot:
    payload: dict[str, Any]
    age_s: float | None
    stale: bool


class VisionStateReader:
    """Reads the atomic JSON state produced by the vision service."""

    def __init__(self, state_path: str | Path, *, stale_after_s: float = 3.0) -> None:
        self.state_path = Path(state_path)
        self.stale_after_s = stale_after_s

    def read(self, *, now_ts: float | None = None) -> VisionStateSnapshot:
        now = time.time() if now_ts is None else now_ts
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("vision state payload must be a JSON object")
        updated_at_ts = _coerce_float(payload.get("updated_at_ts"), default=None)
        frame_ts = _coerce_float(payload.get("frame_captured_at_ts"), default=None)
        timestamp = updated_at_ts if updated_at_ts is not None else frame_ts
        age_s = None if timestamp is None else max(0.0, now - timestamp)
        stale = age_s is None or age_s > self.stale_after_s
        return VisionStateSnapshot(payload=payload, age_s=age_s, stale=stale)


class VisionStateWriter:
    """Writes vision state atomically so monitors never read half JSON."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)

    def write(self, payload: dict[str, Any]) -> Path:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = dict(payload)
        normalized.setdefault("updated_at_ts", time.time())
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
            dir=str(self.state_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(normalized, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            Path(tmp_name).replace(self.state_path)
        except Exception:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            finally:
                raise
        return self.state_path


def normalize_detection(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    bbox = raw.get("bbox", {})
    if not isinstance(bbox, dict):
        return None
    try:
        normalized_bbox = {
            "x_min": _clip01(float(bbox.get("x_min", bbox.get("xmin", 0.0)))),
            "y_min": _clip01(float(bbox.get("y_min", bbox.get("ymin", 0.0)))),
            "x_max": _clip01(float(bbox.get("x_max", bbox.get("xmax", 0.0)))),
            "y_max": _clip01(float(bbox.get("y_max", bbox.get("ymax", 0.0)))),
        }
        score = float(raw.get("score", raw.get("confidence", 0.0)))
    except (TypeError, ValueError):
        return None
    item: dict[str, Any] = {
        "label": str(raw.get("label", "unknown") or "unknown"),
        "score": score,
        "bbox": normalized_bbox,
    }
    if raw.get("class_id") is not None:
        try:
            item["class_id"] = int(raw["class_id"])
        except (TypeError, ValueError):
            safe_class_id = _json_safe(raw["class_id"])
            if safe_class_id is not _UNSAFE:
                item["class_id"] = safe_class_id
    if raw.get("track_id") is not None:
        safe_track_id = _json_safe(raw["track_id"])
        if safe_track_id is not _UNSAFE:
            item["track_id"] = safe_track_id
    for field in DETECTION_EXTENSION_FIELDS:
        if field not in raw:
            continue
        safe_value = _json_safe(raw[field])
        if safe_value is not _UNSAFE:
            item[field] = safe_value
    return item


def build_vision_state(
    *,
    detections: list[dict[str, Any]],
    frame_path: str | Path | None,
    status: str = "ok",
    frame_captured_at_ts: float | None = None,
    backend: str = "hailo8",
    details: dict[str, Any] | None = None,
    schema_version: int = 2,
    pipeline: dict[str, Any] | None = None,
    objects: list[dict[str, Any]] | None = None,
    scene: dict[str, Any] | None = None,
    spatial: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    frame_id: str | None = None,
    camera_id: str | None = None,
    scene_id: str | None = None,
    place_hint: str | None = None,
) -> dict[str, Any]:
    normalized = [item for item in (normalize_detection(detection) for detection in detections) if item is not None]
    normalized_objects = (
        normalized
        if objects is None
        else [item for item in (normalize_detection(obj) for obj in objects) if item is not None]
    )
    captured_at = time.time() if frame_captured_at_ts is None else frame_captured_at_ts
    scene_labels = sorted({str(item.get("label", "unknown")) for item in normalized})
    scene_summary = summarize_detections(normalized)
    normalized_scene = _normalize_scene(
        scene,
        summary=scene_summary,
        labels=scene_labels,
        place_hint=place_hint,
    )
    return {
        "schema_version": schema_version,
        "status": status,
        "backend": backend,
        "updated_at_ts": time.time(),
        "frame_path": str(frame_path) if frame_path is not None else None,
        "frame_captured_at_ts": captured_at,
        "frame_id": frame_id,
        "camera_id": camera_id,
        "scene_id": scene_id,
        "detections": normalized,
        "objects": normalized_objects,
        "detection_count": len(normalized),
        "top_detection": normalized[0] if normalized else None,
        "scene_labels": scene_labels,
        "scene_summary": scene_summary,
        "scene": normalized_scene,
        "spatial": _normalize_spatial(spatial),
        "events": _normalize_events(events),
        "pipeline": _normalize_mapping(pipeline),
        "details": _normalize_mapping(details),
    }


def summarize_detections(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return "no detections in current frame"
    counts: dict[str, int] = {}
    for detection in detections:
        label = str(detection.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))


def _coerce_float(value: Any, *, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class _UnsafeJsonValue:
    pass


_UNSAFE = _UnsafeJsonValue()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSAFE
    if isinstance(value, list):
        items = []
        for item in value:
            safe_item = _json_safe(item)
            if safe_item is not _UNSAFE:
                items.append(safe_item)
        return items
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            safe_item = _json_safe(item)
            if safe_item is not _UNSAFE:
                result[key] = safe_item
        return result
    return _UNSAFE


def _normalize_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    safe_value = _json_safe(value or {})
    return safe_value if isinstance(safe_value, dict) else {}


def _normalize_scene(
    scene: dict[str, Any] | None,
    *,
    summary: str,
    labels: list[str],
    place_hint: str | None,
) -> dict[str, Any]:
    normalized = _normalize_mapping(scene)
    normalized["summary"] = str(normalized.get("summary", summary))
    raw_labels = normalized.get("labels", labels)
    normalized["labels"] = [str(label) for label in raw_labels] if isinstance(raw_labels, list) else labels
    normalized["place_hint"] = place_hint if place_hint is not None else normalized.get("place_hint")
    return normalized


def _normalize_spatial(spatial: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_mapping(spatial)
    relations = normalized.get("relations")
    normalized["relations"] = relations if isinstance(relations, list) else []
    return normalized


def _normalize_events(events: list[dict[str, Any]] | None) -> list[Any]:
    safe_events = _json_safe(events or [])
    return safe_events if isinstance(safe_events, list) else []
