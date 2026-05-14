"""Reusable Hailo ROI metadata parsing helpers.

This module intentionally does not import the Hailo SDK at import time.
Callers pass an already-loaded ``hailo_module`` so tests and non-Hailo hosts
can exercise the parser without the native dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class HailoMetadataParseError(ValueError):
    """Raised when Hailo metadata cannot be parsed in strict mode."""


def parse_hailo_detections(
    buffer: Any,
    hailo_module: Any,
    model_id: str,
    score_threshold: float,
    *,
    labels: Sequence[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Parse Hailo ROI metadata into normalized detection dicts."""

    detections: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        roi = hailo_module.get_roi_from_buffer(buffer)
        detection_type = getattr(hailo_module, "HAILO_DETECTION", "HAILO_DETECTION")
        raw_detections = roi.get_objects_typed(detection_type)
    except Exception as exc:
        if strict:
            raise HailoMetadataParseError(
                f"failed to read Hailo ROI metadata: {exc.__class__.__name__}: {exc}"
            ) from exc
        return {
            "detections": detections,
            "parse_error_count": 1,
            "errors": [
                {
                    "index": None,
                    "exception": exc.__class__.__name__,
                    "message": str(exc),
                }
            ],
        }

    for index, raw_detection in enumerate(raw_detections):
        try:
            parsed = _parse_detection(
                raw_detection,
                model_id=model_id,
                score_threshold=score_threshold,
                labels=labels,
            )
        except Exception as exc:
            if strict:
                raise HailoMetadataParseError(
                    f"failed to parse detection at index {index}: {exc.__class__.__name__}: {exc}"
                ) from exc
            errors.append(
                {
                    "index": index,
                    "exception": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            continue
        if parsed is not None:
            detections.append(parsed)

    return {
        "detections": detections,
        "parse_error_count": len(errors),
        "errors": errors,
    }


def _parse_detection(
    raw_detection: Any,
    *,
    model_id: str,
    score_threshold: float,
    labels: Sequence[str] | None,
) -> dict[str, Any] | None:
    confidence = float(raw_detection.get_confidence())
    if confidence < float(score_threshold):
        return None

    class_id = int(raw_detection.get_class_id())
    bbox = raw_detection.get_bbox()
    label = _resolve_label(raw_detection, class_id=class_id, labels=labels)

    payload: dict[str, Any] = {
        "label": label,
        "score": confidence,
        "confidence": confidence,
        "bbox": {
            "x_min": float(bbox.xmin()),
            "y_min": float(bbox.ymin()),
            "x_max": float(bbox.xmax()),
            "y_max": float(bbox.ymax()),
        },
        "class_id": class_id,
        "source": "hailo",
        "model_id": str(model_id),
    }

    track_id = _read_track_id(raw_detection)
    if track_id is not None:
        payload["track_id"] = track_id
    return payload


def _resolve_label(raw_detection: Any, *, class_id: int, labels: Sequence[str] | None) -> str:
    label_value = ""
    if hasattr(raw_detection, "get_label"):
        label_value = str(raw_detection.get_label() or "").strip()
    if label_value:
        return label_value
    if labels is not None and 0 <= class_id < len(labels):
        fallback = labels[class_id]
        if fallback:
            return str(fallback)
    return f"class_{class_id}"


def _read_track_id(raw_detection: Any) -> object | None:
    if hasattr(raw_detection, "get_track_id"):
        try:
            track_id = raw_detection.get_track_id()
        except Exception:
            track_id = None
        if track_id is not None:
            return track_id

    try:
        unique_ids = raw_detection.get_objects_typed("HAILO_UNIQUE_ID")
    except Exception:
        return None

    for unique_id in unique_ids:
        try:
            return unique_id.get_id()
        except Exception:
            continue
    return None


__all__ = [
    "HailoMetadataParseError",
    "parse_hailo_detections",
]
