"""Compact realtime vision state for voice dialogue grounding."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import time
from typing import Any, Iterable, Mapping


SCHEMA = "eibrain.vision_voice_context.v1"


def build_vision_voice_context(
    visual_state: Mapping[str, Any] | None = None,
    *,
    scene: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    now_ts: float | None = None,
    stale_after_s: float = 3.0,
    max_events: int = 3,
) -> dict[str, Any]:
    """Build a JSON-ready visual context bridge for the voice chain.

    The output only reflects supplied visual evidence. Stale visual state keeps
    descriptive facts but disables deictic grounding and lowers reliability.
    """

    state = dict(visual_state or {})
    scene_payload = dict(scene or {})
    target_payload = dict(target or {})
    now = _coerce_float(now_ts, default=time.time())
    stale_after = max(0.0, _coerce_float(stale_after_s, default=3.0))

    objects = _compact_objects(scene_payload or state)
    if not objects and scene_payload is not state:
        objects = _compact_objects(state)
    person_objects = [item for item in objects if _is_person_label(str(item["label"]))]
    face_objects = [item for item in objects if _is_face_label(str(item["label"]))]

    observed_at_ts = _first_timestamp(state, scene_payload, target_payload)
    age_s = round(max(0.0, now - observed_at_ts), 1) if observed_at_ts is not None else None
    stale = bool(age_s is not None and age_s > stale_after)
    freshness = {
        "observed_at_ts": observed_at_ts,
        "age_s": age_s,
        "stale_after_s": stale_after,
        "stale": stale,
    }

    target_context = _target_context(target_payload, objects=objects, stale=stale)
    event_context = _compact_events(events, state=state, scene=scene_payload, now_ts=now, max_events=max_events)
    tracking = {
        "target_locked": target_context["status"] == "locked",
        "target_lost": target_context["status"] == "lost",
        "following": bool(target_context.get("following")),
        "status": _tracking_status(target_context),
    }
    reference = _reference_context(
        has_person=bool(person_objects),
        target_status=str(target_context["status"]),
        stale=stale,
    )
    reliability = _reliability(
        target_context=target_context,
        objects=objects,
        stale=stale,
        has_grounding=bool(reference["can_resolve_deictic"]),
    )
    text = _dialogue_text(
        stale=stale,
        age_s=age_s,
        person_objects=person_objects,
        face_objects=face_objects,
        target_context=target_context,
        events=event_context,
    )

    return {
        "schema": SCHEMA,
        "source": "vision_voice_context",
        "has_person": bool(person_objects),
        "has_face": bool(face_objects),
        "person_count": len(person_objects),
        "face_count": len(face_objects),
        "object_count": len(objects),
        "objects": objects,
        "freshness": freshness,
        "target": target_context,
        "tracking": tracking,
        "events": event_context,
        "reference": reference,
        "reliability": reliability,
        "dialogue_context_text": text,
        "summary_text": text,
    }


def _compact_objects(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_objects = source.get("objects") or source.get("detections") or []
    if not isinstance(raw_objects, list):
        return []
    objects: list[dict[str, Any]] = []
    for raw in raw_objects:
        if not isinstance(raw, Mapping):
            continue
        label = _clean_text(raw.get("label") or raw.get("name") or raw.get("class"))
        if not label:
            continue
        bbox = _bbox(raw.get("bbox"))
        region = _clean_text(raw.get("region") or raw.get("position") or raw.get("location"))
        if not region and bbox is not None:
            region = _region(_center(bbox))
        item = {
            "label": label,
            "track_id": _clean_text(
                raw.get("track_id") or raw.get("trackId") or raw.get("stable_id") or raw.get("stableId") or raw.get("id")
            ),
            "region": region,
            "confidence": round(_confidence(raw), 3),
        }
        temporal_state = _clean_text(raw.get("temporal_state") or raw.get("temporalState"))
        if temporal_state:
            item["temporal_state"] = temporal_state
        objects.append(item)
    return objects


def _target_context(target: Mapping[str, Any], *, objects: list[dict[str, Any]], stale: bool) -> dict[str, Any]:
    if not target:
        return {
            "status": "none",
            "label": "",
            "track_id": "",
            "region": "",
            "distance_m": None,
            "bearing": "",
            "confidence": 0.0,
            "last_seen_age_s": None,
            "following": False,
            "stale": stale,
        }

    track_id = _clean_text(target.get("track_id") or target.get("trackId") or target.get("stable_id") or target.get("id"))
    label = _clean_text(target.get("label") or target.get("name") or target.get("class"))
    matched = _matching_object(track_id=track_id, label=label, objects=objects)
    if not label and matched:
        label = str(matched["label"])
    if not track_id and matched:
        track_id = str(matched["track_id"])
    region = _clean_text(
        target.get("region")
        or target.get("target_region")
        or target.get("position")
        or target.get("location")
        or target.get("last_region")
        or target.get("lastRegion")
    )
    if not region and matched:
        region = str(matched["region"])
    if not region:
        bbox = _bbox(target.get("bbox"))
        if bbox is not None:
            region = _region(_center(bbox))

    lost = _truthy(target.get("lost") or target.get("target_lost")) or _clean_text(target.get("status")).lower() == "lost"
    locked = _truthy(target.get("locked") or target.get("target_locked")) or _clean_text(target.get("status")).lower() == "locked"
    if stale and (locked or target):
        status = "stale"
    elif lost:
        status = "lost"
    elif locked or track_id or label:
        status = "locked" if locked else "observed"
    else:
        status = "none"

    confidence = _coerce_optional_float(_first_present(target, "confidence", "score"))
    if confidence is None and matched:
        confidence = _coerce_float(matched.get("confidence"), default=0.0)
    return {
        "status": status,
        "label": label,
        "track_id": track_id,
        "region": region,
        "distance_m": _coerce_optional_float(_first_present(target, "distance_m", "distanceMeters", "distance")),
        "bearing": _clean_text(target.get("bearing") or target.get("direction") or target.get("azimuth")),
        "confidence": round(confidence or 0.0, 3),
        "last_seen_age_s": _coerce_optional_float(_first_present(target, "last_seen_age_s", "lastSeenAgeS")),
        "following": bool(_truthy(target.get("following") or target.get("is_following")) and status == "locked"),
        "stale": stale,
    }


def _compact_events(
    events: Iterable[Mapping[str, Any]] | None,
    *,
    state: Mapping[str, Any],
    scene: Mapping[str, Any],
    now_ts: float,
    max_events: int,
) -> list[dict[str, Any]]:
    raw_events = list(events) if events is not None else _event_list(state.get("events") or scene.get("events"))
    normalized: list[tuple[float, int, dict[str, Any]]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            continue
        event_type = _clean_text(raw.get("type") or raw.get("eventType") or raw.get("event_type"))
        if not event_type:
            continue
        if event_type.strip().lower() == "stationary":
            continue
        subject = raw.get("subject") if isinstance(raw.get("subject"), Mapping) else {}
        details = raw.get("details") if isinstance(raw.get("details"), Mapping) else {}
        label = _clean_text(raw.get("label") or subject.get("label"))
        track_id = _clean_text(raw.get("track_id") or raw.get("trackId") or subject.get("track_id") or subject.get("trackId"))
        region = _clean_text(
            raw.get("region")
            or raw.get("to_region")
            or raw.get("toRegion")
            or details.get("toRegion")
            or details.get("to_region")
        )
        observed_ts = _timestamp(raw)
        age_s = round(max(0.0, now_ts - observed_ts), 1) if observed_ts is not None else None
        summary = " ".join(part for part in (event_type, label, region) if part)
        normalized.append(
            (
                observed_ts if observed_ts is not None else float("-inf"),
                index,
                {
                    "type": event_type,
                    "label": label,
                    "track_id": track_id,
                    "region": region,
                    "age_s": age_s,
                    "summary": summary,
                },
            )
        )
    normalized.sort(key=lambda item: (item[0], item[1]), reverse=True)
    limit = max(0, int(max_events))
    return [item for _, _, item in normalized[:limit]]


def _reference_context(*, has_person: bool, target_status: str, stale: bool) -> dict[str, Any]:
    if stale:
        reason = "visual_context_stale"
        can_resolve = False
    elif target_status == "lost":
        reason = "target_lost"
        can_resolve = False
    elif target_status in {"locked", "observed"} or has_person:
        reason = "fresh_visual_grounding"
        can_resolve = True
    else:
        reason = "no_live_person_or_target"
        can_resolve = False
    return {
        "can_resolve_deictic": can_resolve,
        "can_use_this": can_resolve,
        "can_use_there": can_resolve,
        "can_use_look": can_resolve,
        "reason": reason,
    }


def _reliability(
    *,
    target_context: Mapping[str, Any],
    objects: list[dict[str, Any]],
    stale: bool,
    has_grounding: bool,
) -> float:
    target_confidence = _coerce_float(target_context.get("confidence"), default=0.0)
    object_confidence = max((_coerce_float(item.get("confidence"), default=0.0) for item in objects), default=0.0)
    if target_context.get("status") in {"locked", "observed", "lost", "stale"}:
        base = target_confidence if target_confidence > 0.0 else object_confidence
    else:
        base = object_confidence
    if not has_grounding:
        base = min(base, 0.49)
    if stale:
        base *= 0.3
    return round(max(0.0, min(1.0, base)), 3)


def _dialogue_text(
    *,
    stale: bool,
    age_s: float | None,
    person_objects: list[dict[str, Any]],
    face_objects: list[dict[str, Any]],
    target_context: Mapping[str, Any],
    events: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    if stale:
        age = f"{age_s:.1f}s" if age_s is not None else "unknown age"
        parts.append(f"vision stale {age}")
    elif age_s is not None:
        parts.append(f"vision fresh {age_s:.1f}s")
    else:
        parts.append("vision freshness unknown")

    if person_objects:
        person = person_objects[0]
        region = str(person.get("region") or "unknown")
        parts.append(f"person at {region}")
    else:
        parts.append("no person visible")
    if face_objects:
        parts.append("face visible")

    target_status = str(target_context.get("status") or "none")
    target_region = str(target_context.get("region") or "")
    if target_status in {"locked", "observed", "lost", "stale"}:
        target_part = f"target {target_status}"
        if target_region:
            target_part = f"{target_part} {target_region}"
        distance = target_context.get("distance_m")
        if distance is not None:
            target_part = f"{target_part} {distance}m"
        bearing = str(target_context.get("bearing") or "")
        if bearing:
            target_part = f"{target_part} {bearing}"
        parts.append(target_part)
    if target_context.get("following"):
        parts.append("following target")
    if events:
        parts.append("events: " + "; ".join(str(item["summary"]) for item in events if item.get("summary")))
    return "; ".join(parts)


def _tracking_status(target_context: Mapping[str, Any]) -> str:
    if target_context.get("following"):
        return "following"
    status = str(target_context.get("status") or "none")
    if status == "locked":
        return "locked"
    if status == "lost":
        return "lost"
    if status == "stale":
        return "stale"
    return "idle"


def _matching_object(*, track_id: str, label: str, objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    if track_id:
        for item in objects:
            if item.get("track_id") == track_id:
                return item
    if label:
        for item in objects:
            if item.get("label") == label:
                return item
    return None


def _event_list(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _first_present(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _first_timestamp(*sources: Mapping[str, Any]) -> float | None:
    for source in sources:
        timestamp = _timestamp(source)
        if timestamp is not None:
            return timestamp
    return None


def _timestamp(source: Mapping[str, Any]) -> float | None:
    for key in (
        "updated_at_ts",
        "state_updated_at_ts",
        "observed_at_ts",
        "observedAtTs",
        "frame_captured_at_ts",
        "frame_updated_at_ts",
        "timestamp",
        "ts",
    ):
        value = _coerce_optional_float(source.get(key))
        if value is not None:
            return value
    for key in ("observed_at", "observedAt", "updated_at", "updatedAt"):
        parsed = _parse_datetime_ts(source.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime_ts(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _confidence(raw: Mapping[str, Any]) -> float:
    return max(0.0, min(1.0, _coerce_float(raw.get("confidence", raw.get("score", 0.0)), default=0.0)))


def _bbox(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, Mapping):
        return None
    try:
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


def _center(bbox: Mapping[str, float]) -> tuple[float, float]:
    return (
        (_coerce_float(bbox.get("x_min"), default=0.0) + _coerce_float(bbox.get("x_max"), default=0.0)) / 2.0,
        (_coerce_float(bbox.get("y_min"), default=0.0) + _coerce_float(bbox.get("y_max"), default=0.0)) / 2.0,
    )


def _region(center: tuple[float, float]) -> str:
    x_name = "left" if center[0] < 0.33 else "center" if center[0] < 0.66 else "right"
    y_name = "top" if center[1] < 0.25 else "middle" if center[1] < 0.75 else "bottom"
    return f"{x_name}_{y_name}"


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _coerce_optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "locked", "lost", "following"}
    return bool(value)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_person_label(label: str) -> bool:
    return label.strip().lower() in {"person", "human", "people"}


def _is_face_label(label: str) -> bool:
    return label.strip().lower() in {"face", "person_face", "human_face"}


__all__ = ["SCHEMA", "build_vision_voice_context"]
