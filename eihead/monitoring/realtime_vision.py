"""Standard realtime eye/vision monitor payload helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from eihead.protocol import (
    EYE_REALTIME_CHANNEL,
    VISION_REALTIME_ALIAS,
    VISION_REALTIME_MODE,
    VISION_STATIC_COMPAT_MODE,
)
from eihead.protocol.base import serialize_message


REALTIME_VISION_SCHEMA = "eihead.monitor.vision_realtime.v1"
REALTIME_VISION_ATTRS = (
    "eye_realtime",
    "vision_realtime",
    "realtime_vision",
    "latest_eye_realtime",
    "latest_vision_realtime",
    "latest_realtime_vision",
)


def realtime_vision_payload_from_app(app: Any, *, timestamp: float) -> dict[str, Any]:
    """Read the first realtime eye/vision hook from an app and standardize it."""

    first_non_live_payload: dict[str, Any] | None = None
    for attr_name in REALTIME_VISION_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        raw_payload = _resolve_realtime_observation_candidate(source() if callable(source) else source)
        payload = build_realtime_vision_payload(
            raw_payload,
            timestamp=timestamp,
            source=attr_name,
            wired=raw_payload is not None,
        )
        if payload.get("wired") is True:
            return payload
        if first_non_live_payload is None:
            first_non_live_payload = payload

    if first_non_live_payload is not None:
        return first_non_live_payload

    return build_realtime_vision_payload(
        None,
        timestamp=timestamp,
        source=None,
        wired=False,
    )


def build_realtime_vision_payload(
    observation: Any = None,
    *,
    timestamp: float,
    source: str | None = None,
    wired: bool | None = None,
) -> dict[str, Any]:
    """Return the monitor JSON envelope for primary realtime eye state."""

    resolved_observation = _resolve_realtime_observation_candidate(observation)
    serialized_observation = _serialize_observation(resolved_observation) if resolved_observation is not None else None
    derived_status, derived_wired, derived_message = _derive_realtime_status(serialized_observation)
    is_wired = derived_wired if wired is None else bool(wired and derived_wired)
    diagnostic = _build_realtime_diagnostic(
        serialized_observation,
        status=derived_status if serialized_observation is not None else "not_wired",
        wired=is_wired,
        timestamp=timestamp,
        source=source,
    )
    payload: dict[str, Any] = {
        "schema": REALTIME_VISION_SCHEMA,
        "runtime": "eihead",
        "status": derived_status if serialized_observation is not None else "not_wired",
        "wired": is_wired,
        "source": source,
        "channel": EYE_REALTIME_CHANNEL,
        "aliases": [VISION_REALTIME_ALIAS],
        "primary_mode": VISION_REALTIME_MODE,
        "compat_static": {"mode": VISION_STATIC_COMPAT_MODE, "primary": False},
        "captured_at_ts": timestamp,
        "observation": serialized_observation,
        "diagnostic": diagnostic,
        "backend": diagnostic["backend"],
        "frame_id": diagnostic["frame_id"],
        "fps": diagnostic["fps"],
        "last_frame_age": diagnostic["last_frame_age"],
        "last_frame_age_s": diagnostic["last_frame_age_s"],
        "detections": diagnostic["detections"],
        "boxes": diagnostic["boxes"],
        "scores": diagnostic["scores"],
        "score_threshold": diagnostic["score_threshold"],
        "top_k": diagnostic["top_k"],
        "top_detection": diagnostic["top_detection"],
        "not_wired": diagnostic["not_wired"],
        "placeholder": diagnostic["placeholder"],
        "stream_ready": diagnostic["stream_ready"],
        "stale": diagnostic["stale"],
        "degraded": diagnostic["degraded"],
        "status_reason": diagnostic["status_reason"],
        "not_wired_reason": diagnostic["not_wired_reason"],
        "stale_reason": diagnostic["stale_reason"],
        "degraded_reason": diagnostic["degraded_reason"],
        "readiness": diagnostic["readiness"],
        "compat_static_active": diagnostic["compat_static"],
        "compatibility_static_image": diagnostic["compatibility_static_image"],
        "frame_interval_ms": diagnostic["frame_interval_ms"],
        "jitter_guard": diagnostic["jitter_guard"],
        "hooks_used": diagnostic["hooks_used"],
        "pipeline": diagnostic["pipeline"],
        "devices": diagnostic["devices"],
        "readiness_message": diagnostic["readiness_message"],
        "parse_error_count": diagnostic["parse_error_count"],
        "parse_errors": diagnostic["parse_errors"],
        "overlay": diagnostic["overlay"],
        "visual_diagnostic": diagnostic["overlay"],
        "source_freshness": diagnostic["source_freshness"],
        "latency_ms": diagnostic["latency_ms"],
        "scene": diagnostic["scene"],
        "scene_id": diagnostic["scene_id"],
        "scene_summary": diagnostic["scene_summary"],
        "scene_graph_summary": diagnostic["scene_graph_summary"],
        "tracks": diagnostic["tracks"],
        "track_count": diagnostic["track_count"],
        "track_summary": diagnostic["track_summary"],
        "tracking_stability": diagnostic["tracking_stability"],
        "tracking_stability_score": diagnostic["tracking_stability_score"],
        "tracking_switch_count": diagnostic["tracking_switch_count"],
        "tracking_lost_count": diagnostic["tracking_lost_count"],
        "tracking_reacquired_count": diagnostic["tracking_reacquired_count"],
        "multimodal_availability": diagnostic["multimodal_availability"],
        "pose_availability": diagnostic["pose_availability"],
        "clip_availability": diagnostic["clip_availability"],
        "depth_availability": diagnostic["depth_availability"],
        "events": diagnostic["events"],
        "event_count": diagnostic["event_count"],
        "event_summary": diagnostic["event_summary"],
        "score_labels": diagnostic["score_labels"],
        "target": diagnostic["target"],
        "target_center": diagnostic["target_center"],
        "target_error": diagnostic["target_error"],
        "target_score_label": diagnostic["target_score_label"],
        "detections_summary": diagnostic["detections_summary"],
        "health_state": diagnostic["health_state"],
    }
    if not is_wired:
        if derived_message:
            payload["message"] = derived_message
        elif source:
            payload["message"] = f"runtime app.{source} did not return an eye.realtime payload"
        else:
            payload["message"] = "runtime app does not expose eye.realtime or vision.realtime payload"
    return payload


def _derive_realtime_status(observation: Mapping[str, Any] | None) -> tuple[str, bool, str]:
    if observation is None:
        return "not_wired", False, ""
    status = _coerce_status(observation.get("status"))
    kind = str(observation.get("kind", "") or "").strip().lower()
    mode = str(observation.get("mode", "") or "").strip().lower()
    primary_mode = observation.get("primary_mode")
    placeholder = _truthy(observation.get("placeholder", False))
    not_wired = _truthy(observation.get("not_wired", False))
    compatibility_mode = _truthy(observation.get("compatibility_mode", False))
    degraded = _truthy(observation.get("degraded", False)) or status == "degraded"
    stale = _is_stale_observation(observation, status=status)

    if kind == "realtime_vision_scene_bridge":
        live = _truthy(observation.get("live", True))
        reason = _coerce_status(observation.get("reason")) or status
        if not live and reason in {"compat_static", "static"}:
            return "compat_static", False, "compat/static vision payload is not accepted as realtime eye data"
        if not live and reason == "stale":
            return "stale", True, ""
        if not live:
            return "not_wired", False, "eye.realtime payload is present but not ready"

    if not_wired or placeholder or _status_is_not_wired(status):
        return "not_wired", False, "eye.realtime payload is present but not ready"
    if kind == "vision_observation" or mode == VISION_STATIC_COMPAT_MODE or primary_mode is False or compatibility_mode:
        return "compat_static", False, "compat/static vision payload is not accepted as realtime eye data"
    if degraded and (kind == "realtime_vision_observation" or mode in {VISION_REALTIME_MODE, "realtime_stream"}):
        return "degraded", True, ""
    if stale and (kind == "realtime_vision_observation" or mode in {VISION_REALTIME_MODE, "realtime_stream"}):
        return "stale", True, ""
    if kind == "realtime_vision_observation" or mode in {VISION_REALTIME_MODE, "realtime_stream"}:
        return "wired", True, ""
    return "unknown", False, "payload is not a recognized realtime eye observation"


def _build_realtime_diagnostic(
    observation: Mapping[str, Any] | None,
    *,
    status: str,
    wired: bool,
    timestamp: float,
    source: str | None,
) -> dict[str, Any]:
    detections = _normalized_detections(_first_present(observation, "detections") if observation else None)
    score_threshold = _detection_score_threshold(observation)
    top_k = _detection_top_k(observation)
    filtered_detections, scores = _filter_detections(
        detections,
        score_threshold=score_threshold,
        top_k=top_k,
    )
    boxes = [
        box
        for box in (_normalize_box(item.get("bbox")) for item in filtered_detections)
        if box is not None
    ]
    top_detection = _normalized_detection(_first_present(observation, "top_detection") if observation else None)
    if top_detection is None:
        top_detection = _top_detection(filtered_detections)

    pipeline_status = _coerce_status(_first_present(observation, "status") if observation else None)
    last_frame_age = _last_frame_age(observation, timestamp=timestamp)
    placeholder = _truthy(_first_present(observation, "placeholder")) if observation else False
    not_wired = _status_is_not_wired(status) or bool(
        observation and _truthy(_first_present(observation, "not_wired"))
    )
    compat_static = status == "compat_static" or _is_compat_static_observation(observation)
    degraded = status == "degraded" or bool(observation and _truthy(_first_present(observation, "degraded")))
    stale = status == "stale" or _is_stale_observation(observation, status=pipeline_status or "")
    stream_ready = _stream_ready(
        observation,
        wired=wired,
        not_wired=not_wired,
        compat_static=compat_static,
        stale=stale,
        degraded=degraded,
    )
    status_reason = _string_or_none(_first_nested_present(observation, "status_reason")) or _default_status_reason(
        status=status,
        stream_ready=stream_ready,
        not_wired=not_wired,
        compat_static=compat_static,
        stale=stale,
        degraded=degraded,
    )
    not_wired_reason = _string_or_none(_first_nested_present(observation, "not_wired_reason"))
    stale_reason = _string_or_none(_first_nested_present(observation, "stale_reason"))
    degraded_reason = _string_or_none(_first_nested_present(observation, "degraded_reason"))
    readiness = _readiness_payload(
        _first_nested_present(observation, "readiness"),
        ready=stream_ready,
        reason=status_reason,
    )
    compatibility_static_image = _compat_static_payload(
        _first_nested_present(observation, "compatibility_static_image"),
        active=compat_static,
    )
    frame_interval_ms = _number_or_none(_first_nested_present(observation, "frame_interval_ms", "frame_interval"))
    jitter_guard = _first_nested_present(observation, "jitter_guard", "jitter")
    hooks_used = _hooks_used(_first_nested_present(observation, "hooks_used", "hooks"))
    pipeline = _json_object_or_none(_first_nested_present(observation, "pipeline"))
    devices = _json_object_or_none(_first_nested_present(observation, "devices", "device_paths"))
    readiness_message = _string_or_none(_first_nested_present(observation, "readiness_message", "message"))
    parse_error_count = _int_or_none(_first_nested_present(observation, "parse_error_count"))
    parse_errors = _parse_errors(_first_nested_present(observation, "parse_errors", "errors"))
    source_freshness = _source_freshness(
        observation,
        source=source,
        status=status,
        wired=wired,
        stream_ready=stream_ready,
        stale=stale,
        degraded=degraded,
        not_wired=not_wired,
        last_frame_age=last_frame_age,
    )
    tracks = _tracks_payload(observation)
    events = _events_payload(observation)
    detections_summary = _detections_summary(filtered_detections, source_freshness=source_freshness)
    health_state = _health_state(source_freshness=source_freshness, degraded=degraded, stale=stale)
    overlay = _build_visual_overlay(
        observation,
        filtered_detections=filtered_detections,
        stream_ready=stream_ready,
    )
    scene = _scene_payload(observation)
    target = _target_payload(observation, overlay.get("top_target"))
    score_labels = _score_labels(overlay, target)
    tracking_stability = _tracking_stability_payload(observation)
    multimodal_availability = _multimodal_availability(observation)

    return {
        "status": status,
        "pipeline_status": pipeline_status,
        "wired": bool(wired),
        "not_wired": bool(not_wired),
        "placeholder": bool(placeholder),
        "stream_ready": stream_ready,
        "compat_static": bool(compat_static),
        "stale": bool(stale),
        "degraded": bool(degraded),
        "status_reason": status_reason,
        "not_wired_reason": not_wired_reason,
        "stale_reason": stale_reason,
        "degraded_reason": degraded_reason,
        "readiness": readiness,
        "compatibility_static_image": compatibility_static_image,
        "backend": _backend(observation),
        "frame_id": _frame_id(observation),
        "fps": _number_or_none(_first_nested_present(observation, "fps")),
        "last_frame_age": last_frame_age,
        "last_frame_age_s": last_frame_age,
        "detection_count": len(filtered_detections),
        "boxes": boxes,
        "scores": scores,
        "top_detection": top_detection,
        "detection_count_raw": len(detections),
        "score_threshold": score_threshold,
        "detection_score_threshold": score_threshold,
        "top_k": top_k,
        "frame_interval_ms": frame_interval_ms,
        "jitter_guard": _coerce_optional_bool(jitter_guard),
        "hooks_used": hooks_used,
        "pipeline": pipeline,
        "devices": devices,
        "readiness_message": readiness_message,
        "parse_error_count": parse_error_count,
        "parse_errors": parse_errors,
        "detections": filtered_detections,
        "source_freshness": source_freshness,
        "latency_ms": _latency_ms(observation),
        "scene": scene,
        "scene_id": scene.get("scene_id"),
        "scene_summary": scene.get("summary"),
        "scene_graph_summary": scene.get("graph_summary") or scene.get("summary") or "waiting",
        "tracks": tracks,
        "track_count": tracks["count"],
        "track_summary": tracks["summary"],
        "tracking_stability": tracking_stability,
        "tracking_stability_score": tracking_stability.get("score"),
        "tracking_switch_count": _int_or_zero(_first_nested_present(observation, "tracking_switch_count", "switch_count", "switches")),
        "tracking_lost_count": _int_or_zero(_first_nested_present(observation, "tracking_lost_count", "lost_count", "lost")),
        "tracking_reacquired_count": _int_or_zero(_first_nested_present(observation, "tracking_reacquired_count", "reacquired_count", "reacquired")),
        "multimodal_availability": multimodal_availability,
        "pose_availability": multimodal_availability["pose"]["status"],
        "clip_availability": multimodal_availability["clip"]["status"],
        "depth_availability": multimodal_availability["depth"]["status"],
        "events": events,
        "event_count": events["count"],
        "event_summary": events["summary"],
        "score_labels": score_labels,
        "target": target,
        "target_center": target.get("center") if target else None,
        "target_error": target.get("error") if target else None,
        "target_score_label": target.get("score_label") if target else None,
        "detections_summary": detections_summary,
        "health_state": health_state,
        "overlay": overlay,
        "visual_diagnostic": overlay,
    }


def _is_compat_static_observation(observation: Mapping[str, Any] | None) -> bool:
    if observation is None:
        return False
    kind = str(observation.get("kind", "") or "").strip().lower()
    mode = str(observation.get("mode", "") or "").strip().lower()
    primary_mode = observation.get("primary_mode")
    compatibility_mode = _truthy(observation.get("compatibility_mode", False))
    return (
        kind == "vision_observation"
        or mode in {VISION_STATIC_COMPAT_MODE, "compat_static_frame"}
        or primary_mode is False
        or compatibility_mode
    )


def _is_stale_observation(observation: Mapping[str, Any] | None, *, status: str = "") -> bool:
    if observation is None:
        return False
    if _coerce_status(status) == "stale":
        return True
    if _truthy(observation.get("stale")):
        return True
    for key in ("health", "stream", "payload"):
        nested = observation.get(key)
        if isinstance(nested, Mapping) and _truthy(nested.get("stale")):
            return True
    return False


def _stream_ready(
    observation: Mapping[str, Any] | None,
    *,
    wired: bool,
    not_wired: bool,
    compat_static: bool,
    stale: bool,
    degraded: bool,
) -> bool:
    explicit = _first_nested_present(observation, "stream_ready")
    if explicit is not None:
        return _truthy(explicit)
    return bool(wired and not not_wired and not compat_static and not stale and not degraded)


def _default_status_reason(
    *,
    status: str,
    stream_ready: bool,
    not_wired: bool,
    compat_static: bool,
    stale: bool,
    degraded: bool,
) -> str:
    if compat_static:
        return "compat_static_frame_test_only"
    if stale:
        return "last_frame_stale"
    if degraded:
        return "degraded"
    if not_wired:
        return "not_wired"
    if stream_ready:
        return "realtime_stream_ready"
    return status or "unknown"


def _readiness_payload(value: Any, *, ready: bool, reason: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = {str(k): _json_ready(v) for k, v in value.items()}
        payload.setdefault("ready", ready)
        payload.setdefault("reason", reason)
        return payload
    return {"ready": ready, "reason": reason}


def _compat_static_payload(value: Any, *, active: bool) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = {str(k): _json_ready(v) for k, v in value.items()}
        payload.setdefault("active", active)
        payload.setdefault("test_only", active)
        return payload
    return {
        "active": active,
        "mode": VISION_STATIC_COMPAT_MODE if active else "",
        "test_only": active,
    }


def _normalized_detections(raw_detections: Any) -> list[dict[str, Any]]:
    if raw_detections is None:
        return []
    if isinstance(raw_detections, Mapping) or isinstance(raw_detections, (str, bytes)):
        raw_items = [raw_detections]
    else:
        try:
            raw_items = list(raw_detections)
        except TypeError:
            raw_items = [raw_detections]
    return [
        detection
        for detection in (_normalized_detection(item) for item in raw_items)
        if detection is not None
    ]


def _coerce_status(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "ok" if value else "not_wired"
    status = str(value).strip().lower()
    if status in {"", "none", "null", "false", "0"}:
        return ""
    if status in {"not_wired", "offline", "missing", "unavailable"}:
        return "not_wired"
    return status


def _status_is_not_wired(status: str) -> bool:
    return _coerce_status(status) == "not_wired"


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "0"}:
            return None
        if lowered in {"1", "true", "yes", "on", "enabled", "enable", "open"}:
            return True
        if lowered in {"false", "off", "disabled", "disable", "close", "closed"}:
            return False
        return bool(lowered)
    return bool(value)


def _detection_score_threshold(observation: Mapping[str, Any] | None) -> float:
    if observation is None:
        return 0.0
    raw = _first_nested_present(
        observation,
        "score_threshold",
        "detection_score_threshold",
        "filter_score_threshold",
    )
    threshold = _number_or_none(raw)
    if threshold is None or threshold < 0.0:
        return 0.0
    return threshold


def _detection_top_k(observation: Mapping[str, Any] | None) -> int | None:
    if observation is None:
        return None
    raw = _first_nested_present(observation, "top_k", "max_detections", "max_detection_count")
    value = _number_or_none(raw)
    if value is None:
        return None
    rounded = int(value)
    if rounded <= 0:
        return None
    return rounded


def _filter_detections(
    detections: list[dict[str, Any]],
    *,
    score_threshold: float,
    top_k: int | None,
) -> tuple[list[dict[str, Any]], list[float]]:
    with_scores: list[tuple[dict[str, Any], float | None]] = []
    for detection in detections:
        score = _detection_score(detection)
        if score is not None and score < score_threshold:
            continue
        with_scores.append((detection, score))

    if top_k is not None:
        with_scores = sorted(with_scores, key=lambda item: item[1] if item[1] is not None else -1.0, reverse=True)
        with_scores = with_scores[:top_k]

    filtered = [detection for detection, _ in with_scores]
    filtered_scores = [score for _, score in with_scores if score is not None]
    return filtered, filtered_scores


def _hooks_used(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, bytes, Mapping)):
        return [_json_ready(value)]
    return [_json_ready(value)]


def _parse_errors(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return [_json_ready(value)]


def _source_freshness(
    observation: Mapping[str, Any] | None,
    *,
    source: str | None,
    status: str,
    wired: bool,
    stream_ready: bool,
    stale: bool,
    degraded: bool,
    not_wired: bool,
    last_frame_age: float | None,
) -> dict[str, Any]:
    simulated = _is_simulated_source(observation)
    offline = not bool(wired) or not_wired or _status_is_not_wired(status)
    if offline:
        state = "offline"
    elif stale:
        state = "stale"
    elif simulated:
        state = "simulated"
    elif degraded:
        state = "degraded"
    elif stream_ready:
        state = "healthy"
    else:
        state = status or "unknown"
    return {
        "state": state,
        "healthy": state in {"healthy", "simulated"},
        "stale": bool(stale),
        "offline": bool(offline),
        "simulated": bool(simulated),
        "age_s": last_frame_age,
        "source": source,
    }


def _is_simulated_source(observation: Mapping[str, Any] | None) -> bool:
    if observation is None:
        return False
    for key in ("simulated", "replay"):
        if _truthy(_first_nested_present(observation, key)):
            return True
    transport = _first_nested_present(observation, "transport")
    source = _first_nested_present(observation, "source")
    source_text = _source_text(source)
    return (
        str(transport or "").lower() in {"simulated", "replay"}
        or "simulator" in source_text
        or "simulated" in source_text
    )


def _source_text(source: Any) -> str:
    if isinstance(source, Mapping):
        return " ".join(str(value).lower() for value in source.values())
    return str(source or "").lower()


def _latency_ms(observation: Mapping[str, Any] | None) -> float | None:
    latency = _first_nested_present(observation, "latency_ms", "latencyMs")
    number = _number_or_none(latency)
    if number is not None:
        return number
    latency_payload = _first_nested_present(observation, "latency")
    if isinstance(latency_payload, Mapping):
        for key in ("ms", "total_ms", "total", "latency_ms", "latencyMs"):
            number = _number_or_none(latency_payload.get(key))
            if number is not None:
                return number
    return _number_or_none(latency_payload)


def _tracks_payload(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_tracks = _first_nested_present(observation, "tracks", "tracked_targets")
    if raw_tracks is None:
        raw_tracks = _first_present(_scene_source(observation), "tracks", "tracked_targets")
    if raw_tracks is None:
        raw_tracks = _first_present(_scene_source(observation), "objects")
    if raw_tracks is None:
        raw_tracks = _first_nested_present(observation, "tracked_target")
    tracks = _normalized_items(raw_tracks)
    return {
        "count": len(tracks),
        "items": tracks,
        "summary": _items_summary(tracks),
    }


def _events_payload(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_events = _first_nested_present(observation, "events", "recent_events")
    if raw_events is None:
        raw_events = _first_present(_scene_source(observation), "events", "recent_events")
    events = _normalized_items(raw_events)
    return {
        "count": len(events),
        "items": events,
        "summary": _items_summary(events, label_keys=("event_type", "eventType", "type", "name", "label")),
    }


def _scene_payload(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    scene_source = _scene_source(observation)
    scene_id = _string_or_none(
        _first_present(
            observation,
            "scene_id",
            "sceneId",
        )
    ) or _string_or_none(
        _first_present(
            scene_source,
            "scene_id",
            "sceneId",
            "id",
        )
    )
    summary = _string_or_none(
        _first_present(
            observation,
            "scene_summary",
            "sceneSummary",
            "sceneGraphSummary",
        )
    ) or _string_or_none(
        _first_present(
            scene_source,
            "summary",
            "scene_summary",
            "sceneSummary",
            "sceneGraphSummary",
        )
    )
    snapshot = _json_object_or_none(_first_present(observation, "sceneSnapshot", "scene_snapshot"))
    graph_summary = _string_or_none(_first_present(observation, "sceneGraphSummary", "scene_graph_summary"))
    return {
        "scene_id": scene_id,
        "summary": summary,
        "snapshot": snapshot,
        "graph_summary": graph_summary,
    }


def _scene_source(observation: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    raw_scene = _first_present(observation, "scene", "scene_bridge", "sceneBridge", "sceneSnapshot", "scene_snapshot")
    normalized = _json_ready(raw_scene)
    if isinstance(normalized, Mapping):
        return {str(k): v for k, v in normalized.items()}
    return None


def _tracking_stability_payload(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = _first_nested_present(observation, "tracking_stability", "stability")
    payload = {str(k): _json_ready(v) for k, v in raw.items()} if isinstance(raw, Mapping) else {}
    state = _string_or_none(_first_present(payload, "state", "status"))
    if state is None:
        state = _string_or_none(_first_nested_present(observation, "stability_state")) or "unknown"
    score = _number_or_none(
        _first_present(payload, "score", "stability_score")
        if payload
        else _first_nested_present(observation, "tracking_stability_score", "stability_score", "stable_ratio")
    )
    return {
        **payload,
        "state": state,
        "score": round(score, 6) if score is not None else None,
    }


def _multimodal_availability(observation: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        "pose": _availability_payload(_multimodal_value(observation, "pose")),
        "clip": _availability_payload(_multimodal_value(observation, "clip", "clipLabels", "clip_labels")),
        "semantic": _availability_payload(
            _multimodal_value(observation, "semantic", "semanticLabels", "semantic_labels")
        ),
        "depth": _availability_payload(_multimodal_value(observation, "depth")),
        "distance": _availability_payload(_multimodal_value(observation, "distance")),
        "tracking": _availability_payload(
            _multimodal_value(observation, "trackingDiagnostics", "tracking_diagnostics")
        ),
    }


def _multimodal_value(observation: Mapping[str, Any] | None, *keys: str) -> Any:
    value = _first_nested_present(observation, *keys)
    if value is not None:
        return value
    scene = _scene_source(observation)
    sources = [scene] if scene is not None else []
    if observation is not None:
        sources.append(observation)
    value = _first_present(scene, *keys)
    if value is not None:
        return value
    collected: list[Any] = []
    collected_from_list = False
    for source in sources:
        if source is None:
            continue
        for collection_key in ("objects", "tracks", "events", "event_contents", "detections"):
            items = source.get(collection_key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                item_value = _first_present(item, *keys)
                if item_value is None:
                    attributes = item.get("attributes")
                    if isinstance(attributes, Mapping):
                        item_value = _first_present(attributes, *keys)
                if item_value is None:
                    continue
                if isinstance(item_value, list):
                    collected_from_list = True
                    collected.extend(item_value)
                else:
                    collected.append(item_value)
    if not collected:
        return None
    return collected if collected_from_list or len(collected) > 1 else collected[0]


def _availability_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "unknown", "summary": "unknown"}
    if isinstance(value, bool):
        return {"status": "present" if value else "waiting", "summary": "available" if value else "waiting"}
    if isinstance(value, list):
        return {
            "status": "present" if value else "waiting",
            "summary": f"{len(value)} label(s)" if value else "waiting",
        }
    if not isinstance(value, Mapping):
        text = str(value)
        return {"status": text or "unknown", "summary": text or "unknown"}

    status = _string_or_none(_first_present(value, "status", "state"))
    available = value.get("available")
    if status is None and isinstance(available, bool):
        status = "present" if available else "waiting"
    normalized = (status or "unknown").strip().lower()
    if normalized in {"ok", "ready", "available", "present", "live", "healthy", "enabled"}:
        normalized = "present"
    elif normalized in {"missing", "false", "unavailable", "disabled", "pending"}:
        normalized = "waiting"
    summary = _string_or_none(_first_present(value, "summary", "reason", "message")) or (
        "available" if normalized == "present" else normalized
    )
    return {"status": normalized, "summary": summary}


def _target_payload(
    observation: Mapping[str, Any] | None,
    overlay_target: Any,
) -> dict[str, Any] | None:
    if isinstance(overlay_target, Mapping):
        return {str(k): _json_ready(v) for k, v in overlay_target.items()}

    raw_target = _first_nested_present(observation, "target", "tracked_target", "top_target")
    normalized = _json_ready(raw_target)
    if not isinstance(normalized, Mapping):
        return None
    target = {str(k): v for k, v in normalized.items()}
    center = _point_payload(_first_present(target, "center", "target_center"))
    if center is None:
        center = _xy_payload(_first_present(target, "center_x", "x"), _first_present(target, "center_y", "y"))
    error = _point_payload(_first_present(target, "error", "target_error"))
    if error is None:
        error = _xy_payload(_first_present(target, "error_x", "dx"), _first_present(target, "error_y", "dy"))
    if error is None and center is not None:
        error = {
            "x": _round_overlay_number(float(center["x"]) - 0.5),
            "y": _round_overlay_number(float(center["y"]) - 0.5),
        }
    label = _string_or_none(_first_present(target, "label", "class", "name")) or "target"
    score = _number_or_none(_first_present(target, "score", "confidence"))
    return {
        "label": label,
        "score": score,
        "score_label": _score_label(label, score),
        "center": center,
        "error": error,
    }


def _score_labels(overlay: Mapping[str, Any], target: Mapping[str, Any] | None) -> list[str]:
    raw_labels = overlay.get("score_labels")
    if isinstance(raw_labels, (list, tuple)):
        labels = [str(item) for item in raw_labels if item not in (None, "")]
        if labels:
            return labels
    if target and target.get("score_label") not in (None, ""):
        return [str(target["score_label"])]
    return []


def _point_payload(value: Any) -> dict[str, float] | None:
    normalized = _json_ready(value)
    if not isinstance(normalized, Mapping):
        return None
    return _xy_payload(normalized.get("x"), normalized.get("y"))


def _xy_payload(x_value: Any, y_value: Any) -> dict[str, float] | None:
    x = _number_or_none(x_value)
    y = _number_or_none(y_value)
    if x is None or y is None:
        return None
    return {"x": _round_overlay_number(x), "y": _round_overlay_number(y)}


def _normalized_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raw_items = [value]
    else:
        try:
            raw_items = list(value)
        except TypeError:
            raw_items = [value]
    items: list[dict[str, Any]] = []
    for item in raw_items:
        normalized = _json_ready(item)
        if isinstance(normalized, Mapping):
            items.append({str(k): v for k, v in normalized.items()})
        else:
            items.append({"value": normalized, "payload_type": type(item).__name__})
    return items


def _detections_summary(
    detections: list[dict[str, Any]],
    *,
    source_freshness: Mapping[str, Any],
) -> str:
    if not detections:
        if source_freshness.get("offline") is True:
            return "no detections (offline)"
        if source_freshness.get("stale") is True:
            return "no detections (stale)"
        return "no detections"
    return _items_summary(detections, label_keys=("label", "class", "name"))


def _items_summary(items: list[dict[str, Any]], *, label_keys: tuple[str, ...] = ("label", "name", "track_id")) -> str:
    if not items:
        return "none"
    parts: list[str] = []
    for item in items:
        label = _string_or_none(_first_present(item, *label_keys)) or "item"
        score = _number_or_none(_first_present(item, "score", "confidence"))
        parts.append(_score_label(label, score))
    return ", ".join(parts) if parts else "none"


def _health_state(*, source_freshness: Mapping[str, Any], degraded: bool, stale: bool) -> str:
    if source_freshness.get("offline") is True:
        return "offline"
    if stale:
        return "stale"
    if degraded:
        return "degraded"
    if source_freshness.get("healthy") is True:
        return "healthy"
    return str(source_freshness.get("state") or "unknown")


def _normalized_detection(raw_detection: Any) -> dict[str, Any] | None:
    if raw_detection is None:
        return None
    if isinstance(raw_detection, Mapping):
        detection = {str(k): _json_ready(v) for k, v in raw_detection.items()}
    else:
        detection = _json_ready(raw_detection)
        if not isinstance(detection, Mapping):
            return {"value": detection, "payload_type": type(raw_detection).__name__}
        detection = {str(k): _json_ready(v) for k, v in detection.items()}
    if "bbox" in detection:
        normalized_box = _normalize_box(detection.get("bbox"), format_hint=_bbox_format(detection))
        detection["bbox"] = normalized_box if normalized_box is not None else detection.get("bbox")
    return detection


def _build_visual_overlay(
    observation: Mapping[str, Any] | None,
    *,
    filtered_detections: list[dict[str, Any]],
    stream_ready: bool,
) -> dict[str, Any]:
    frame_width = _frame_dimension(observation, "width", "frame_width", "image_width")
    frame_height = _frame_dimension(observation, "height", "frame_height", "image_height")
    normalized_boxes = [
        box
        for box in (
            _overlay_box(detection, frame_width=frame_width, frame_height=frame_height)
            for detection in filtered_detections
        )
        if box is not None
    ]
    score_labels = [
        box["score_label"]
        for box in normalized_boxes
        if box.get("score_label") not in (None, "")
    ]
    top_target = _overlay_top_target(normalized_boxes)
    return {
        "frame": {
            "width": frame_width,
            "height": frame_height,
            "frame_id": _frame_id(observation),
            "image_available": False,
            "image_message": "no live frame image yet",
        },
        "stream_ready": bool(stream_ready),
        "normalized_boxes": normalized_boxes,
        "score_labels": score_labels,
        "top_target": top_target,
    }


def _frame_dimension(observation: Mapping[str, Any] | None, *keys: str) -> int | float | None:
    value = _first_nested_present(observation, *keys)
    if value is None and observation is not None:
        for nested_key in ("frame", "image", "frame_size"):
            nested = observation.get(nested_key)
            if isinstance(nested, Mapping):
                value = _first_present(nested, *keys)
                if value is not None:
                    break
    number = _number_or_none(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return _round_overlay_number(number)


def _overlay_box(
    detection: Mapping[str, Any],
    *,
    frame_width: int | float | None,
    frame_height: int | float | None,
) -> dict[str, Any] | None:
    raw_box = _normalize_box(detection.get("bbox"))
    if raw_box is None:
        return None
    x_min = _normalized_axis(raw_box.get("x_min"), frame_width)
    y_min = _normalized_axis(raw_box.get("y_min"), frame_height)
    x_max = _normalized_axis(raw_box.get("x_max"), frame_width)
    y_max = _normalized_axis(raw_box.get("y_max"), frame_height)
    if None in {x_min, y_min, x_max, y_max}:
        return None
    label = _string_or_none(_first_present(detection, "label", "class", "name")) or "target"
    score = _detection_score(detection)
    overlay_box: dict[str, Any] = {
        "label": label,
        "score": score,
        "score_label": _score_label(label, score),
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }
    return overlay_box


def _normalized_axis(value: Any, frame_dimension: int | float | None) -> float | None:
    number = _number_or_none(value)
    if number is None:
        return None
    if frame_dimension and number > 1.0:
        number = number / float(frame_dimension)
    return _round_overlay_number(number)


def _score_label(label: str | None, score: float | None) -> str:
    if label and score is not None:
        return f"{label} {score:.2f}"
    if label:
        return label
    if score is not None:
        return f"{score:.2f}"
    return "target"


def _overlay_top_target(normalized_boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not normalized_boxes:
        return None
    top_box = max(
        normalized_boxes,
        key=lambda item: item.get("score") if isinstance(item.get("score"), (int, float)) else -1.0,
    )
    center_x = _round_overlay_number((float(top_box["x_min"]) + float(top_box["x_max"])) / 2.0)
    center_y = _round_overlay_number((float(top_box["y_min"]) + float(top_box["y_max"])) / 2.0)
    return {
        "label": top_box.get("label"),
        "score": top_box.get("score"),
        "score_label": top_box.get("score_label"),
        "center": {"x": center_x, "y": center_y},
        "error": {
            "x": _round_overlay_number(center_x - 0.5),
            "y": _round_overlay_number(center_y - 0.5),
        },
    }


def _round_overlay_number(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0 else rounded


def _normalize_box(raw_box: Any, *, format_hint: str = "") -> dict[str, Any] | None:
    if raw_box is None:
        return None
    if isinstance(raw_box, Mapping):
        normalized = _normalize_mapping_box(raw_box)
        return normalized or {str(k): _json_ready(v) for k, v in raw_box.items()}
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        x_min, y_min, x_max, y_max = _normalize_sequence_box(raw_box, format_hint=format_hint)
        return {
            "x_min": _json_ready(x_min),
            "y_min": _json_ready(y_min),
            "x_max": _json_ready(x_max),
            "y_max": _json_ready(y_max),
        }
    return None


def _normalize_sequence_box(raw_box: list[Any] | tuple[Any, ...], *, format_hint: str = "") -> tuple[float, float, float, float]:
    x_min = float(raw_box[0])
    y_min = float(raw_box[1])
    third = float(raw_box[2])
    fourth = float(raw_box[3])
    if format_hint == "xyxy":
        return (x_min, y_min, third, fourth)
    if format_hint == "xywh":
        return (x_min, y_min, x_min + third, y_min + fourth)
    if max(abs(x_min), abs(y_min), abs(third), abs(fourth)) <= 1.0:
        return (x_min, y_min, x_min + third, y_min + fourth)
    if third <= x_min or fourth <= y_min:
        return (x_min, y_min, x_min + third, y_min + fourth)
    return (x_min, y_min, third, fourth)


def _bbox_format(detection: Mapping[str, Any]) -> str:
    for key in ("bboxFormat", "bbox_format", "boxFormat", "box_format", "format"):
        value = detection.get(key)
        if value is not None:
            normalized = str(value).strip().lower().replace("-", "").replace("_", "")
            if normalized in {"xyxy", "x1y1x2y2"}:
                return "xyxy"
            if normalized in {"xywh", "ltwh"}:
                return "xywh"
    return ""


def _normalize_mapping_box(raw_box: Mapping[str, Any]) -> dict[str, Any] | None:
    for keys in (
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("x1", "y1", "x2", "y2"),
        ("left", "top", "right", "bottom"),
    ):
        if all(key in raw_box for key in keys):
            x_min, y_min, x_max, y_max = (_json_ready(raw_box[key]) for key in keys)
            return {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
    for keys in (("x", "y", "w", "h"), ("x", "y", "width", "height")):
        if all(key in raw_box for key in keys):
            x = _number_or_none(raw_box[keys[0]])
            y = _number_or_none(raw_box[keys[1]])
            width = _number_or_none(raw_box[keys[2]])
            height = _number_or_none(raw_box[keys[3]])
            if None in {x, y, width, height}:
                return None
            return {
                "x_min": _json_ready(x),
                "y_min": _json_ready(y),
                "x_max": _json_ready(x + width),
                "y_max": _json_ready(y + height),
            }
    return None


def _detection_score(detection: Mapping[str, Any]) -> float | None:
    score = _first_present(detection, "score")
    if score is None:
        score = _first_present(detection, "confidence")
    numeric_score = _number_or_none(score)
    return round(numeric_score, 6) if numeric_score is not None else None


def _top_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None
    return max(detections, key=lambda item: _detection_score(item) if _detection_score(item) is not None else -1.0)


def _backend(observation: Mapping[str, Any] | None) -> str | None:
    value = _first_nested_present(observation, "backend")
    if value is None:
        stream = observation.get("stream") if observation else None
        if isinstance(stream, Mapping):
            value = stream.get("transport")
    return _string_or_none(value)


def _frame_id(observation: Mapping[str, Any] | None) -> str | None:
    return _string_or_none(_first_nested_present(observation, "frame_id", "last_frame_id"))


def _last_frame_age(observation: Mapping[str, Any] | None, *, timestamp: float) -> float | None:
    age = _number_or_none(_first_nested_present(observation, "last_frame_age", "last_frame_age_s"))
    if age is not None:
        return age
    captured_at_ts = _number_or_none(
        _first_nested_present(observation, "last_frame_captured_at_ts", "captured_at_ts")
    )
    if captured_at_ts is None or captured_at_ts > timestamp:
        return None
    return round(max(0.0, timestamp - captured_at_ts), 4)


def _first_nested_present(observation: Mapping[str, Any] | None, *keys: str) -> Any:
    value = _first_present(observation, *keys)
    if value is not None:
        return value
    if observation is None:
        return None
    for nested_key in ("health", "stream", "payload"):
        nested = observation.get(nested_key)
        if isinstance(nested, Mapping):
            value = _first_present(nested, *keys)
            if value is not None:
                return value
    return None


def _first_present(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    for key in keys:
        if key in mapping and mapping[key] not in ("", None):
            return mapping[key]
    return None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _number_or_none(value)
    if number is None:
        return None
    return int(number)


def _int_or_zero(value: Any) -> int:
    number = _number_or_none(value)
    return int(number) if number is not None else 0


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _json_object_or_none(value: Any) -> Any:
    if value is None:
        return None
    return _json_ready(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "stale"}
    return bool(value)


def _resolve_realtime_observation_candidate(observation: Any, *, seen: set[int] | None = None) -> Any:
    if observation is None:
        return None
    seen = seen or set()
    candidate_id = id(observation)
    if candidate_id in seen:
        return observation
    seen.add(candidate_id)

    latest_status = getattr(observation, "latest_status", None)
    latest_observation = getattr(observation, "latest_observation", None)
    if callable(latest_observation):
        try:
            resolved = _resolve_realtime_observation_candidate(latest_observation(), seen=seen)
        except TypeError:
            resolved = None
        if resolved is not None:
            return resolved

    if latest_status is not None:
        resolved = _resolve_realtime_observation_candidate(latest_status, seen=seen)
        if resolved is not None:
            return resolved

    for method_name in ("status", "poll"):
        method = getattr(observation, method_name, None)
        if not callable(method):
            continue
        try:
            resolved = _resolve_realtime_observation_candidate(method(), seen=seen)
        except TypeError:
            continue
        if resolved is not None:
            return resolved

    return observation


def _serialize_observation(observation: Any) -> dict[str, Any]:
    if isinstance(observation, Mapping):
        return {str(k): _json_ready(v) for k, v in observation.items()}
    if hasattr(observation, "to_dict") and callable(observation.to_dict):
        payload = observation.to_dict()
        if isinstance(payload, Mapping):
            return {str(k): _json_ready(v) for k, v in payload.items()}
    if is_dataclass(observation):
        return {str(k): _json_ready(v) for k, v in asdict(observation).items()}
    try:
        payload = serialize_message(observation)
    except TypeError:
        return {
            "value": _json_ready(observation),
            "payload_type": type(observation).__name__,
        }
    return {str(k): _json_ready(v) for k, v in payload.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _json_ready(asdict(value))
    return str(value)


__all__ = [
    "REALTIME_VISION_ATTRS",
    "REALTIME_VISION_SCHEMA",
    "build_realtime_vision_payload",
    "realtime_vision_payload_from_app",
]
