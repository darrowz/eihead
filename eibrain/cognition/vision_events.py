"""Pure shaping helpers for realtime vision observation events.

The shaper accepts scene, target-lock, and tracking deltas and returns JSON-like
content dictionaries that can be handed to the generic eiprotocol vision-event
bridge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
import math
from typing import Any


DEFAULT_SOURCE = "vision.events"


class VisionEventShaper:
    """Turn visual deltas into throttled eiprotocol-friendly event contents."""

    def __init__(
        self,
        *,
        source: str | Mapping[str, Any] = DEFAULT_SOURCE,
        dedupe_window_ms: int | float = 1000,
        movement_threshold: float = 0.05,
    ) -> None:
        self.source = _jsonish(source)
        self.dedupe_window_ms = max(0.0, float(dedupe_window_ms))
        self.movement_threshold = max(0.0, float(movement_threshold))
        self._last_emitted_ms: dict[tuple[str, str, str], float] = {}
        self._last_target_locked = False
        self._last_target_subject: dict[str, Any] | None = None
        self._last_attention_subject: dict[str, Any] | None = None
        self._last_follow_state = ""
        self._logical_time_ms = 0.0

    def shape(
        self,
        *,
        scene_delta: Mapping[str, Any] | None = None,
        target_delta: Mapping[str, Any] | None = None,
        tracking_delta: Mapping[str, Any] | None = None,
        timestamp: str | int | float | None = None,
        timestamp_ms: int | float | None = None,
        source: str | Mapping[str, Any] | None = None,
        freshness_ms: int | float | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Shape one batch of visual deltas into observation event contents."""

        event_time_ms = self._event_time_ms(timestamp=timestamp, timestamp_ms=timestamp_ms)
        timestamp_text = _timestamp_text(timestamp=timestamp, timestamp_ms=event_time_ms, deltas=(scene_delta, target_delta, tracking_delta))
        resolved_source = _jsonish(source if source is not None else self.source)
        base_diagnostics = _dict_from(diagnostics)
        freshness = _round_float(freshness_ms, default=0.0)

        candidates: list[dict[str, Any]] = []
        candidates.extend(
            self._scene_events(
                scene_delta,
                timestamp=timestamp_text,
                timestamp_ms=event_time_ms,
                source=resolved_source,
                freshness_ms=freshness,
                base_diagnostics=base_diagnostics,
            )
        )
        candidates.extend(
            self._target_events(
                target_delta,
                timestamp=timestamp_text,
                timestamp_ms=event_time_ms,
                source=resolved_source,
                freshness_ms=freshness,
                base_diagnostics=base_diagnostics,
            )
        )
        candidates.extend(
            self._tracking_events(
                tracking_delta,
                timestamp=timestamp_text,
                timestamp_ms=event_time_ms,
                source=resolved_source,
                freshness_ms=freshness,
                base_diagnostics=base_diagnostics,
            )
        )

        return [event for event in candidates if self._allow_emit(event, event_time_ms)]

    def shape_events(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Compatibility alias for callers that use event-centric naming."""

        return self.shape(**kwargs)

    def update(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Compatibility alias for stateful realtime callers."""

        return self.shape(**kwargs)

    def _scene_events(
        self,
        scene_delta: Mapping[str, Any] | None,
        *,
        timestamp: str,
        timestamp_ms: float,
        source: Any,
        freshness_ms: float,
        base_diagnostics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(scene_delta, Mapping):
            return []
        raw_scene = _scene_mapping(scene_delta)
        scene_id = _first_text(raw_scene.get("sceneId"), raw_scene.get("scene_id"), scene_delta.get("sceneId"), scene_delta.get("scene_id"))
        frame_id = _first_text(raw_scene.get("frameId"), raw_scene.get("frame_id"), scene_delta.get("frameId"), scene_delta.get("frame_id"))
        common_diagnostics = {
            **_dict_from(base_diagnostics),
            **_dict_from(raw_scene.get("diagnostics")),
        }
        if frame_id:
            common_diagnostics["frameId"] = frame_id

        derived = _derive_scene_changes(raw_scene)
        events: list[dict[str, Any]] = []
        for item in _scene_items(raw_scene, "appeared", "person_appeared", "entered", "added", "new") + derived["appeared"]:
            subject = _subject_from(item)
            if not subject or not _is_person_subject(subject):
                continue
            events.append(
                _event_content(
                    event_type="person_appeared",
                    subject=subject,
                    confidence=_confidence(item),
                    timestamp=timestamp,
                    timestamp_ms=timestamp_ms,
                    source=source,
                    freshness_ms=freshness_ms,
                    scene_id=scene_id,
                    details=_movement_details(item),
                    diagnostics={**common_diagnostics, **_event_diagnostics(item)},
                )
            )

        for item in _scene_items(raw_scene, "left", "person_left", "disappeared", "removed", "missing") + derived["left"]:
            subject = _subject_from(item)
            if not subject or not _is_person_subject(subject):
                continue
            events.append(
                _event_content(
                    event_type="person_left",
                    subject=subject,
                    confidence=_confidence(item),
                    timestamp=timestamp,
                    timestamp_ms=timestamp_ms,
                    source=source,
                    freshness_ms=freshness_ms,
                    scene_id=scene_id,
                    details=_movement_details(item),
                    diagnostics={**common_diagnostics, **_event_diagnostics(item)},
                )
            )

        for item in _scene_items(raw_scene, "moved", "object_moved", "movements", "movement") + derived["moved"]:
            subject = _subject_from(item)
            if not subject or not self._is_significant_motion(item):
                continue
            events.append(
                _event_content(
                    event_type="object_moved",
                    subject=subject,
                    confidence=_confidence(item),
                    timestamp=timestamp,
                    timestamp_ms=timestamp_ms,
                    source=source,
                    freshness_ms=freshness_ms,
                    scene_id=scene_id,
                    details=_movement_details(item),
                    diagnostics={**common_diagnostics, **_event_diagnostics(item)},
                )
            )

        for item in _events_from_snapshot(raw_scene):
            events.extend(
                self._scene_events(
                    {"sceneId": scene_id, "frameId": frame_id, **item},
                    timestamp=timestamp,
                    timestamp_ms=timestamp_ms,
                    source=source,
                    freshness_ms=freshness_ms,
                    base_diagnostics=common_diagnostics,
                )
            )

        return events

    def _target_events(
        self,
        target_delta: Mapping[str, Any] | None,
        *,
        timestamp: str,
        timestamp_ms: float,
        source: Any,
        freshness_ms: float,
        base_diagnostics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(target_delta, Mapping):
            return []

        events: list[dict[str, Any]] = []
        diagnostics = {
            **_dict_from(base_diagnostics),
            **_dict_from(target_delta.get("diagnostics")),
        }
        reason = _first_text(
            target_delta.get("reason"),
            target_delta.get("switchReason"),
            target_delta.get("switch_reason"),
            target_delta.get("lockState"),
            target_delta.get("lock_state"),
            target_delta.get("status"),
        )
        if reason:
            diagnostics["reason"] = reason

        for item in _scene_items(target_delta, "locked", "target_locked"):
            subject = _subject_from(item)
            if subject:
                events.append(
                    _event_content(
                        event_type="target_locked",
                        subject=subject,
                        confidence=_confidence(item, fallback=_confidence(target_delta)),
                        timestamp=timestamp,
                        timestamp_ms=timestamp_ms,
                        source=source,
                        freshness_ms=freshness_ms,
                        details=_target_details(item, current_locked=True),
                        diagnostics={**diagnostics, **_event_diagnostics(item)},
                    )
                )
                self._last_target_locked = True
                self._last_target_subject = subject

        for item in _scene_items(target_delta, "lost", "target_lost"):
            subject = _subject_from(item) or self._last_target_subject
            if subject:
                events.append(
                    _event_content(
                        event_type="target_lost",
                        subject=subject,
                        confidence=_confidence(item, fallback=_confidence(target_delta)),
                        timestamp=timestamp,
                        timestamp_ms=timestamp_ms,
                        source=source,
                        freshness_ms=freshness_ms,
                        details=_target_details(item, current_locked=False),
                        diagnostics={**diagnostics, **_event_diagnostics(item)},
                    )
                )
                self._last_target_locked = False

        if events:
            return events

        current_raw = _first_mapping(
            target_delta.get("current"),
            target_delta.get("target"),
            target_delta.get("trackedTarget"),
            target_delta.get("tracked_target"),
            target_delta if _has_subject(target_delta) else None,
        )
        previous_raw = _first_mapping(
            target_delta.get("previous"),
            target_delta.get("previousTarget"),
            target_delta.get("previous_target"),
            target_delta.get("last"),
        )
        current_subject = _subject_from(current_raw) if current_raw is not None else None
        previous_subject = _subject_from(previous_raw) if previous_raw is not None else self._last_target_subject
        current_locked = _target_is_locked(target_delta, current_raw)
        explicit_previous = "previous" in target_delta or "previousTarget" in target_delta or "previous_target" in target_delta
        previous_locked = bool(previous_subject) if explicit_previous else self._last_target_locked

        if current_locked and current_subject and (not previous_locked or current_subject.get("trackId") != (previous_subject or {}).get("trackId")):
            events.append(
                _event_content(
                    event_type="target_locked",
                    subject=current_subject,
                    confidence=_confidence(current_raw, fallback=_confidence(target_delta)),
                    timestamp=timestamp,
                    timestamp_ms=timestamp_ms,
                    source=source,
                    freshness_ms=freshness_ms,
                    details=_target_details(current_raw, current_locked=True),
                    diagnostics={**diagnostics, **_event_diagnostics(current_raw)},
                )
            )
        elif previous_locked and not current_locked and previous_subject:
            events.append(
                _event_content(
                    event_type="target_lost",
                    subject=previous_subject,
                    confidence=_confidence(previous_raw, fallback=_confidence(target_delta)),
                    timestamp=timestamp,
                    timestamp_ms=timestamp_ms,
                    source=source,
                    freshness_ms=freshness_ms,
                    details=_target_details(previous_raw, current_locked=False),
                    diagnostics={**diagnostics, **_event_diagnostics(target_delta), **_event_diagnostics(previous_raw)},
                )
            )

        self._last_target_locked = current_locked
        self._last_target_subject = current_subject or (previous_subject if current_locked else self._last_target_subject)
        return events

    def _tracking_events(
        self,
        tracking_delta: Mapping[str, Any] | None,
        *,
        timestamp: str,
        timestamp_ms: float,
        source: Any,
        freshness_ms: float,
        base_diagnostics: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(tracking_delta, Mapping):
            return []

        events: list[dict[str, Any]] = []
        diagnostics = {
            **_dict_from(base_diagnostics),
            **_dict_from(tracking_delta.get("diagnostics")),
        }

        previous_tracking = _first_mapping(tracking_delta.get("previous"), tracking_delta.get("previousTracking"), tracking_delta.get("previous_tracking"))
        current_tracking = _first_mapping(tracking_delta.get("current"), tracking_delta.get("currentTracking"), tracking_delta.get("current_tracking"))
        attention = _first_mapping(tracking_delta.get("attention"), tracking_delta.get("attentionDelta"), tracking_delta.get("attention_delta"))
        if attention is None and (previous_tracking is not None or current_tracking is not None):
            previous_attention = _first_mapping((previous_tracking or {}).get("attention"))
            current_attention = _first_mapping((current_tracking or {}).get("attention"))
            if previous_attention is not None or current_attention is not None:
                attention = {"previous": previous_attention or {}, "current": current_attention or {}}
        if attention is not None:
            current_raw = _first_mapping(attention.get("current"), attention.get("target"), attention if _has_subject(attention) else None)
            previous_raw = _first_mapping(attention.get("previous"), attention.get("previousTarget"), attention.get("previous_target"))
            current_subject = _subject_from(current_raw) if current_raw is not None else None
            previous_subject = _subject_from(previous_raw) if previous_raw is not None else self._last_attention_subject
            if current_subject and current_subject.get("trackId") != (previous_subject or {}).get("trackId"):
                attention_diagnostics = {**diagnostics, **_event_diagnostics(attention)}
                if previous_subject:
                    attention_diagnostics["previousSubject"] = previous_subject
                events.append(
                    _event_content(
                        event_type="attention_changed",
                        subject=current_subject,
                        confidence=_confidence(current_raw, fallback=_confidence(attention)),
                        timestamp=timestamp,
                        timestamp_ms=timestamp_ms,
                        source=source,
                        freshness_ms=freshness_ms,
                        details={"previousTrackId": (previous_subject or {}).get("trackId", ""), "currentTrackId": current_subject.get("trackId", "")},
                        diagnostics=attention_diagnostics,
                    )
                )
            self._last_attention_subject = current_subject or self._last_attention_subject

        follow = _first_mapping(
            tracking_delta.get("followState"),
            tracking_delta.get("follow_state"),
            tracking_delta.get("follow"),
            tracking_delta.get("followDelta"),
            tracking_delta.get("follow_delta"),
        )
        if follow is None and any(key in tracking_delta for key in ("previousState", "previous_state", "currentState", "current_state")):
            follow = dict(tracking_delta)
        if follow is None and (previous_tracking is not None or current_tracking is not None):
            previous_state = _follow_state_from(previous_tracking)
            current_state = _follow_state_from(current_tracking)
            if previous_state or current_state:
                follow_subject = _subject_from(current_tracking) or _subject_from(_first_mapping((current_tracking or {}).get("attention")))
                follow = {
                    "previous": previous_state,
                    "current": current_state,
                    **(follow_subject or {}),
                }
        if follow is not None:
            previous_state = _first_text(follow.get("previous"), follow.get("previousState"), follow.get("previous_state"), self._last_follow_state)
            current_state = _first_text(follow.get("current"), follow.get("currentState"), follow.get("current_state"), follow.get("state"))
            subject = _subject_from(follow) or _subject_from(_first_mapping(follow.get("target"), follow.get("currentTarget"), follow.get("current_target")))
            if current_state and current_state != previous_state:
                events.append(
                    _event_content(
                        event_type="follow_state_changed",
                        subject=subject or {"trackId": "follow", "label": "follow"},
                        confidence=_confidence(follow),
                        timestamp=timestamp,
                        timestamp_ms=timestamp_ms,
                        source=source,
                        freshness_ms=freshness_ms,
                        details={"previousState": previous_state, "currentState": current_state},
                        diagnostics={**diagnostics, **_event_diagnostics(follow)},
                    )
                )
            if current_state:
                self._last_follow_state = current_state

        return events

    def _is_significant_motion(self, item: Mapping[str, Any]) -> bool:
        details = _movement_details(item)
        distance = _coerce_float(details.get("distance"), default=0.0)
        if distance >= self.movement_threshold:
            return True
        from_region = _first_text(details.get("fromRegion"))
        to_region = _first_text(details.get("toRegion"))
        return bool(from_region and to_region and from_region != to_region)

    def _allow_emit(self, event: Mapping[str, Any], timestamp_ms: float) -> bool:
        track_id = _first_text(event.get("trackId"), fallback="unknown")
        signature = _event_signature(event)
        key = (_first_text(event.get("eventType")), track_id, signature)
        last_ms = self._last_emitted_ms.get(key)
        if last_ms is not None and timestamp_ms - last_ms < self.dedupe_window_ms:
            return False
        self._last_emitted_ms[key] = timestamp_ms
        return True

    def _event_time_ms(self, *, timestamp: str | int | float | None, timestamp_ms: int | float | None) -> float:
        resolved = _coerce_optional_float(timestamp_ms)
        if resolved is None:
            resolved = _timestamp_ms_from_value(timestamp)
        if resolved is None:
            self._logical_time_ms += 1.0
            return self._logical_time_ms
        self._logical_time_ms = max(self._logical_time_ms, resolved)
        return resolved


def shape_vision_event_contents(
    *,
    scene_delta: Mapping[str, Any] | None = None,
    target_delta: Mapping[str, Any] | None = None,
    tracking_delta: Mapping[str, Any] | None = None,
    timestamp: str | int | float | None = None,
    timestamp_ms: int | float | None = None,
    source: str | Mapping[str, Any] = DEFAULT_SOURCE,
    freshness_ms: int | float | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    dedupe_window_ms: int | float = 1000,
    movement_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Stateless helper for one-off shaping of vision event contents."""

    return VisionEventShaper(
        source=source,
        dedupe_window_ms=dedupe_window_ms,
        movement_threshold=movement_threshold,
    ).shape(
        scene_delta=scene_delta,
        target_delta=target_delta,
        tracking_delta=tracking_delta,
        timestamp=timestamp,
        timestamp_ms=timestamp_ms,
        freshness_ms=freshness_ms,
        diagnostics=diagnostics,
    )


def shape_vision_events(**kwargs: Any) -> list[dict[str, Any]]:
    """Alias matching the module name used by some call sites."""

    return shape_vision_event_contents(**kwargs)


def to_eiprotocol_event_contents(delta: Mapping[str, Any] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Compatibility wrapper mirroring the realtime simulator naming."""

    if delta is not None:
        kwargs.setdefault("scene_delta", delta)
    return shape_vision_event_contents(**kwargs)


def _event_content(
    *,
    event_type: str,
    subject: Mapping[str, Any],
    confidence: float,
    timestamp: str,
    timestamp_ms: float,
    source: Any,
    freshness_ms: float,
    scene_id: str = "",
    details: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean_subject = _subject_from(subject) or {"trackId": "unknown", "label": "unknown"}
    track_id = _first_text(clean_subject.get("trackId"), fallback="unknown")
    event = {
        "eventId": _event_id(scene_id=scene_id, event_type=event_type, track_id=track_id, timestamp_ms=timestamp_ms),
        "eventType": event_type,
        "subject": clean_subject,
        "trackId": track_id,
        "confidence": _round_float(confidence, default=0.0),
        "timestamp": timestamp,
        "observedAt": timestamp,
        "sceneId": scene_id,
        "source": source,
        "freshnessMs": _round_float(freshness_ms, default=0.0),
        "details": _dict_from(details),
        "diagnostics": _dict_from(diagnostics),
    }
    return _jsonish(event)


def _event_id(*, scene_id: str, event_type: str, track_id: str, timestamp_ms: float) -> str:
    if scene_id:
        return f"{scene_id}:{event_type}:{track_id}"
    return f"vision_event:{event_type}:{track_id}:{int(timestamp_ms)}"


def _event_signature(event: Mapping[str, Any]) -> str:
    details = _dict_from(event.get("details"))
    event_type = _first_text(event.get("eventType"))
    if event_type == "follow_state_changed":
        return _first_text(details.get("currentState"))
    if event_type == "attention_changed":
        return _first_text(details.get("previousTrackId")) + ">" + _first_text(details.get("currentTrackId"))
    if event_type == "object_moved":
        return _first_text(details.get("toRegion"))
    if event_type in {"target_locked", "target_lost"}:
        diagnostics = _dict_from(event.get("diagnostics"))
        return _first_text(diagnostics.get("reason"))
    return ""


def _scene_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    scene = value.get("sceneDelta") or value.get("scene_delta") or value.get("sceneSnapshot") or value.get("scene")
    return scene if isinstance(scene, Mapping) else value


def _scene_items(scene: Mapping[str, Any], *keys: str) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for key in keys:
        value = scene.get(key)
        if isinstance(value, Mapping):
            items.append(value)
        elif isinstance(value, list):
            items.extend(item for item in value if isinstance(item, Mapping))
    return items


def _events_from_snapshot(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    events = scene.get("events")
    if not isinstance(events, list):
        return converted
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        event_type = _first_text(raw.get("eventType"), raw.get("event_type"), raw.get("type"))
        subject = _dict_from(raw.get("subject"))
        item = {**subject, **_dict_from(raw.get("details")), **_dict_from(raw.get("metadata"))}
        item.setdefault("confidence", raw.get("confidence"))
        if event_type in {"appeared", "person_appeared"}:
            converted.append({"appeared": [item]})
        elif event_type in {"disappeared", "left", "person_left"}:
            converted.append({"left": [item]})
        elif event_type in {"moved", "object_moved"}:
            converted.append({"moved": [item]})
    return converted


def _derive_scene_changes(scene: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    previous = _first_mapping(scene.get("previous"), scene.get("before"), scene.get("previousScene"), scene.get("previous_scene"))
    current = _first_mapping(scene.get("current"), scene.get("after"), scene.get("currentScene"), scene.get("current_scene"))
    if previous is None or current is None:
        return {"appeared": [], "left": [], "moved": []}

    previous_objects = _object_map(previous)
    current_objects = _object_map(current)
    appeared = [current_objects[track_id] for track_id in sorted(set(current_objects) - set(previous_objects))]
    left = [previous_objects[track_id] for track_id in sorted(set(previous_objects) - set(current_objects))]
    moved: list[Mapping[str, Any]] = []
    for track_id in sorted(set(previous_objects) & set(current_objects)):
        previous_item = previous_objects[track_id]
        current_item = current_objects[track_id]
        distance = _distance_between_objects(previous_item, current_item)
        from_region = _first_text(previous_item.get("region"), previous_item.get("fromRegion"), previous_item.get("from_region"))
        to_region = _first_text(current_item.get("region"), current_item.get("toRegion"), current_item.get("to_region"))
        if distance > 0.0 or (from_region and to_region and from_region != to_region):
            moved.append(
                {
                    **dict(current_item),
                    "fromRegion": from_region,
                    "toRegion": to_region,
                    "distance": distance,
                }
            )
    return {"appeared": appeared, "left": left, "moved": moved}


def _object_map(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for item in _object_items(scene):
        subject = _subject_from(item)
        if subject is None:
            continue
        mapped[str(subject["trackId"])] = item
    return mapped


def _object_items(scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("objects", "tracks", "detections"):
        value = scene.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _distance_between_objects(previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
    previous_center = _center_from(previous)
    current_center = _center_from(current)
    if previous_center is None or current_center is None:
        return 0.0
    return math.hypot(previous_center[0] - current_center[0], previous_center[1] - current_center[1])


def _movement_details(item: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = item if isinstance(item, Mapping) else {}
    from_region = _first_text(raw.get("fromRegion"), raw.get("from_region"), raw.get("from"))
    to_region = _first_text(raw.get("toRegion"), raw.get("to_region"), raw.get("to"), raw.get("region"))
    distance = _coerce_float(raw.get("distance"), default=_distance_from_item(raw))
    return {
        "fromRegion": from_region,
        "toRegion": to_region,
        "distance": _round_float(distance, default=0.0),
    }


def _target_details(item: Mapping[str, Any] | None, *, current_locked: bool) -> dict[str, Any]:
    raw = item if isinstance(item, Mapping) else {}
    details = {
        "locked": current_locked,
        "lockState": "locked" if current_locked else "lost",
    }
    for key in ("target_x", "targetX", "center", "bbox", "tracking_target_error_x", "trackingTargetErrorX"):
        if key in raw:
            details[str(key)] = _jsonish(raw[key])
    return details


def _event_diagnostics(item: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {}
    diagnostics = _dict_from(item.get("diagnostics"))
    for key in ("reason", "status", "switchReason", "switch_reason", "lockState", "lock_state"):
        if item.get(key) not in (None, ""):
            diagnostics.setdefault(_camel_key(key), _jsonish(item[key]))
    return diagnostics


def _subject_from(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        return None
    raw_subject = item.get("subject")
    if isinstance(raw_subject, Mapping):
        subject = _subject_from(raw_subject)
        if subject is not None:
            return subject
    track_id = _first_text(item.get("trackId"), item.get("track_id"), item.get("id"), item.get("target_id"), item.get("targetId"))
    label = _first_text(item.get("label"), item.get("name"), item.get("class"), item.get("type"), fallback="object")
    if not track_id:
        return None
    subject: dict[str, Any] = {"trackId": track_id, "label": label}
    for key in ("personId", "person_id", "registered_identity", "identity"):
        if item.get(key) not in (None, ""):
            subject[_camel_key(key)] = _jsonish(item[key])
    return subject


def _has_subject(item: Mapping[str, Any]) -> bool:
    return any(key in item for key in ("subject", "trackId", "track_id", "id", "target_id", "targetId"))


def _is_person_subject(subject: Mapping[str, Any]) -> bool:
    label = _first_text(subject.get("label")).lower()
    return label in {"person", "human", "face", "registered_person", "registeredperson"}


def _target_is_locked(target_delta: Mapping[str, Any], current_raw: Mapping[str, Any] | None) -> bool:
    value = _first_present(
        target_delta.get("isLocked"),
        target_delta.get("is_locked"),
        target_delta.get("locked"),
        target_delta.get("tracking_locked"),
    )
    if value is not None:
        return _truthy(value)
    state = _first_text(target_delta.get("lockState"), target_delta.get("lock_state"), target_delta.get("status"))
    if state:
        return state in {"locked", "tracking", "holding_target", "lost_hold", "switched"}
    return current_raw is not None


def _follow_state_from(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return ""
    return _first_text(
        value.get("followState"),
        value.get("follow_state"),
        value.get("follow"),
        value.get("state"),
        value.get("status"),
    )


def _confidence(item: Mapping[str, Any] | None, *, fallback: float = 0.0) -> float:
    if not isinstance(item, Mapping):
        return _round_float(fallback, default=0.0)
    return _round_float(
        _first_present(item.get("confidence"), item.get("score"), item.get("target_score"), item.get("tracking_target_score")),
        default=fallback,
    )


def _distance_from_item(item: Mapping[str, Any]) -> float:
    previous = _first_mapping(item.get("previous"), item.get("fromObject"), item.get("from_object"))
    current = _first_mapping(item.get("current"), item.get("toObject"), item.get("to_object"))
    previous_center = _center_from(previous) or _center_from_bbox(_first_mapping(item.get("previousBbox"), item.get("previous_bbox")))
    current_center = _center_from(current) or _center_from_bbox(_first_mapping(item.get("bbox"), item.get("currentBbox"), item.get("current_bbox")))
    if previous_center is None or current_center is None:
        return 0.0
    return math.hypot(previous_center[0] - current_center[0], previous_center[1] - current_center[1])


def _center_from(item: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(item, Mapping):
        return None
    center = item.get("center")
    if isinstance(center, Mapping):
        x_value = _coerce_optional_float(center.get("x"))
        y_value = _coerce_optional_float(center.get("y"))
        if x_value is not None and y_value is not None:
            return (x_value, y_value)
    return _center_from_bbox(_first_mapping(item.get("bbox")))


def _center_from_bbox(bbox: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(bbox, Mapping):
        return None
    x_min = _coerce_optional_float(_first_present(bbox.get("x_min"), bbox.get("xmin"), bbox.get("left"), bbox.get("x1")))
    y_min = _coerce_optional_float(_first_present(bbox.get("y_min"), bbox.get("ymin"), bbox.get("top"), bbox.get("y1")))
    x_max = _coerce_optional_float(_first_present(bbox.get("x_max"), bbox.get("xmax"), bbox.get("right"), bbox.get("x2")))
    y_max = _coerce_optional_float(_first_present(bbox.get("y_max"), bbox.get("ymax"), bbox.get("bottom"), bbox.get("y2")))
    if None in (x_min, y_min, x_max, y_max):
        return None
    return ((float(x_min) + float(x_max)) / 2.0, (float(y_min) + float(y_max)) / 2.0)


def _timestamp_text(
    *,
    timestamp: str | int | float | None,
    timestamp_ms: float,
    deltas: Iterable[Mapping[str, Any] | None],
) -> str:
    if isinstance(timestamp, str) and timestamp:
        return timestamp
    for delta in deltas:
        if not isinstance(delta, Mapping):
            continue
        observed = _first_text(
            delta.get("observedAt"),
            delta.get("observed_at"),
            delta.get("timestamp"),
            delta.get("capturedAt"),
            delta.get("captured_at"),
        )
        if observed:
            return observed
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp_ms_from_value(value: str | int | float | None) -> float | None:
    number = _coerce_optional_float(value)
    if number is not None:
        return number
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp() * 1000.0
    except ValueError:
        return None


def _dict_from(value: Any) -> dict[str, Any]:
    return _jsonish(dict(value)) if isinstance(value, Mapping) else {}


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _first_text(*values: Any, fallback: str = "") -> str:
    for value in values:
        if value is not None and value != "":
            return str(value)
    return fallback


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    number = _coerce_optional_float(value)
    return default if number is None else number


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round_float(value: Any, *, default: float = 0.0) -> float:
    number = _coerce_float(value, default=default)
    return round(number, 3)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "locked", "active"}


def _camel_key(key: str) -> str:
    replacements = {
        "switch_reason": "switchReason",
        "lock_state": "lockState",
        "person_id": "personId",
        "registered_identity": "registeredIdentity",
    }
    return replacements.get(str(key), str(key))


def _jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    if isinstance(value, set):
        return [_jsonish(item) for item in sorted(value, key=str)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonish(value.to_dict())
        except TypeError:
            pass
    return str(value)


__all__ = [
    "VisionEventShaper",
    "shape_vision_event_contents",
    "shape_vision_events",
    "to_eiprotocol_event_contents",
]
