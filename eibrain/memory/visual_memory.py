"""Visual memory filtering and compression policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from time import time
from typing import Any, Mapping


SOURCE = "eibrain.vision"
TRACE_SCHEMA = "eibrain.memory.visual_trace.v1"
CONTRACT_VERSION = "visual-memory.v1"

MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000

NOISE_EVENT_TYPES = {
    "",
    "frame",
    "visual_frame",
    "video_frame",
    "detection",
    "detections",
    "object_detected",
}
REGISTERED_FACE_EVENT_TYPES = {
    "registered_face_recognized",
    "registered_face",
    "face_registered",
    "known_face",
}
TARGET_PRESENT_EVENT_TYPES = {
    "target_long_present",
    "target_present",
    "target_stable",
    "target_acquired",
}
TARGET_ABSENT_EVENT_TYPES = {
    "target_long_absent",
    "target_absent",
    "target_left",
    "target_lost",
}
INTERACTION_EVENT_TYPES = {
    "user_device_interaction",
    "device_interaction",
    "user_interaction",
    "touch",
    "button_press",
}
FOLLOW_SUCCESS_EVENT_TYPES = {
    "follow_success",
    "tracking_success",
    "target_followed",
}
FOLLOW_FAILED_EVENT_TYPES = {
    "follow_failed",
    "follow_failure",
    "tracking_failed",
    "tracking_failure",
}
USER_FEEDBACK_EVENT_TYPES = {
    "user_feedback",
    "feedback",
    "visual_feedback",
    "tracking_feedback",
}


@dataclass(slots=True)
class VisualMemoryPolicy:
    """Pure visual-event policy that returns inert eimemory candidates."""

    min_stable_frames: int = 8
    min_presence_ms: int = 8_000
    min_absence_ms: int = 5_000
    dedupe_window_ms: int = 10 * 60 * 1000
    source: str = SOURCE
    last_decision: dict[str, object] = field(default_factory=dict, init=False)
    _seen_at_ms: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def evaluate(
        self,
        *,
        event: object,
        target_lock: object | None = None,
        follow_score: object | None = None,
        context: object | None = None,
    ) -> dict[str, object] | None:
        """Return a memory trace/upsert candidate, or None for noise/duplicates."""

        candidate = build_visual_memory_candidate(
            event=event,
            target_lock=target_lock,
            follow_score=follow_score,
            context=context,
            min_stable_frames=self.min_stable_frames,
            min_presence_ms=self.min_presence_ms,
            min_absence_ms=self.min_absence_ms,
            source=self.source,
        )
        if candidate is None:
            self.last_decision = {
                "decision": "skip_visual_memory",
                "reason": "unstable_or_unimportant_visual_event",
            }
            return None

        dedupe_key = str(candidate["dedupe_key"])
        timestamp_ms = int(candidate["timestamp_ms"])
        last_seen_ms = self._seen_at_ms.get(dedupe_key)
        if last_seen_ms is not None and abs(timestamp_ms - last_seen_ms) < self.dedupe_window_ms:
            self.last_decision = {
                "decision": "skip_visual_memory",
                "reason": "duplicate_visual_event",
                "dedupe_key": dedupe_key,
            }
            return None

        self._seen_at_ms[dedupe_key] = timestamp_ms
        self.last_decision = {
            "decision": "visual_memory_candidate",
            "reason": "stable_or_important_visual_event",
            "dedupe_key": dedupe_key,
        }
        return candidate


def build_visual_memory_candidate(
    *,
    event: object,
    target_lock: object | None = None,
    follow_score: object | None = None,
    context: object | None = None,
    min_stable_frames: int = 8,
    min_presence_ms: int = 8_000,
    min_absence_ms: int = 5_000,
    source: str = SOURCE,
) -> dict[str, object] | None:
    """Build a JSON-ready eimemory upsert candidate from a visual event."""

    event_map = _to_plain_dict(event)
    lock_map = _to_plain_dict(target_lock)
    follow_map = _to_plain_dict(follow_score)
    context_map = _to_plain_dict(context)
    resolved_source = _clean_text(source) or SOURCE

    event_type = _classify_event(
        event_map=event_map,
        lock_map=lock_map,
        follow_map=follow_map,
        context_map=context_map,
        min_stable_frames=min_stable_frames,
        min_presence_ms=min_presence_ms,
        min_absence_ms=min_absence_ms,
    )
    if event_type is None:
        return None

    timestamp_ms = _timestamp_ms(event_map, lock_map, follow_map, context_map)
    session_id = _first_text_from_maps(context_map, event_map, keys=("session_id", "session", "conversation_id"))
    actor_id = _first_text_from_maps(context_map, event_map, keys=("actor_id", "user_id", "subject_id"))
    target_id = _target_id(event_type=event_type, event_map=event_map, lock_map=lock_map, context_map=context_map)
    summary = _summary_for(
        event_type=event_type,
        event_map=event_map,
        lock_map=lock_map,
        follow_map=follow_map,
        target_id=target_id,
    )
    importance = _importance_for(event_type=event_type, event_map=event_map, lock_map=lock_map, follow_map=follow_map)
    confidence = _confidence_for(event_map=event_map, follow_map=follow_map)
    retention = _retention_policy(event_type)
    dedupe_key = _dedupe_key(
        event_type=event_type,
        session_id=session_id,
        target_id=target_id,
        summary=summary,
        event_map=event_map,
    )
    event_reference = _event_reference(event_map=event_map, event_type=event_type, source=resolved_source)
    tracking_provenance = _tracking_provenance(lock_map=lock_map, follow_map=follow_map, event_reference=event_reference)
    scene_provenance = _scene_provenance(event_map=event_map, context_map=context_map, event_reference=event_reference)
    content = _content_for(
        event_type=event_type,
        summary=summary,
        event_map=event_map,
        lock_map=lock_map,
        follow_map=follow_map,
        context_map=context_map,
    )
    content["confidence"] = confidence
    content["tracking_provenance"] = tracking_provenance
    content["scene_provenance"] = scene_provenance

    meta: dict[str, object] = {
        "source_system": "eibrain",
        "source": resolved_source,
        "memory_contract_version": CONTRACT_VERSION,
        "event_type": event_type,
        "trace_id": event_reference["trace_id"],
        "source_event_id": event_reference["event_id"],
        "session_id": session_id,
        "eiprotocol_event_reference": event_reference,
        "event_reference": event_reference,
        "dedupe_key": dedupe_key,
        "importance": importance,
        "confidence": confidence,
        "tracking_provenance": tracking_provenance,
        "scene_provenance": scene_provenance,
        "ttl_ms": retention["ttl_ms"],
        "retention": retention["retention"],
        "memory_kind": retention["memory_kind"],
        "promotion_status": retention["promotion_status"],
        "training_candidate": retention["training_candidate"],
        "identity_memory": False,
        "persona_memory": False,
        "privacy": _privacy_for(event_type),
        "writeback": {
            "eligible": True,
            "durable": True,
            "reason": "important_visual_event",
            "target_memory_type": retention["memory_type"],
        },
    }

    trace = {
        "schema": TRACE_SCHEMA,
        "source": resolved_source,
        "event_type": event_type,
        "timestamp_ms": timestamp_ms,
        "session_id": session_id,
        "actor_id": actor_id,
        "dedupe_key": dedupe_key,
        "importance": importance,
        "confidence": confidence,
        "retention": retention["retention"],
        "ttl_ms": retention["ttl_ms"],
        "event_reference": event_reference,
        "tracking_provenance": tracking_provenance,
        "scene_provenance": scene_provenance,
        "decision": {
            "decision": "visual_memory_candidate",
            "why": _decision_reason(event_type),
            "upsert_candidate": True,
            "durable": True,
        },
    }
    scope = _scope(session_id=session_id, actor_id=actor_id)
    params: dict[str, object] = {
        "text": summary,
        "title": _title_for(event_type),
        "memory_type": retention["memory_type"],
        "source": resolved_source,
        "organ": "eye",
        "modality": "vision",
        "confidence": confidence,
        "content": content,
        "meta": meta,
        "tags": _tags_for(
            event_type=event_type,
            memory_type=str(retention["memory_type"]),
            retention=str(retention["retention"]),
            content=content,
        ),
    }
    if scope:
        params["scope"] = scope

    return _json_safe(
        {
            "kind": "visual_memory_candidate",
            "event_type": event_type,
            "source": resolved_source,
            "timestamp_ms": timestamp_ms,
            "dedupe_key": dedupe_key,
            "importance": importance,
            "retention": retention["retention"],
            "ttl_ms": retention["ttl_ms"],
            "memory_trace": trace,
            "upsert_payload": {
                "method": "memory.upsert",
                "params": params,
            },
        }
    )


def _classify_event(
    *,
    event_map: Mapping[str, object],
    lock_map: Mapping[str, object],
    follow_map: Mapping[str, object],
    context_map: Mapping[str, object],
    min_stable_frames: int,
    min_presence_ms: int,
    min_absence_ms: int,
) -> str | None:
    raw_type = _event_type(event_map)
    if raw_type in USER_FEEDBACK_EVENT_TYPES or _feedback_text(event_map, context_map):
        return "user_feedback"
    if raw_type in REGISTERED_FACE_EVENT_TYPES or _is_registered_face(event_map):
        return "registered_face_recognized"
    if raw_type in INTERACTION_EVENT_TYPES or _first_text(event_map, "interaction", "gesture", "control"):
        return "user_device_interaction"
    if raw_type in FOLLOW_FAILED_EVENT_TYPES or _follow_success(event_map, follow_map) is False:
        return "follow_failed"
    if raw_type in FOLLOW_SUCCESS_EVENT_TYPES or _follow_success(event_map, follow_map) is True:
        return "follow_success"
    if raw_type in TARGET_ABSENT_EVENT_TYPES or _target_long_absent(
        lock_map=lock_map,
        min_absence_ms=min_absence_ms,
    ):
        return "target_long_absent"
    if raw_type in TARGET_PRESENT_EVENT_TYPES or _target_long_present(
        lock_map=lock_map,
        min_stable_frames=min_stable_frames,
        min_presence_ms=min_presence_ms,
    ):
        return "target_long_present"
    if raw_type in NOISE_EVENT_TYPES:
        return None
    return None


def _event_type(event_map: Mapping[str, object]) -> str:
    return _normalize_token(_first_text(event_map, "event_type", "type", "kind", "name"))


def _is_registered_face(event_map: Mapping[str, object]) -> bool:
    raw_type = _event_type(event_map)
    if raw_type not in {"face_recognized", "face_match", "identity_match"}:
        return False
    if _coerce_bool(event_map.get("registered")) is True:
        return True
    return bool(_first_text(event_map, "person_id", "registered_person_id", "identity_id", "profile_id"))


def _target_long_present(*, lock_map: Mapping[str, object], min_stable_frames: int, min_presence_ms: int) -> bool:
    locked = _coerce_bool(_first_value(lock_map, "locked", "is_locked", "target_locked"))
    if locked is not True:
        return False
    if not _first_text(lock_map, "target_id", "track_id", "subject_id", "actor_id"):
        return False
    stable_frames = _extract_int(lock_map, "stable_frames", "consecutive_frames", "frames")
    duration_ms = _extract_int(lock_map, "duration_ms", "stable_duration_ms", "visible_duration_ms")
    return stable_frames >= min_stable_frames or duration_ms >= min_presence_ms


def _target_long_absent(*, lock_map: Mapping[str, object], min_absence_ms: int) -> bool:
    locked = _coerce_bool(_first_value(lock_map, "locked", "is_locked", "target_locked"))
    if locked is True:
        return False
    if not _first_text(lock_map, "target_id", "track_id", "subject_id", "actor_id", "last_target_id"):
        return False
    lost_duration_ms = _extract_int(lock_map, "lost_duration_ms", "absent_duration_ms", "missing_duration_ms")
    return lost_duration_ms >= min_absence_ms


def _follow_success(event_map: Mapping[str, object], follow_map: Mapping[str, object]) -> bool | None:
    raw_success = _first_value(event_map, "success", "follow_success", "tracking_success")
    if raw_success is None:
        raw_success = _first_value(follow_map, "success", "follow_success", "tracking_success")
    success = _coerce_bool(raw_success)
    if success is not None:
        return success
    status = _normalize_token(_first_text_from_maps(event_map, follow_map, keys=("status", "outcome", "result")))
    if status in {"success", "succeeded", "ok", "stable", "completed"}:
        return True
    if status in {"failed", "failure", "lost", "timeout", "unstable", "error"}:
        return False
    return None


def _target_id(
    *,
    event_type: str,
    event_map: Mapping[str, object],
    lock_map: Mapping[str, object],
    context_map: Mapping[str, object],
) -> str:
    if event_type == "registered_face_recognized":
        return _first_text_from_maps(
            event_map,
            lock_map,
            context_map,
            keys=("person_id", "registered_person_id", "identity_id", "profile_id", "target_id", "actor_id", "user_id"),
        )
    if event_type == "user_device_interaction":
        return _first_text_from_maps(event_map, context_map, keys=("device_id", "target_id", "actor_id", "user_id"))
    if event_type == "user_feedback":
        feedback = _feedback_text(event_map, context_map)
        return _digest({"feedback": feedback}, length=12)
    return _first_text_from_maps(
        lock_map,
        event_map,
        context_map,
        keys=("target_id", "track_id", "subject_id", "actor_id", "user_id", "last_target_id"),
    )


def _summary_for(
    *,
    event_type: str,
    event_map: Mapping[str, object],
    lock_map: Mapping[str, object],
    follow_map: Mapping[str, object],
    target_id: str,
) -> str:
    name = _first_text(event_map, "display_name", "name", "label") or _first_text(lock_map, "label") or target_id
    if event_type == "registered_face_recognized":
        confidence = _extract_float(event_map, "confidence", "score")
        suffix = f" confidence={confidence:.2f}" if confidence > 0 else ""
        return f"Registered face recognized: {name}{suffix}"
    if event_type == "target_long_present":
        duration_ms = _extract_int(lock_map, "duration_ms", "stable_duration_ms", "visible_duration_ms")
        if duration_ms:
            return f"Target {name or 'subject'} remained visible for {duration_ms}ms"
        return f"Target {name or 'subject'} remained visible"
    if event_type == "target_long_absent":
        lost_duration_ms = _extract_int(lock_map, "lost_duration_ms", "absent_duration_ms", "missing_duration_ms")
        if lost_duration_ms:
            return f"Target {name or 'subject'} left view for {lost_duration_ms}ms"
        return f"Target {name or 'subject'} left view"
    if event_type == "user_device_interaction":
        interaction = _first_text(event_map, "interaction", "gesture", "control", "summary")
        if interaction:
            return f"User/device interaction: {interaction}"
        return f"User interacted with {name or 'device'}"
    if event_type == "follow_success":
        score = _extract_float(follow_map, "score", "follow_score", "tracking_score")
        return f"Visual target follow succeeded with score {score:.2f}" if score > 0 else "Visual target follow succeeded"
    if event_type == "follow_failed":
        error = _first_text(event_map, "error", "reason", "message")
        return f"Visual target follow failed: {error}" if error else "Visual target follow failed"
    if event_type == "user_feedback":
        feedback = _feedback_text(event_map, {})
        return f"Visual user feedback: {feedback}" if feedback else "Visual user feedback"
    return "Visual memory event"


def _importance_for(
    *,
    event_type: str,
    event_map: Mapping[str, object],
    lock_map: Mapping[str, object],
    follow_map: Mapping[str, object],
) -> float:
    if event_type == "registered_face_recognized":
        return _clamp_round(0.86 + min(0.10, _extract_float(event_map, "confidence", "score") * 0.08))
    if event_type == "target_long_present":
        stable_frames = _extract_int(lock_map, "stable_frames", "consecutive_frames", "frames")
        duration_ms = _extract_int(lock_map, "duration_ms", "stable_duration_ms", "visible_duration_ms")
        stability_bonus = min(0.08, stable_frames / 200)
        duration_bonus = min(0.05, duration_ms / 300_000)
        return _clamp_round(0.72 + stability_bonus + duration_bonus)
    if event_type == "target_long_absent":
        lost_duration_ms = _extract_int(lock_map, "lost_duration_ms", "absent_duration_ms", "missing_duration_ms")
        return _clamp_round(0.78 + min(0.08, lost_duration_ms / 120_000))
    if event_type == "user_device_interaction":
        return 0.82
    if event_type == "follow_success":
        score = _extract_float(follow_map, "score", "follow_score", "tracking_score")
        return _clamp_round(0.70 + min(0.08, max(0.0, score) * 0.08))
    if event_type == "follow_failed":
        score = _extract_float(follow_map, "score", "follow_score", "tracking_score")
        instability_bonus = min(0.08, max(0.0, 1.0 - score) * 0.08)
        return _clamp_round(0.84 + instability_bonus)
    if event_type == "user_feedback":
        return 0.96
    return 0.5


def _confidence_for(*, event_map: Mapping[str, object], follow_map: Mapping[str, object]) -> float | None:
    follow_confidence = _extract_float(follow_map, "score", "follow_score", "tracking_score", "confidence")
    if follow_confidence > 0:
        return follow_confidence
    event_confidence = _extract_float(event_map, "confidence", "score")
    if event_confidence > 0:
        return event_confidence
    objects = event_map.get("objects") or event_map.get("detections")
    if isinstance(objects, list):
        scores = [
            _extract_float(item, "confidence", "score")
            for item in objects
            if isinstance(item, Mapping)
        ]
        scores = [score for score in scores if score > 0]
        if scores:
            return max(scores)
    return None


def _tracking_provenance(
    *,
    lock_map: Mapping[str, object],
    follow_map: Mapping[str, object],
    event_reference: Mapping[str, object],
) -> dict[str, object]:
    return {
        "target_id": _first_text(lock_map, "target_id", "subject_id", "actor_id", "last_target_id"),
        "track_id": _first_text(lock_map, "track_id"),
        "stable_frames": _extract_int(lock_map, "stable_frames", "consecutive_frames", "frames"),
        "duration_ms": _extract_int(lock_map, "duration_ms", "stable_duration_ms", "visible_duration_ms"),
        "follow_score": _confidence_for(event_map={}, follow_map=follow_map),
        "source_event_id": event_reference.get("event_id") or "",
        "trace_id": event_reference.get("trace_id") or "",
    }


def _scene_provenance(
    *,
    event_map: Mapping[str, object],
    context_map: Mapping[str, object],
    event_reference: Mapping[str, object],
) -> dict[str, object]:
    return {
        "source_event_id": event_reference.get("event_id") or "",
        "trace_id": event_reference.get("trace_id") or "",
        "device_id": _first_text_from_maps(event_map, context_map, keys=("device_id", "camera_id", "sensor_id")),
    }


def _retention_policy(event_type: str) -> dict[str, object]:
    policies = {
        "registered_face_recognized": {
            "memory_type": "visual_identity_event",
            "memory_kind": "episodic",
            "retention": "identity_episode",
            "ttl_ms": 30 * MILLISECONDS_PER_DAY,
            "promotion_status": "candidate",
            "training_candidate": False,
        },
        "target_long_present": {
            "memory_type": "visual_event",
            "memory_kind": "episodic",
            "retention": "episode",
            "ttl_ms": 7 * MILLISECONDS_PER_DAY,
            "promotion_status": "not_promoted",
            "training_candidate": False,
        },
        "target_long_absent": {
            "memory_type": "visual_event",
            "memory_kind": "episodic",
            "retention": "episode",
            "ttl_ms": 7 * MILLISECONDS_PER_DAY,
            "promotion_status": "not_promoted",
            "training_candidate": False,
        },
        "user_device_interaction": {
            "memory_type": "visual_interaction",
            "memory_kind": "episodic",
            "retention": "episode",
            "ttl_ms": 14 * MILLISECONDS_PER_DAY,
            "promotion_status": "not_promoted",
            "training_candidate": False,
        },
        "follow_success": {
            "memory_type": "visual_tracking_outcome",
            "memory_kind": "episodic",
            "retention": "episode",
            "ttl_ms": 3 * MILLISECONDS_PER_DAY,
            "promotion_status": "not_promoted",
            "training_candidate": False,
        },
        "follow_failed": {
            "memory_type": "visual_tracking_outcome",
            "memory_kind": "procedural",
            "retention": "adjustment_candidate",
            "ttl_ms": 30 * MILLISECONDS_PER_DAY,
            "promotion_status": "candidate",
            "training_candidate": True,
        },
        "user_feedback": {
            "memory_type": "visual_feedback",
            "memory_kind": "training",
            "retention": "training_candidate",
            "ttl_ms": 90 * MILLISECONDS_PER_DAY,
            "promotion_status": "candidate",
            "training_candidate": True,
        },
    }
    return dict(policies[event_type])


def _event_reference(*, event_map: Mapping[str, object], event_type: str, source: str) -> dict[str, object]:
    raw_event_type = _first_text(event_map, "event_type", "type", "kind", "name") or event_type
    return {
        "protocol": "eiprotocol",
        "event_id": _first_text(event_map, "event_id", "id", "source_event_id"),
        "trace_id": _first_text(event_map, "trace_id", "request_id", "round_id"),
        "event_type": raw_event_type,
        "source": _first_text(event_map, "source") or source,
    }


def _content_for(
    *,
    event_type: str,
    summary: str,
    event_map: Mapping[str, object],
    lock_map: Mapping[str, object],
    follow_map: Mapping[str, object],
    context_map: Mapping[str, object],
) -> dict[str, object]:
    content: dict[str, object] = {
        "event_type": event_type,
        "summary": summary,
        "modality": "vision",
        "organ": "eye",
        "visual_event": _compact_event(event_map),
        "target_lock": _compact_target_lock(lock_map),
        "follow_score": _compact_follow_score(follow_map),
        "context": _compact_context(context_map),
    }
    feedback = _feedback_text(event_map, context_map)
    if feedback:
        content["user_feedback"] = feedback
    object_summary = _object_summary(event_map.get("objects") or event_map.get("detections"))
    if object_summary:
        content["objects"] = object_summary
    return content


def _compact_event(event_map: Mapping[str, object]) -> dict[str, object]:
    bulky_keys = {"image", "image_bytes", "frame", "frame_bytes", "raw_frame", "pixels", "jpeg", "png"}
    compact: dict[str, object] = {}
    for key, value in event_map.items():
        if key in bulky_keys or key in {"objects", "detections"}:
            continue
        compact[key] = value
    return compact


def _compact_target_lock(lock_map: Mapping[str, object]) -> dict[str, object]:
    return _compact_keys(
        lock_map,
        (
            "locked",
            "is_locked",
            "target_locked",
            "target_id",
            "track_id",
            "last_target_id",
            "label",
            "stable_frames",
            "consecutive_frames",
            "duration_ms",
            "stable_duration_ms",
            "visible_duration_ms",
            "lost_duration_ms",
            "absent_duration_ms",
            "missing_duration_ms",
        ),
    )


def _compact_follow_score(follow_map: Mapping[str, object]) -> dict[str, object]:
    return _compact_keys(
        follow_map,
        (
            "score",
            "follow_score",
            "tracking_score",
            "success",
            "status",
            "outcome",
            "duration_ms",
            "window_ms",
            "sample_count",
        ),
    )


def _compact_context(context_map: Mapping[str, object]) -> dict[str, object]:
    return _compact_keys(
        context_map,
        (
            "session_id",
            "actor_id",
            "user_id",
            "device_id",
            "conversation_id",
            "locale",
        ),
    )


def _compact_keys(mapping: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    return {key: mapping[key] for key in keys if key in mapping}


def _object_summary(raw_objects: object) -> list[dict[str, object]]:
    if not isinstance(raw_objects, list):
        return []
    objects: list[dict[str, object]] = []
    for raw in raw_objects[:5]:
        if not isinstance(raw, Mapping):
            continue
        item = _compact_keys(
            {str(key): _json_safe(value) for key, value in raw.items()},
            ("label", "name", "class", "confidence", "score", "stable_id", "track_id", "bbox"),
        )
        if item:
            objects.append(item)
    return objects


def _dedupe_key(
    *,
    event_type: str,
    session_id: str,
    target_id: str,
    summary: str,
    event_map: Mapping[str, object],
) -> str:
    session = _slug(session_id or "global")
    subject = _slug(target_id)
    if not subject:
        subject = _digest({"event_type": event_type, "summary": summary, "event": _compact_event(event_map)}, length=12)
    return f"visual_memory:{event_type}:{session}:{subject}"


def _privacy_for(event_type: str) -> dict[str, str]:
    if event_type in {"follow_failed", "user_feedback"}:
        return {
            "scope": "operational_feedback",
            "sensitivity": "operational",
            "allowed_use": "embodied_response",
        }
    if event_type == "registered_face_recognized":
        return {
            "scope": "situational_identity",
            "sensitivity": "personal",
            "allowed_use": "embodied_response",
        }
    return {
        "scope": "situational_awareness",
        "sensitivity": "environmental",
        "allowed_use": "embodied_response",
    }


def _decision_reason(event_type: str) -> str:
    return {
        "registered_face_recognized": "registered face recognition is important enough for episodic identity grounding",
        "target_long_present": "target remained stable long enough to avoid per-frame writes",
        "target_long_absent": "target left the view long enough to affect closed-loop behavior",
        "user_device_interaction": "user/device interaction changes embodied visual context",
        "follow_success": "tracking outcome is useful for closed-loop calibration",
        "follow_failed": "tracking failure can improve future follow behavior",
        "user_feedback": "explicit user feedback should be retained as a training candidate",
    }.get(event_type, "stable or important visual event")


def _scope(*, session_id: str, actor_id: str) -> dict[str, str]:
    del session_id
    scope: dict[str, str] = {"tenant_id": "default"}
    if actor_id:
        scope["user_id"] = actor_id
    return scope


def _title_for(event_type: str) -> str:
    return {
        "registered_face_recognized": "Registered face recognition",
        "target_long_present": "Stable visual target presence",
        "target_long_absent": "Visual target left view",
        "user_device_interaction": "User/device visual interaction",
        "follow_success": "Visual follow success",
        "follow_failed": "Visual follow failure",
        "user_feedback": "Visual user feedback",
    }.get(event_type, "Visual memory event")


def _tags_for(*, event_type: str, memory_type: str, retention: str, content: Mapping[str, object]) -> list[str]:
    labels = []
    for item in content.get("objects", []):
        if isinstance(item, Mapping):
            labels.append(item.get("label") or item.get("name") or item.get("class"))
    tags = ["visual_memory", event_type, memory_type, "vision", "eye", retention, *labels]
    if retention in {"adjustment_candidate", "training_candidate"}:
        tags.append("training_candidate")
    return _unique_texts(tags)


def _feedback_text(*mappings: Mapping[str, object]) -> str:
    return _first_text_from_maps(
        *mappings,
        keys=("feedback", "user_feedback", "text", "comment", "utterance"),
    )


def _timestamp_ms(*mappings: Mapping[str, object]) -> int:
    for mapping in mappings:
        value = _first_value(mapping, "timestamp_ms", "ts_ms")
        if value is None:
            seconds = _first_value(mapping, "timestamp", "ts", "time")
            if seconds is not None:
                try:
                    parsed = float(str(seconds))
                except ValueError:
                    parsed = 0.0
                if parsed > 0:
                    return int(parsed * 1000 if parsed < 10_000_000_000 else parsed)
            continue
        try:
            return int(float(str(value)))
        except ValueError:
            continue
    return int(time() * 1000)


def _to_plain_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {str(key): _json_safe(item) for key, item in asdict(value).items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return {str(key): _json_safe(item) for key, item in mapped.items()}
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, Mapping):
        return {str(key): _json_safe(item) for key, item in attrs.items()}
    return {}


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, str | int | float | bool) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return 0.0
        return value
    return str(value)


def _first_text(mapping: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_text_from_maps(*mappings: Mapping[str, object], keys: tuple[str, ...]) -> str:
    for mapping in mappings:
        value = _first_text(mapping, *keys)
        if value:
            return value
    return ""


def _first_value(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _extract_int(mapping: Mapping[str, object], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(float(str(value)))
        except ValueError:
            continue
    return 0


def _extract_float(mapping: Mapping[str, object], *keys: str) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            parsed = float(str(value))
        except ValueError:
            continue
        return parsed if math.isfinite(parsed) else 0.0
    return 0.0


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = _normalize_token(str(value))
    if normalized in {"true", "1", "yes", "y", "locked", "success", "ok"}:
        return True
    if normalized in {"false", "0", "no", "n", "unlocked", "failed", "lost"}:
        return False
    return None


def _normalize_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _slug(value: object) -> str:
    raw = str(value or "").strip()
    output: list[str] = []
    last_dash = False
    for char in raw:
        if char.isalnum() or char in {"_", "-"}:
            output.append(char)
            last_dash = False
        elif not last_dash:
            output.append("-")
            last_dash = True
    return "".join(output).strip("-")[:80]


def _unique_texts(values: list[object]) -> list[str]:
    unique: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _clamp_round(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _digest(payload: object, *, length: int) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]


__all__ = [
    "CONTRACT_VERSION",
    "SOURCE",
    "TRACE_SCHEMA",
    "VisualMemoryPolicy",
    "build_visual_memory_candidate",
]
