"""Software-only realtime vision event simulator.

This module intentionally has no honjia device dependency. It turns sequential
frame detections into stable tracks, lifecycle events, attention, and
eiprotocol-friendly observation content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping


@dataclass(slots=True)
class _Track:
    track_id: str
    label: str
    bbox: dict[str, float]
    confidence: float
    first_seen_frame: str
    last_seen_frame: str
    last_observed_at: str
    missing_frames: int = 0
    temporal_state: str = "appeared"
    stationary_frames: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class RealtimeVisionSimulator:
    """Track detections across frames and emit realtime scene events."""

    def __init__(
        self,
        *,
        match_distance: float = 0.35,
        move_threshold: float = 0.12,
        approach_area_delta: float = 0.04,
        max_missing_frames: int = 1,
        attention_switch_margin: float = 0.20,
        attention_switch_cooldown_frames: int = 2,
    ) -> None:
        self.match_distance = float(match_distance)
        self.move_threshold = float(move_threshold)
        self.approach_area_delta = max(0.0, float(approach_area_delta))
        self.max_missing_frames = int(max_missing_frames)
        self.attention_switch_margin = max(0.0, float(attention_switch_margin))
        self.attention_switch_cooldown_frames = max(1, int(attention_switch_cooldown_frames))
        self._tracks: dict[str, _Track] = {}
        self._next_ids: dict[str, int] = {}
        self._attention_track_id = ""
        self._attention_candidate_track_id = ""
        self._attention_candidate_frames = 0

    def update(
        self,
        *,
        frame_id: str,
        observed_at: str,
        detections: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized = [_normalize_detection(item) for item in detections]
        normalized = [item for item in normalized if item is not None]
        events: list[dict[str, Any]] = []
        matches = self._match_detections(normalized)
        matched_detection_indexes = {detection_index for detection_index, _ in matches}
        matched_track_ids = {track_id for _, track_id in matches}

        for detection_index, track_id in matches:
            detection = normalized[detection_index]
            track = self._tracks[track_id]
            previous_bbox = dict(track.bbox)
            previous_region = _region(_center(previous_bbox))
            distance = _distance(_center(previous_bbox), _center(detection["bbox"]))
            area_delta = _area(detection["bbox"]) - _area(previous_bbox)
            track.bbox = dict(detection["bbox"])
            track.confidence = float(detection["confidence"])
            track.extras = dict(detection.get("extras", {}))
            track.last_seen_frame = frame_id
            track.last_observed_at = observed_at
            track.missing_frames = 0
            current_region = _region(_center(track.bbox))
            temporal_state = _matched_temporal_state(
                distance=distance,
                area_delta=area_delta,
                previous_region=previous_region,
                current_region=current_region,
                move_threshold=self.move_threshold,
                approach_area_delta=self.approach_area_delta,
            )
            track.temporal_state = temporal_state
            track.stationary_frames = track.stationary_frames + 1 if temporal_state == "stationary" else 0
            if temporal_state in {"moving", "approaching"}:
                events.append(
                    _event(
                        scene_id="",
                        event_type="approaching" if temporal_state == "approaching" else "moved",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        confidence=track.confidence,
                        from_region=previous_region,
                        to_region=current_region,
                        distance=distance,
                        temporal_state=temporal_state,
                        area_delta=area_delta,
                    )
                )

        for detection_index, detection in enumerate(normalized):
            if detection_index in matched_detection_indexes:
                continue
            track = self._new_track(detection, frame_id=frame_id, observed_at=observed_at)
            track.temporal_state = "appeared"
            self._tracks[track.track_id] = track
            matched_track_ids.add(track.track_id)
            events.append(
                _event(
                    scene_id="",
                    event_type="appeared",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=track,
                    confidence=track.confidence,
                    from_region="",
                    to_region=_region(_center(track.bbox)),
                    distance=0.0,
                    temporal_state="appeared",
                )
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                track.temporal_state = "disappeared"
                events.append(
                    _event(
                        scene_id="",
                        event_type="disappeared",
                        observed_at=observed_at,
                        frame_id=frame_id,
                        track=track,
                        confidence=track.confidence,
                        from_region=_region(_center(track.bbox)),
                        to_region="",
                        distance=0.0,
                        temporal_state="disappeared",
                    )
                )
                del self._tracks[track_id]

        active_tracks = sorted(
            [track for track in self._tracks.values() if track.missing_frames == 0],
            key=lambda item: item.track_id,
        )
        previous_attention_track_id = self._attention_track_id
        attention = self._attention_track(active_tracks, current_track_id=previous_attention_track_id)
        self._attention_track_id = attention.track_id if attention is not None else ""
        if attention is not None and previous_attention_track_id and attention.track_id != previous_attention_track_id:
            events.append(
                _event(
                    scene_id="",
                    event_type="attention_changed",
                    observed_at=observed_at,
                    frame_id=frame_id,
                    track=attention,
                    confidence=attention.confidence,
                    from_region="",
                    to_region=_region(_center(attention.bbox)),
                    distance=0.0,
                    temporal_state="attention_changed",
                )
            )

        objects = [_track_object(track) for track in active_tracks]
        relationships = _relationships(objects)
        summary = _summary(objects, events)
        event_summary = _event_summary(objects, events)
        scene_id = _scene_id(frame_id, observed_at, objects, relationships)
        attention_object = _attention_object(attention)
        scene_snapshot = {
            "sceneId": scene_id,
            "observedAt": observed_at,
            "frameId": frame_id,
            "objects": objects,
            "relationships": relationships,
            "attention": attention_object,
            "stableTarget": dict(attention_object),
            "summary": summary,
            "eventSummary": event_summary,
            "metadata": {
                "frameId": frame_id,
                "realtime": True,
                "simulator": "software",
                "trackCount": len(objects),
                "eventCount": len(events),
                "lastEventType": str(events[-1]["eventType"]) if events else "",
                "stableTargetTrackId": str(attention_object.get("trackId", "")),
                "stableTargetCandidateTrackId": self._attention_candidate_track_id,
                "attentionSwitchCooldownFrames": self.attention_switch_cooldown_frames,
            },
        }
        for event in events:
            event["sceneId"] = scene_id
            event["eventId"] = f"{scene_id}:{event['eventType']}:{event['subject']['trackId']}"
        return {
            "frameId": frame_id,
            "observedAt": observed_at,
            "events": events,
            "sceneSnapshot": scene_snapshot,
            "sceneGraphSummary": summary,
            "eventSummary": event_summary,
        }

    def replay(self, frames: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Replay a deterministic sequence of detection frames through the tracker."""

        snapshots: list[dict[str, Any]] = []
        for index, frame in enumerate(frames, start=1):
            detections = frame.get("detections", [])
            snapshots.append(
                self.update(
                    frame_id=str(frame.get("frame_id", frame.get("frameId", f"frame-{index:03d}"))),
                    observed_at=str(frame.get("observed_at", frame.get("observedAt", ""))),
                    detections=[dict(item) for item in detections] if isinstance(detections, list) else [],
                )
            )
        return snapshots

    def _match_detections(self, detections: list[dict[str, Any]]) -> list[tuple[int, str]]:
        candidates: list[tuple[float, int, str]] = []
        active_tracks = [track for track in self._tracks.values() if track.missing_frames <= self.max_missing_frames]
        for detection_index, detection in enumerate(detections):
            for track in active_tracks:
                if detection["label"] != track.label:
                    continue
                distance = _distance(_center(detection["bbox"]), _center(track.bbox))
                if distance <= self.match_distance:
                    candidates.append((distance, detection_index, track.track_id))

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

    def _new_track(self, detection: dict[str, Any], *, frame_id: str, observed_at: str) -> _Track:
        label = str(detection["label"])
        next_id = self._next_ids.get(label, 0) + 1
        self._next_ids[label] = next_id
        return _Track(
            track_id=f"{label}-{next_id:03d}",
            label=label,
            bbox=dict(detection["bbox"]),
            confidence=float(detection["confidence"]),
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
            last_observed_at=observed_at,
            extras=dict(detection.get("extras", {})),
        )

    def _attention_track(self, tracks: list[_Track], *, current_track_id: str) -> _Track | None:
        if not tracks:
            self._attention_candidate_track_id = ""
            self._attention_candidate_frames = 0
            return None

        best = max(tracks, key=lambda track: (_track_salience(track), track.confidence, track.track_id))
        current = next((track for track in tracks if track.track_id == current_track_id), None)
        if current is None:
            self._attention_candidate_track_id = ""
            self._attention_candidate_frames = 0
            return best

        current_salience = _track_salience(current)
        best_salience = _track_salience(best)
        should_switch = best.track_id != current.track_id and best_salience > current_salience * (
            1.0 + self.attention_switch_margin
        )
        if not should_switch:
            self._attention_candidate_track_id = ""
            self._attention_candidate_frames = 0
            return current

        if self._attention_candidate_track_id == best.track_id:
            self._attention_candidate_frames += 1
        else:
            self._attention_candidate_track_id = best.track_id
            self._attention_candidate_frames = 1

        if self._attention_candidate_frames >= self.attention_switch_cooldown_frames:
            self._attention_candidate_track_id = ""
            self._attention_candidate_frames = 0
            return best
        return current


def to_eiprotocol_scene_content(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Map a simulator result to VisionSceneObservation content."""

    scene = _scene_snapshot(snapshot)
    objects = [dict(item) for item in _dict_list(scene.get("objects"))]
    relationships = [dict(item) for item in _dict_list(scene.get("relationships"))]
    content = {
        "sceneId": str(scene.get("sceneId", "")),
        "observedAt": str(scene.get("observedAt", "")),
        "summary": str(snapshot.get("sceneGraphSummary") or scene.get("summary") or ""),
        "objects": objects,
        "relationships": relationships,
        "environment": {"source": "realtime_vision_simulator"},
        "imageUrl": "",
        "metadata": dict(scene.get("metadata") if isinstance(scene.get("metadata"), Mapping) else {}),
        "attention": dict(scene.get("attention") if isinstance(scene.get("attention"), Mapping) else {}),
        "stableTarget": dict(scene.get("stableTarget") if isinstance(scene.get("stableTarget"), Mapping) else {}),
        "eventSummary": str(snapshot.get("eventSummary") or scene.get("eventSummary") or ""),
    }
    for key, value in (
        ("clipLabels", _first_non_empty(scene.get("clipLabels"), _aggregate_items(objects, "clipLabels"))),
        ("semanticLabels", _first_non_empty(scene.get("semanticLabels"), _aggregate_items(objects, "semanticLabels"))),
        ("depth", _first_non_empty(scene.get("depth"), _first_mapping_from_items(objects, "depth"))),
        ("distance", _first_non_empty(scene.get("distance"), _first_mapping_from_items(objects, "distance"))),
        (
            "trackingDiagnostics",
            _first_non_empty(scene.get("trackingDiagnostics"), _first_mapping_from_items(objects, "trackingDiagnostics")),
        ),
    ):
        if value:
            content[key] = value
    if objects:
        content["sceneGraph"] = {
            "nodes": [{"id": item.get("trackId"), "label": item.get("label")} for item in objects],
            "edges": relationships,
        }
        content["sceneGraphProvenance"] = {"builder": "realtime_vision_simulator"}
    return content


def to_eiprotocol_event_contents(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map simulator events to VisionEventObservation content dictionaries."""

    contents: list[dict[str, Any]] = []
    for event in _dict_list(snapshot.get("events")):
        contents.append(
            {
                "eventId": str(event.get("eventId", "")),
                "eventType": str(event.get("eventType", "")),
                "observedAt": str(event.get("observedAt", "")),
                "sceneId": str(event.get("sceneId", "")),
                "subject": dict(event.get("subject") if isinstance(event.get("subject"), Mapping) else {}),
                "confidence": event.get("confidence"),
                "pose": dict(event.get("pose") if isinstance(event.get("pose"), Mapping) else {}),
                "clipLabels": [dict(item) for item in _dict_list(event.get("clipLabels"))],
                "semanticLabels": [dict(item) for item in _dict_list(event.get("semanticLabels"))],
                "depth": dict(event.get("depth") if isinstance(event.get("depth"), Mapping) else {}),
                "distance": dict(event.get("distance") if isinstance(event.get("distance"), Mapping) else {}),
                "trackingDiagnostics": dict(
                    event.get("trackingDiagnostics") if isinstance(event.get("trackingDiagnostics"), Mapping) else {}
                ),
                "sceneGraphProvenance": dict(
                    event.get("sceneGraphProvenance") if isinstance(event.get("sceneGraphProvenance"), Mapping) else {}
                ),
                "details": dict(event.get("details") if isinstance(event.get("details"), Mapping) else {}),
                "metadata": dict(event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}),
            }
        )
    return contents


def _normalize_detection(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    label = str(raw.get("label") or raw.get("name") or raw.get("class") or "").strip()
    bbox = _normalize_bbox(
        raw.get("bbox"),
        width=_raw_value(raw, "width", "frame_width", "image_width"),
        height=_raw_value(raw, "height", "frame_height", "image_height"),
        format_hint=_bbox_format(raw),
    )
    if not label or bbox is None:
        return None
    return {
        "label": label,
        "bbox": bbox,
        "confidence": _coerce_float(raw.get("confidence", raw.get("score", 0.0))),
        "extras": _detection_extras(raw),
    }


def _normalize_bbox(raw: Any, *, width: Any = None, height: Any = None, format_hint: str = "") -> dict[str, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x_min, y_min, x_max, y_max = _sequence_bbox_values(raw, format_hint=format_hint)
        except (TypeError, ValueError):
            return None
        frame_width = _positive_float(width)
        frame_height = _positive_float(height)
        if frame_width and max(abs(x_min), abs(x_max)) > 1.0:
            x_min /= frame_width
            x_max /= frame_width
        if frame_height and max(abs(y_min), abs(y_max)) > 1.0:
            y_min /= frame_height
            y_max /= frame_height
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
            return {"x_min": round(x_min, 4), "y_min": round(y_min, 4), "x_max": round(x_max, 4), "y_max": round(y_max, 4)}
        if "x1" in raw and "y1" in raw and "x2" in raw and "y2" in raw:
            x_min = _clip01(float(raw.get("x1", 0.0)))
            y_min = _clip01(float(raw.get("y1", 0.0)))
            x_max = _clip01(float(raw.get("x2", 0.0)))
            y_max = _clip01(float(raw.get("y2", 0.0)))
            if x_max < x_min:
                x_min, x_max = x_max, x_min
            if y_max < y_min:
                y_min, y_max = y_max, y_min
            return {"x_min": round(x_min, 4), "y_min": round(y_min, 4), "x_max": round(x_max, 4), "y_max": round(y_max, 4)}
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


def _event(
    *,
    scene_id: str,
    event_type: str,
    observed_at: str,
    frame_id: str,
    track: _Track,
    confidence: float,
    from_region: str,
    to_region: str,
    distance: float,
    temporal_state: str,
    area_delta: float = 0.0,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "fromRegion": from_region,
        "toRegion": to_region,
        "distance": round(float(distance), 3),
        "temporalState": temporal_state,
    }
    if area_delta:
        details["areaDelta"] = round(float(area_delta), 4)
    payload = {
        "eventId": f"{scene_id}:{event_type}:{track.track_id}" if scene_id else "",
        "eventType": event_type,
        "observedAt": observed_at,
        "sceneId": scene_id,
        "subject": {"trackId": track.track_id, "label": track.label},
        "confidence": round(float(confidence), 3),
        "details": details,
        "metadata": {"frameId": frame_id},
    }
    for key in ("pose", "clipLabels", "semanticLabels", "depth", "distance", "trackingDiagnostics"):
        value = track.extras.get(key)
        if value:
            payload[key] = value
    if track.extras:
        payload["sceneGraphProvenance"] = {"builder": "realtime_vision_simulator"}
    return payload


def _track_object(track: _Track) -> dict[str, Any]:
    center = _center(track.bbox)
    payload = {
        "trackId": track.track_id,
        "label": track.label,
        "confidence": round(float(track.confidence), 3),
        "bbox": dict(track.bbox),
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "region": _region(center),
        "missingFrames": track.missing_frames,
        "temporalState": track.temporal_state,
        "stationaryFrames": track.stationary_frames,
    }
    payload.update({key: value for key, value in track.extras.items() if value})
    return payload


def _relationships(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for subject in objects:
        for obj in objects:
            if subject["trackId"] == obj["trackId"]:
                continue
            subject_center = _object_center(subject)
            object_center = _object_center(obj)
            dx = object_center[0] - subject_center[0]
            dy = object_center[1] - subject_center[1]
            for relation in _relation_types(dx, dy):
                relations.append(
                    {
                        "subjectId": subject["trackId"],
                        "subjectLabel": subject["label"],
                        "relation": relation,
                        "objectId": obj["trackId"],
                        "objectLabel": obj["label"],
                    }
                )
    return sorted(relations, key=lambda item: (item["subjectId"], item["relation"], item["objectId"]))


def _relation_types(dx: float, dy: float) -> list[str]:
    relations: list[str] = []
    if dx > 0.18:
        relations.append("left_of")
    if dx < -0.18:
        relations.append("right_of")
    if dy > 0.18:
        relations.append("above")
    if dy < -0.18:
        relations.append("below")
    if math.hypot(dx, dy) <= 0.6:
        relations.append("near")
    return relations


def _summary(objects: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    if objects:
        labels: dict[str, int] = {}
        for obj in objects:
            label = str(obj.get("label", ""))
            labels[label] = labels.get(label, 0) + 1
        observed = ", ".join(
            f"{count} {label}" if count > 1 else label for label, count in sorted(labels.items())
        )
    else:
        observed = "empty scene"
    event_types = sorted({str(event.get("eventType")) for event in events if event.get("eventType")})
    temporal_states = sorted({str(obj.get("temporalState")) for obj in objects if obj.get("temporalState")})
    state_summary = ", ".join(temporal_states) if temporal_states else "none"
    if event_types:
        return f"Observed {observed}; realtime events: {', '.join(event_types)}; temporal states: {state_summary}"
    return f"Observed {observed}; realtime events: none; temporal states: {state_summary}"


def _event_summary(objects: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    if events:
        parts = [
            f"{event['eventType']} {event['subject']['trackId']}"
            for event in events
            if isinstance(event.get("subject"), Mapping)
        ]
        return "; ".join(parts)
    states = [
        f"{item['temporalState']} {item['trackId']}"
        for item in objects
        if item.get("temporalState")
    ]
    return "; ".join(states) if states else "no temporal changes"


def _track_salience(track: _Track) -> float:
    return _area(track.bbox) * max(track.confidence, 0.0)


def _attention_object(track: _Track | None) -> dict[str, Any]:
    if track is None:
        return {}
    return {
        "trackId": track.track_id,
        "label": track.label,
        "confidence": round(float(track.confidence), 3),
        "region": _region(_center(track.bbox)),
    }


def _scene_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    scene = snapshot.get("sceneSnapshot")
    return scene if isinstance(scene, Mapping) else snapshot


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _detection_extras(raw: Mapping[str, Any]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    source_track_id = _source_track_id(raw)
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


def _source_track_id(raw: Mapping[str, Any]) -> str:
    for key in ("trackId", "track_id", "id", "stable_id"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def _aggregate_items(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        raw = item.get(key)
        if not isinstance(raw, list):
            continue
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            payload = dict(entry)
            label = str(payload.get("label") or payload.get("name") or payload)
            if label in seen:
                continue
            seen.add(label)
            aggregated.append(payload)
    return aggregated


def _first_mapping_from_items(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for item in items:
        value = item.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


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


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _scene_id(frame_id: str, observed_at: str, objects: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> str:
    payload = {
        "frameId": frame_id,
        "observedAt": observed_at,
        "objects": [
            {"trackId": item["trackId"], "label": item["label"], "region": item["region"]}
            for item in objects
        ],
        "relationships": relationships,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"scene_rt_{digest}"


def _center(bbox: Mapping[str, float]) -> tuple[float, float]:
    return ((float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0)


def _object_center(obj: Mapping[str, Any]) -> tuple[float, float]:
    center = obj.get("center")
    if isinstance(center, Mapping):
        return (_coerce_float(center.get("x")), _coerce_float(center.get("y")))
    bbox = obj.get("bbox")
    if isinstance(bbox, Mapping):
        return _center(bbox)  # type: ignore[arg-type]
    return (0.0, 0.0)


def _region(center: tuple[float, float]) -> str:
    x_name = "left" if center[0] < 0.33 else "center" if center[0] < 0.66 else "right"
    y_name = "top" if center[1] < 0.25 else "middle" if center[1] < 0.75 else "bottom"
    return f"{x_name}_{y_name}"


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _area(bbox: Mapping[str, float]) -> float:
    return max(0.0, float(bbox["x_max"]) - float(bbox["x_min"])) * max(0.0, float(bbox["y_max"]) - float(bbox["y_min"]))


def _matched_temporal_state(
    *,
    distance: float,
    area_delta: float,
    previous_region: str,
    current_region: str,
    move_threshold: float,
    approach_area_delta: float,
) -> str:
    if area_delta >= approach_area_delta and distance < move_threshold and previous_region == current_region:
        return "approaching"
    if distance >= move_threshold or previous_region != current_region:
        return "moving"
    return "stationary"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "RealtimeVisionSimulator",
    "to_eiprotocol_event_contents",
    "to_eiprotocol_scene_content",
]
