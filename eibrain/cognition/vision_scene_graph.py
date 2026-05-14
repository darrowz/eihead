"""Build a lightweight spatial scene graph from visual detections and tracks.

The module is intentionally model-free: it accepts detector/tracker-shaped
payloads, normalizes their bounding boxes, and derives spatial semantics with
deterministic heuristics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


_PERSON_LABELS = {"person", "human", "face", "body", "head"}
_DEVICE_LABELS = {"phone", "mobile", "smartphone", "tablet", "laptop", "monitor", "screen", "device"}
_HAND_KEYPOINTS = {"left_wrist", "right_wrist", "left_hand", "right_hand", "hand", "wrist"}
_EMPTY_REGIONS: dict[str, list[dict[str, Any]]] = {
    "left": [],
    "center": [],
    "right": [],
    "near": [],
    "far": [],
}
_FRAME_KEYS = {
    "detections",
    "objects",
    "tracks",
    "previousScene",
    "previous_scene",
}


def build_vision_scene_graph(
    detections: list[Mapping[str, Any]] | None = None,
    *,
    tracks: list[Mapping[str, Any]] | None = None,
    frame_metadata: Mapping[str, Any] | None = None,
    previous_scene: Mapping[str, Any] | None = None,
    near_area_threshold: float = 0.10,
    near_relation_gap: float = 0.12,
    motion_threshold: float = 0.08,
    approach_area_delta: float = 0.04,
) -> dict[str, Any]:
    """Build people, objects, relations, regions, and summary from one frame."""

    frame = _frame_metadata(frame_metadata)
    objects = _build_objects(
        detections=detections,
        tracks=tracks,
        frame=frame,
        near_area_threshold=near_area_threshold,
    )
    temporal = _temporal_analysis(
        objects=objects,
        previous_scene=previous_scene,
        motion_threshold=motion_threshold,
        approach_area_delta=approach_area_delta,
    )
    people = [dict(item) for item in objects if item["kind"] == "person"]
    relations = _build_relations(objects, near_relation_gap=near_relation_gap)
    lightweight_events = _build_lightweight_events(objects, relations)
    events = [dict(item) for item in temporal["events"]] + lightweight_events
    regions = _build_regions(objects)
    dominant_target = _dominant_target(objects)
    safety = {"pathClear": None, "nearObstacle": None}
    summary = _summary(objects, relations, dominant_target, safety, temporal)
    scene_id = _scene_id(frame, objects, relations)
    event_summary = _combined_event_summary(temporal, lightweight_events)

    return {
        "sceneId": scene_id,
        "observedAt": frame["observedAt"],
        "frameId": frame["frameId"],
        "people": people,
        "objects": objects,
        "relations": relations,
        "relationships": relations,
        "regions": regions,
        "dominant_target": dominant_target,
        "summary": summary,
        "events": events,
        "temporal": temporal,
        "event_summary": event_summary,
        "safety": safety,
        "metadata": _public_metadata(frame),
    }


def build_scene_graph(
    payload: Mapping[str, Any] | None = None,
    *,
    detections: list[Mapping[str, Any]] | None = None,
    tracks: list[Mapping[str, Any]] | None = None,
    frame_metadata: Mapping[str, Any] | None = None,
    previous_scene: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Payload-style wrapper for callers that already hold a vision frame dict."""

    if payload is None:
        return build_vision_scene_graph(
            detections=detections,
            tracks=tracks,
            frame_metadata=frame_metadata,
            previous_scene=previous_scene,
        )

    metadata = {key: value for key, value in payload.items() if key not in _FRAME_KEYS}
    if frame_metadata is not None:
        metadata.update(dict(frame_metadata))
    return build_vision_scene_graph(
        detections=detections if detections is not None else _mapping_list(payload.get("detections", payload.get("objects"))),
        tracks=tracks if tracks is not None else _mapping_list(payload.get("tracks")),
        frame_metadata=metadata,
        previous_scene=previous_scene or _first_mapping(payload.get("previousScene"), payload.get("previous_scene")),
    )


def to_eiprotocol_scene_content(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Map a scene graph to generic VisionSceneObservation content."""

    metadata = dict(scene.get("metadata") if isinstance(scene.get("metadata"), Mapping) else {})
    objects = [dict(item) for item in _mapping_list(scene.get("objects"))]
    relationships = [dict(item) for item in _mapping_list(scene.get("relations", scene.get("relationships")))]
    content = {
        "sceneId": str(scene.get("sceneId", "")),
        "observedAt": str(scene.get("observedAt", "")),
        "summary": str(scene.get("summary", "")),
        "objects": objects,
        "relationships": relationships,
        "environment": {"source": "vision_scene_graph"},
        "imageUrl": str(metadata.get("imageUrl", "")),
        "metadata": metadata,
    }
    clip_labels = _aggregate_label_annotations(objects, "clip_labels")
    semantic_labels = _aggregate_semantic_labels(objects)
    depth = _first_mapping_or_depth(objects, "depth")
    distance = _first_mapping_or_depth(objects, "distance")
    tracking_diagnostics = _first_mapping_or_depth(objects, "tracking_diagnostics")
    if clip_labels:
        content["clipLabels"] = clip_labels
    if semantic_labels:
        content["semanticLabels"] = [{"label": label} for label in semantic_labels]
    if depth:
        content["depth"] = depth
    if distance:
        content["distance"] = distance
    if tracking_diagnostics:
        content["trackingDiagnostics"] = tracking_diagnostics
    if objects:
        content["sceneGraph"] = {
            "nodes": [{"id": item.get("id"), "label": item.get("label")} for item in objects],
            "edges": relationships,
        }
        content["sceneGraphProvenance"] = {"builder": "vision_scene_graph"}
    if isinstance(scene.get("temporal"), Mapping):
        content["temporal"] = dict(scene["temporal"])  # type: ignore[index]
    if isinstance(scene.get("events"), list):
        content["events"] = [dict(item) for item in _mapping_list(scene.get("events"))]
    if scene.get("event_summary"):
        content["eventSummary"] = str(scene.get("event_summary"))
    return content


def _build_objects(
    *,
    detections: list[Mapping[str, Any]] | None,
    tracks: list[Mapping[str, Any]] | None,
    frame: Mapping[str, Any],
    near_area_threshold: float,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    objects_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tracks or []):
        item = _normalize_object(
            raw,
            source="track",
            index=index,
            frame=frame,
            near_area_threshold=near_area_threshold,
            seen_ids=seen_ids,
        )
        if item is not None:
            objects.append(item)
            objects_by_id[str(item["id"])] = item
    for index, raw in enumerate(detections or []):
        object_id = _raw_object_id(raw)
        if object_id and object_id in objects_by_id:
            item = _normalize_object(
                raw,
                source="detection",
                index=index,
                frame=frame,
                near_area_threshold=near_area_threshold,
                seen_ids=set(),
            )
            if item is not None:
                _merge_object_extras(objects_by_id[object_id], item)
            continue
        item = _normalize_object(
            raw,
            source="detection",
            index=index,
            frame=frame,
            near_area_threshold=near_area_threshold,
            seen_ids=seen_ids,
        )
        if item is not None:
            objects.append(item)
            objects_by_id[str(item["id"])] = item
    return sorted(objects, key=lambda item: (str(item["label"]), str(item["id"])))


def _raw_object_id(raw: Mapping[str, Any]) -> str:
    return _first_text(
        raw.get("track_id"),
        raw.get("trackId"),
        raw.get("sourceTrackId"),
        raw.get("source_track_id"),
        raw.get("object_id"),
        raw.get("objectId"),
        raw.get("stable_id"),
        raw.get("stableId"),
        raw.get("id"),
    )


def _merge_object_extras(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in {"id", "trackId", "label", "kind"} or value in (None, "", [], {}):
            continue
        if key not in target or target.get(key) in (None, "", [], {}):
            target[key] = value
        elif isinstance(target.get(key), list) and isinstance(value, list):
            target[key] = [*target[key], *[item for item in value if item not in target[key]]]


def _normalize_object(
    raw: Mapping[str, Any],
    *,
    source: str,
    index: int,
    frame: Mapping[str, Any],
    near_area_threshold: float,
    seen_ids: set[str],
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    label = _first_text(raw.get("label"), raw.get("name"), raw.get("class"))
    bbox = _normalize_bbox(
        raw.get("bbox"),
        width=frame.get("width"),
        height=frame.get("height"),
        format_hint=_bbox_format(raw),
    )
    if not label or bbox is None:
        return None

    confidence = round(_coerce_float(raw.get("confidence", raw.get("score", 0.0))), 3)
    center = _center(bbox)
    area = _area(bbox)
    horizontal = _horizontal_region(center[0])
    vertical = _vertical_region(center[1])
    kind = "person" if _is_person(label) else "object"
    object_id = _object_id(raw, label=label, center=center, source=source, index=index, seen_ids=seen_ids)
    raw_depth = _raw_value(raw, "depth")
    raw_distance = _raw_value(raw, "distance")
    depth_m = _structured_depth_m(raw) or _optional_float(_raw_value(raw, "depth_m", "distance_m", "z_m"))
    distance_band = _first_text(_raw_value(raw, "distance_band", "depth_band"))
    if not distance_band:
        distance_band = _distance_band(depth_m=depth_m, area=area, near_area_threshold=near_area_threshold)
    depth = "near" if distance_band == "near" else "far"
    source_name = _first_text(raw.get("source"), raw.get("provider"), source)
    model_id = _first_text(raw.get("model_id"), raw.get("modelId"), raw.get("model"))
    provenance = _normalize_provenance(raw, source=source_name, model_id=model_id)
    item = {
        "id": object_id,
        "trackId": object_id,
        "label": label,
        "kind": kind,
        "confidence": confidence,
        "bbox": bbox,
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)},
        "region": f"{horizontal}_{vertical}",
        "horizontal_region": horizontal,
        "vertical_region": vertical,
        "depth": depth,
        "distance_band": distance_band,
        "area": round(area, 4),
        "source": source_name,
        "model_id": model_id,
        "provenance": provenance,
    }
    if depth_m is not None:
        item["depth_m"] = depth_m
    if isinstance(raw_depth, Mapping):
        item["depth_payload"] = dict(raw_depth)
    if isinstance(raw_distance, Mapping):
        item["distance"] = dict(raw_distance)
    pose = _normalize_pose(_raw_value(raw, "pose"), frame=frame)
    if pose:
        item["pose"] = pose
        item["keypoints"] = list(pose["keypoints"])
    clip_labels = _normalize_label_annotations(_raw_value(raw, "clip_labels", "clipLabels"))
    if clip_labels:
        item["clip_labels"] = clip_labels
    semantic_labels = _normalize_semantic_labels(_raw_value(raw, "semantic_labels", "semanticLabels"))
    if semantic_labels:
        item["semantic_labels"] = semantic_labels
    tracking_diagnostics = _raw_value(raw, "tracking_diagnostics", "trackingDiagnostics")
    if isinstance(tracking_diagnostics, Mapping):
        item["tracking_diagnostics"] = dict(tracking_diagnostics)
    if kind == "person":
        item["looking_at_device"] = _optional_bool(_raw_value(raw, "looking_at_device", "lookingAtDevice"))
    return item


def _build_relations(objects: list[dict[str, Any]], *, near_relation_gap: float) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for subject in objects:
        for obj in objects:
            if subject["id"] == obj["id"]:
                continue
            subject_center = _object_center(subject)
            object_center = _object_center(obj)
            dx = object_center[0] - subject_center[0]
            dy = object_center[1] - subject_center[1]
            for relation in _relation_types(subject["bbox"], obj["bbox"], dx, dy, near_relation_gap=near_relation_gap):
                relations.append(
                    {
                        "subjectId": subject["id"],
                        "subjectLabel": subject["label"],
                        "relation": relation,
                        "objectId": obj["id"],
                        "objectLabel": obj["label"],
                    }
                )
            if subject.get("kind") == "person" and _hand_near_object(subject, obj, near_relation_gap=near_relation_gap):
                relations.append(
                    {
                        "subjectId": subject["id"],
                        "subjectLabel": subject["label"],
                        "relation": "hand_near_object",
                        "objectId": obj["id"],
                        "objectLabel": obj["label"],
                    }
                )
    return sorted(
        relations,
        key=lambda item: (str(item["subjectId"]), str(item["relation"]), str(item["objectId"])),
    )


def _relation_types(
    subject_bbox: Mapping[str, float],
    object_bbox: Mapping[str, float],
    dx: float,
    dy: float,
    *,
    near_relation_gap: float,
) -> list[str]:
    relations: list[str] = []
    if dx > 0.12:
        relations.append("left_of")
    if dx < -0.12:
        relations.append("right_of")
    if dy > 0.18:
        relations.append("above")
    if dy < -0.18:
        relations.append("below")
    if _overlap_area(subject_bbox, object_bbox) > 0.0:
        relations.append("overlaps")
    if _bbox_gap_distance(subject_bbox, object_bbox) <= near_relation_gap:
        relations.append("near")
    return relations


def _build_regions(objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    regions = {key: [dict(item) for item in value] for key, value in _EMPTY_REGIONS.items()}
    for item in objects:
        summary = {
            "id": item["id"],
            "label": item["label"],
            "region": item["region"],
            "depth": item["depth"],
        }
        regions[str(item["horizontal_region"])].append(summary)
        regions[str(item["depth"])].append(summary)
    for key, values in regions.items():
        regions[key] = sorted(values, key=lambda item: (str(item["label"]), str(item["id"])))
    return regions


def _temporal_analysis(
    *,
    objects: list[dict[str, Any]],
    previous_scene: Mapping[str, Any] | None,
    motion_threshold: float,
    approach_area_delta: float,
) -> dict[str, Any]:
    previous_objects = _previous_object_map(previous_scene)
    current_objects = {_object_key(item): item for item in objects}
    events: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []

    if not previous_objects:
        default_state = "appeared" if isinstance(previous_scene, Mapping) else "observed"
        for item in objects:
            _set_temporal_state(item, default_state)
            state_item = _temporal_state_item(item, default_state)
            states.append(state_item)
            if default_state == "appeared":
                events.append(_temporal_event(state_item, event_type="appeared", from_region="", to_region=str(item.get("region", ""))))
        return _temporal_payload(states, events)

    for key, item in sorted(current_objects.items()):
        previous = previous_objects.get(key)
        if previous is None:
            state = "appeared"
            _set_temporal_state(item, state)
            state_item = _temporal_state_item(item, state)
            states.append(state_item)
            events.append(_temporal_event(state_item, event_type=state, from_region="", to_region=str(item.get("region", ""))))
            continue

        distance = _distance_between_objects(previous, item)
        area_delta = _coerce_float(item.get("area"), default=0.0) - _coerce_float(previous.get("area"), default=_area_from_object(previous))
        from_region = _first_text(previous.get("region"))
        to_region = _first_text(item.get("region"))
        state = _matched_temporal_state(
            distance=distance,
            area_delta=area_delta,
            from_region=from_region,
            to_region=to_region,
            motion_threshold=motion_threshold,
            approach_area_delta=approach_area_delta,
        )
        _set_temporal_state(item, state)
        state_item = _temporal_state_item(
            item,
            state,
            from_region=from_region,
            to_region=to_region,
            distance=distance,
            area_delta=area_delta,
        )
        states.append(state_item)
        if state != "stationary":
            events.append(_temporal_event(state_item, event_type=state, from_region=from_region, to_region=to_region))

    for key, previous in sorted(previous_objects.items()):
        if key in current_objects:
            continue
        state_item = _temporal_state_item(previous, "disappeared", from_region=_first_text(previous.get("region")), to_region="")
        states.append(state_item)
        events.append(_temporal_event(state_item, event_type="disappeared", from_region=state_item["fromRegion"], to_region=""))

    return _temporal_payload(states, events)


def _previous_object_map(previous_scene: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(previous_scene, Mapping):
        return {}
    objects = _mapping_list(previous_scene.get("objects"))
    return {_object_key(item): item for item in objects if _object_key(item)}


def _object_key(item: Mapping[str, Any]) -> str:
    return _first_text(item.get("trackId"), item.get("track_id"), item.get("id"))


def _set_temporal_state(item: dict[str, Any], state: str) -> None:
    item["temporalState"] = state


def _temporal_state_item(
    item: Mapping[str, Any],
    state: str,
    *,
    from_region: str = "",
    to_region: str = "",
    distance: float = 0.0,
    area_delta: float = 0.0,
) -> dict[str, Any]:
    return {
        "trackId": _object_key(item),
        "label": _first_text(item.get("label")),
        "state": state,
        "fromRegion": from_region,
        "toRegion": to_region or _first_text(item.get("region")),
        "distance": round(float(distance), 3),
        "areaDelta": round(float(area_delta), 4),
    }


def _temporal_event(
    state_item: Mapping[str, Any],
    *,
    event_type: str,
    from_region: str,
    to_region: str,
) -> dict[str, Any]:
    return {
        "eventType": event_type,
        "trackId": state_item.get("trackId", ""),
        "label": state_item.get("label", ""),
        "fromRegion": from_region,
        "toRegion": to_region,
        "distance": state_item.get("distance", 0.0),
        "areaDelta": state_item.get("areaDelta", 0.0),
    }


def _temporal_payload(states: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "states": states,
        "events": events,
        "eventSummary": _temporal_summary(states),
    }


def _build_lightweight_events(objects: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    objects_by_id = {str(item.get("id")): item for item in objects}
    for item in objects:
        if item.get("kind") == "person" and item.get("looking_at_device") is True:
            target = _nearest_device(item, objects)
            events.append(
                _lightweight_event(
                    event_type="looking_at_device",
                    subject=item,
                    obj=target,
                    confidence=_coerce_float(item.get("confidence")),
                )
            )
    for relation in relations:
        if relation.get("relation") != "hand_near_object":
            continue
        subject = objects_by_id.get(str(relation.get("subjectId")))
        obj = objects_by_id.get(str(relation.get("objectId")))
        if subject is None or obj is None:
            continue
        events.append(
            _lightweight_event(
                event_type="hand_near_object",
                subject=subject,
                obj=obj,
                confidence=min(_coerce_float(subject.get("confidence")), _coerce_float(obj.get("confidence"))),
            )
        )
    return sorted(events, key=lambda item: (str(item["eventType"]), str(item["trackId"]), str(item.get("objectId", ""))))


def _lightweight_event(
    *,
    event_type: str,
    subject: Mapping[str, Any],
    obj: Mapping[str, Any] | None,
    confidence: float,
) -> dict[str, Any]:
    event = {
        "eventType": event_type,
        "trackId": _object_key(subject),
        "label": _first_text(subject.get("label")),
        "confidence": round(confidence, 3),
        "source": _first_text(subject.get("source")),
        "model_id": _first_text(subject.get("model_id")),
        "provenance": dict(subject.get("provenance")) if isinstance(subject.get("provenance"), Mapping) else {},
    }
    if obj is not None:
        event.update(
            {
                "objectId": _object_key(obj),
                "objectLabel": _first_text(obj.get("label")),
            }
        )
    return event


def _combined_event_summary(temporal: Mapping[str, Any], lightweight_events: list[dict[str, Any]]) -> str:
    parts = [_first_text(temporal.get("eventSummary"))]
    if lightweight_events:
        parts.append(
            "; ".join(
                f"{event['eventType']} {event['trackId']}"
                for event in lightweight_events
                if event.get("trackId")
            )
        )
    return "; ".join(part for part in parts if part)


def _temporal_summary(states: list[dict[str, Any]]) -> str:
    if not states:
        return "no temporal changes"
    return "; ".join(
        f"{item['state']} {item['trackId']}"
        for item in sorted(states, key=lambda value: (str(value.get("state")), str(value.get("trackId"))))
        if item.get("trackId")
    )


def _dominant_target(objects: list[dict[str, Any]]) -> dict[str, Any]:
    if not objects:
        return {}
    target = max(
        objects,
        key=lambda item: (
            float(item.get("area", 0.0)) * max(float(item.get("confidence", 0.0)), 0.0),
            1 if item.get("kind") == "person" else 0,
            str(item.get("id", "")),
        ),
    )
    salience = round(float(target["area"]) * max(float(target["confidence"]), 0.0), 4)
    return {
        "id": target["id"],
        "trackId": target["trackId"],
        "label": target["label"],
        "kind": target["kind"],
        "confidence": target["confidence"],
        "region": target["region"],
        "depth": target["depth"],
        "salience": salience,
    }


def _summary(
    objects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    dominant_target: Mapping[str, Any],
    safety: Mapping[str, Any],
    temporal: Mapping[str, Any],
) -> str:
    path_clear = "unknown" if safety.get("pathClear") is None else str(safety.get("pathClear")).lower()
    near_obstacle = "unknown" if safety.get("nearObstacle") is None else str(safety.get("nearObstacle")).lower()
    if not objects:
        return f"Observed empty visual scene; pathClear {path_clear}; nearObstacle {near_obstacle}"

    counts: dict[str, int] = {}
    for item in objects:
        label = str(item["label"])
        counts[label] = counts.get(label, 0) + 1
    observed = ", ".join(
        f"{count} {label}" if count > 1 else label
        for label, count in sorted(counts.items())
    )
    relation_names = sorted({str(item["relation"]) for item in relations})
    relation_summary = ", ".join(relation_names) if relation_names else "none"
    target_label = str(dominant_target.get("label", "none"))
    target_region = str(dominant_target.get("region", ""))
    return (
        f"Observed {observed}; dominant target {target_label} in {target_region}; "
        f"relations: {relation_summary}; temporal: {temporal.get('eventSummary', 'no temporal changes')}; "
        f"pathClear {path_clear}; nearObstacle {near_obstacle}"
    )


def _frame_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = raw if isinstance(raw, Mapping) else {}
    frame_id = _first_text(raw.get("frameId"), raw.get("frame_id"), raw.get("frame"), raw.get("id"))
    observed_at = _first_text(raw.get("observedAt"), raw.get("observed_at"), raw.get("timestamp"), raw.get("time"))
    metadata: dict[str, Any] = {
        "frameId": frame_id,
        "observedAt": observed_at,
        "source": "vision_scene_graph",
    }
    width = _optional_int(raw.get("width"))
    height = _optional_int(raw.get("height"))
    if width is not None:
        metadata["width"] = width
    if height is not None:
        metadata["height"] = height
    image_url = _first_text(raw.get("imageUrl"), raw.get("image_url"))
    if image_url:
        metadata["imageUrl"] = image_url
    for key, value in raw.items():
        if key in {"frameId", "frame_id", "frame", "id", "observedAt", "observed_at", "timestamp", "time"}:
            continue
        if key in {"width", "height", "imageUrl", "image_url"}:
            continue
        metadata[str(key)] = value
    return metadata


def _public_metadata(frame: Mapping[str, Any]) -> dict[str, Any]:
    return dict(frame)


def _normalize_bbox(raw: Any, *, width: Any = None, height: Any = None, format_hint: str = "") -> dict[str, float] | None:
    values = _bbox_values(raw, format_hint=format_hint)
    if values is None:
        return None
    x_min, y_min, x_max, y_max = values
    frame_width = _optional_int(width)
    frame_height = _optional_int(height)
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


def _bbox_values(raw: Any, *, format_hint: str = "") -> tuple[float, float, float, float] | None:
    try:
        if isinstance(raw, Mapping):
            if "x" in raw and "y" in raw and ("w" in raw or "width" in raw) and ("h" in raw or "height" in raw):
                x_min = float(raw.get("x", 0.0))
                y_min = float(raw.get("y", 0.0))
                return (
                    x_min,
                    y_min,
                    x_min + float(raw.get("w", raw.get("width", 0.0))),
                    y_min + float(raw.get("h", raw.get("height", 0.0))),
                )
            if "x1" in raw and "y1" in raw and "x2" in raw and "y2" in raw:
                return (
                    float(raw.get("x1", 0.0)),
                    float(raw.get("y1", 0.0)),
                    float(raw.get("x2", 0.0)),
                    float(raw.get("y2", 0.0)),
                )
            return (
                float(raw.get("x_min", raw.get("xmin", raw.get("left", 0.0)))),
                float(raw.get("y_min", raw.get("ymin", raw.get("top", 0.0)))),
                float(raw.get("x_max", raw.get("xmax", raw.get("right", 0.0)))),
                float(raw.get("y_max", raw.get("ymax", raw.get("bottom", 0.0)))),
            )
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
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
    except (TypeError, ValueError):
        return None
    return None


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


def _object_id(
    raw: Mapping[str, Any],
    *,
    label: str,
    center: tuple[float, float],
    source: str,
    index: int,
    seen_ids: set[str],
) -> str:
    object_id = _first_text(
        raw.get("track_id"),
        raw.get("trackId"),
        raw.get("object_id"),
        raw.get("objectId"),
        raw.get("stable_id"),
        raw.get("stableId"),
        raw.get("id"),
    )
    if not object_id:
        object_id = f"{_slug(label)}:{int(center[0] * 10):02d}:{int(center[1] * 10):02d}"
    candidate = object_id
    suffix = 2
    while candidate in seen_ids:
        candidate = f"{object_id}:{source}:{index:02d}:{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    return candidate


def _normalize_pose(raw: Any, *, frame: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    keypoints = raw.get("keypoints")
    if not isinstance(keypoints, list):
        return {}
    normalized = []
    for item in keypoints:
        if not isinstance(item, Mapping):
            continue
        name = _first_text(item.get("name"), item.get("label"), item.get("part"))
        x = _optional_float(item.get("x"))
        y = _optional_float(item.get("y"))
        if not name or x is None or y is None:
            continue
        width = _optional_int(frame.get("width"))
        height = _optional_int(frame.get("height"))
        if width and abs(x) > 1.0:
            x /= width
        if height and abs(y) > 1.0:
            y /= height
        point: dict[str, Any] = {"name": name, "x": round(_clip01(x), 4), "y": round(_clip01(y), 4)}
        confidence = _optional_float(item.get("confidence", item.get("score")))
        if confidence is not None:
            point["confidence"] = round(confidence, 3)
        source = _first_text(item.get("source"))
        if source:
            point["source"] = source
        normalized.append(point)
    return {"keypoints": normalized} if normalized else {}


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
            confidence = _optional_float(item.get("confidence", item.get("score")))
            if confidence is not None:
                normalized["confidence"] = round(confidence, 3)
            source = _first_text(item.get("source"))
            if source:
                normalized["source"] = source
            model_id = _first_text(item.get("model_id"), item.get("modelId"), item.get("model"))
            if model_id:
                normalized["model_id"] = model_id
            labels.append(normalized)
        else:
            label = _first_text(item)
            if label:
                labels.append({"label": label})
    return labels


def _normalize_semantic_labels(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    labels = []
    for item in raw:
        label = _first_text(item.get("label") if isinstance(item, Mapping) else item)
        if label and label not in labels:
            labels.append(label)
    return labels


def _structured_depth_m(raw: Mapping[str, Any]) -> float | None:
    depth = _raw_value(raw, "depth")
    if isinstance(depth, Mapping):
        for key in ("median", "subjectMedian", "subject_median", "meters", "m", "value"):
            value = _optional_float(depth.get(key))
            if value is not None:
                return value
    distance = _raw_value(raw, "distance")
    if isinstance(distance, Mapping):
        for key in ("fromCameraM", "trackedTargetM", "nearestObjectM", "meters", "m", "value"):
            value = _optional_float(distance.get(key))
            if value is not None:
                return value
    return None


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


def _aggregate_label_annotations(objects: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in objects:
        labels = _normalize_label_annotations(obj.get(key))
        for label in labels:
            name = str(label.get("label") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            aggregated.append(label)
    return aggregated


def _aggregate_semantic_labels(objects: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for obj in objects:
        for label in _normalize_semantic_labels(obj.get("semantic_labels")):
            if label not in labels:
                labels.append(label)
    return labels


def _first_mapping_or_depth(objects: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for obj in objects:
        value = obj.get("depth_payload" if key == "depth" else key)
        if isinstance(value, Mapping):
            return dict(value)
    for obj in objects:
        if key == "depth" and obj.get("depth_m") is not None:
            return {"median": obj["depth_m"], "unit": "m"}
        if key == "distance" and obj.get("depth_m") is not None:
            return {"fromCameraM": obj["depth_m"]}
    return {}


def _normalize_provenance(raw: Mapping[str, Any], *, source: str, model_id: str) -> dict[str, Any]:
    provenance = dict(raw.get("provenance")) if isinstance(raw.get("provenance"), Mapping) else {}
    if source:
        provenance.setdefault("source", source)
    if model_id:
        provenance.setdefault("model_id", model_id)
    provider = _first_text(raw.get("provider"))
    if provider:
        provenance.setdefault("provider", provider)
    return provenance


def _distance_band(*, depth_m: float | None, area: float, near_area_threshold: float) -> str:
    if depth_m is not None:
        return "near" if depth_m <= 1.0 else "far"
    return "near" if area >= near_area_threshold else "far"


def _hand_near_object(subject: Mapping[str, Any], obj: Mapping[str, Any], *, near_relation_gap: float) -> bool:
    if subject.get("id") == obj.get("id"):
        return False
    bbox = obj.get("bbox")
    if not isinstance(bbox, Mapping):
        return False
    for point in _hand_points(subject):
        if _point_bbox_gap(point, bbox) <= near_relation_gap:
            return True
    return False


def _hand_points(subject: Mapping[str, Any]) -> list[tuple[float, float]]:
    pose = subject.get("pose")
    keypoints = pose.get("keypoints") if isinstance(pose, Mapping) else subject.get("keypoints")
    points: list[tuple[float, float]] = []
    if not isinstance(keypoints, list):
        return points
    for item in keypoints:
        if not isinstance(item, Mapping):
            continue
        name = _first_text(item.get("name"), item.get("label"), item.get("part")).lower()
        if name in _HAND_KEYPOINTS:
            points.append((_coerce_float(item.get("x")), _coerce_float(item.get("y"))))
    return points


def _point_bbox_gap(point: tuple[float, float], bbox: Mapping[str, Any]) -> float:
    x, y = point
    horizontal_gap = max(_coerce_float(bbox.get("x_min")) - x, x - _coerce_float(bbox.get("x_max")), 0.0)
    vertical_gap = max(_coerce_float(bbox.get("y_min")) - y, y - _coerce_float(bbox.get("y_max")), 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _nearest_device(subject: Mapping[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    devices = [item for item in objects if item.get("id") != subject.get("id") and _is_device(item)]
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


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _optional_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _is_person(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in _PERSON_LABELS or normalized.startswith("person")


def _slug(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_") or "object"


def _horizontal_region(x: float) -> str:
    if x < 0.33:
        return "left"
    if x < 0.66:
        return "center"
    return "right"


def _vertical_region(y: float) -> str:
    if y < 0.25:
        return "top"
    if y < 0.75:
        return "middle"
    return "bottom"


def _center(bbox: Mapping[str, float]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _object_center(obj: Mapping[str, Any]) -> tuple[float, float]:
    center = obj.get("center")
    if isinstance(center, Mapping):
        return (_coerce_float(center.get("x")), _coerce_float(center.get("y")))
    bbox = obj.get("bbox")
    return _center(bbox) if isinstance(bbox, Mapping) else (0.0, 0.0)


def _distance_between_objects(previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
    previous_center = _object_center(previous)
    current_center = _object_center(current)
    return math.hypot(previous_center[0] - current_center[0], previous_center[1] - current_center[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _area(bbox: Mapping[str, float]) -> float:
    return max(0.0, float(bbox["x_max"]) - float(bbox["x_min"])) * max(0.0, float(bbox["y_max"]) - float(bbox["y_min"]))


def _area_from_object(item: Mapping[str, Any]) -> float:
    area = _coerce_float(item.get("area"), default=-1.0)
    if area >= 0.0:
        return area
    bbox = item.get("bbox")
    return _area(bbox) if isinstance(bbox, Mapping) else 0.0


def _matched_temporal_state(
    *,
    distance: float,
    area_delta: float,
    from_region: str,
    to_region: str,
    motion_threshold: float,
    approach_area_delta: float,
) -> str:
    if area_delta >= max(0.0, approach_area_delta) and distance < motion_threshold and from_region == to_region:
        return "approaching"
    if distance >= motion_threshold or (from_region and to_region and from_region != to_region):
        return "moving"
    return "stationary"


def _overlap_area(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    width = min(float(a["x_max"]), float(b["x_max"])) - max(float(a["x_min"]), float(b["x_min"]))
    height = min(float(a["y_max"]), float(b["y_max"])) - max(float(a["y_min"]), float(b["y_min"]))
    return max(0.0, width) * max(0.0, height)


def _bbox_gap_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    horizontal_gap = max(float(a["x_min"]) - float(b["x_max"]), float(b["x_min"]) - float(a["x_max"]), 0.0)
    vertical_gap = max(float(a["y_min"]) - float(b["y_max"]), float(b["y_min"]) - float(a["y_max"]), 0.0)
    return math.hypot(horizontal_gap, vertical_gap)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scene_id(frame: Mapping[str, Any], objects: list[dict[str, Any]], relations: list[dict[str, Any]]) -> str:
    payload = {
        "frameId": frame.get("frameId", ""),
        "observedAt": frame.get("observedAt", ""),
        "objects": [
            {
                "id": item["id"],
                "label": item["label"],
                "region": item["region"],
                "depth": item["depth"],
            }
            for item in objects
        ],
        "relations": relations,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()[:12]
    return f"scene_vg_{digest}"


__all__ = [
    "build_scene_graph",
    "build_vision_scene_graph",
    "to_eiprotocol_scene_content",
]
