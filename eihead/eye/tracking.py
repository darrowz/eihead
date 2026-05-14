"""Vision target selection helpers for stable pan-only following."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TRACKING_LABELS = ("person", "face")


@dataclass(frozen=True)
class TrackingTarget:
    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    horizontal_error: float
    score: float
    label: str
    track_id: Any | None = None
    lock_id: Any | None = None
    frame_id: Any | None = None
    age: int = 1
    frame_count: int = 1
    last_seen: Any | None = None
    miss_count: int = 0
    lost: bool = False
    reacquired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "horizontal_error": self.horizontal_error,
            "score": self.score,
            "label": self.label,
            "track_id": self.track_id,
            "lock_id": self.lock_id,
            "frame_id": self.frame_id,
            "age": self.age,
            "frame_count": self.frame_count,
            "last_seen": self.last_seen,
            "miss_count": self.miss_count,
            "lost": self.lost,
            "reacquired": self.reacquired,
        }


@dataclass
class _TrackRecord:
    track_id: Any
    target: TrackingTarget
    frame_count: int = 0
    miss_count: int = 0
    lost: bool = False
    first_seen: Any | None = None
    last_seen: Any | None = None
    last_score: float = 0.0


class LongTermVisualTracker:
    """Stateful selector that keeps a stable lock across realtime frames."""

    def __init__(
        self,
        *,
        preferred_labels: Sequence[str] = DEFAULT_TRACKING_LABELS,
        switch_score_margin: float = 0.12,
        switch_hold_frames: int = 2,
        max_misses: int = 3,
    ) -> None:
        self.preferred_labels = tuple(preferred_labels)
        self.switch_score_margin = max(0.0, float(switch_score_margin))
        self.switch_hold_frames = max(1, int(switch_hold_frames))
        self.max_misses = max(1, int(max_misses))
        self._tracks: dict[Any, _TrackRecord] = {}
        self._active_track_id: Any | None = None
        self._pending_switch_id: Any | None = None
        self._pending_switch_frames = 0
        self._update_count = 0
        self._stable_selection_count = 0
        self._switch_count = 0
        self._reacquired_count = 0
        self._lost_count = 0
        self._suppressed_reason: str | None = None

    def update(
        self,
        detections: Iterable[Mapping[str, Any]],
        *,
        frame_width: float,
        frame_height: float,
        frame_id: Any | None = None,
    ) -> TrackingTarget | None:
        self._update_count += 1
        self._suppressed_reason = None
        candidates = _ranked_tracking_candidates(
            detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_id=frame_id,
            preferred_labels=self.preferred_labels,
        )
        candidates = self._stabilize_synthetic_candidates(candidates)
        observed_ids = {candidate.track_id for candidate in candidates}
        for track_id, record in self._tracks.items():
            if track_id not in observed_ids:
                record.miss_count += 1
                if track_id == self._active_track_id and not record.lost:
                    record.lost = True
                    self._lost_count += 1

        if not candidates:
            self._pending_switch_id = None
            self._pending_switch_frames = 0
            self._suppressed_reason = "target_missing"
            self._clear_expired_active_track()
            return None

        candidate_by_id = {candidate.track_id: candidate for candidate in candidates}
        for candidate in candidates:
            self._note_observed_candidate(candidate, frame_id=frame_id)

        if self._active_track_id is None or self._active_track_id not in self._tracks:
            return self._activate(candidates[0], switched=False)

        active_candidate = candidate_by_id.get(self._active_track_id)
        if active_candidate is None:
            return self._hold_missing_active_or_switch(candidates[0])

        best_candidate = candidates[0]
        if best_candidate.track_id == self._active_track_id:
            self._pending_switch_id = None
            self._pending_switch_frames = 0
            return self._select_active(active_candidate, stable=True)

        if best_candidate.score < active_candidate.score + self.switch_score_margin:
            self._pending_switch_id = None
            self._pending_switch_frames = 0
            self._suppressed_reason = "switch_margin"
            return self._select_active(active_candidate, stable=True)

        if not self._confirm_pending_switch(best_candidate.track_id):
            self._suppressed_reason = "switch_hold"
            return self._select_active(active_candidate, stable=True)

        return self._activate(best_candidate, switched=True)

    def diagnostics(self) -> dict[str, Any]:
        stability_ratio = (
            self._stable_selection_count / self._update_count
            if self._update_count > 0
            else 1.0
        )
        return {
            "track_count": len(self._tracks),
            "active_track_id": self._active_track_id,
            "switch_count": self._switch_count,
            "reacquired_count": self._reacquired_count,
            "lost_count": self._lost_count,
            "stability_ratio": stability_ratio,
            "suppressed_reason": self._suppressed_reason,
        }

    def _note_observed_candidate(self, candidate: TrackingTarget, *, frame_id: Any | None) -> None:
        track_id = candidate.track_id
        record = self._tracks.get(track_id)
        if record is None:
            self._tracks[track_id] = _TrackRecord(
                track_id=track_id,
                target=candidate,
                frame_count=1,
                first_seen=candidate.frame_id if candidate.frame_id is not None else frame_id,
                last_seen=candidate.frame_id if candidate.frame_id is not None else frame_id,
                last_score=candidate.score,
            )
            return
        record.frame_count += 1
        record.target = candidate
        record.last_score = candidate.score
        record.last_seen = candidate.frame_id if candidate.frame_id is not None else frame_id

    def _stabilize_synthetic_candidates(self, candidates: list[TrackingTarget]) -> list[TrackingTarget]:
        stabilized: list[TrackingTarget] = []
        used_track_ids: set[Any] = set()
        for candidate in candidates:
            if not _is_synthetic_bbox_track_id(candidate.track_id, label=candidate.label):
                stabilized.append(candidate)
                used_track_ids.add(candidate.track_id)
                continue
            match = self._nearest_synthetic_match(candidate, used_track_ids=used_track_ids)
            if match is None:
                stabilized.append(candidate)
                used_track_ids.add(candidate.track_id)
                continue
            stable = _copy_target_identity(candidate, track_id=match.track_id, lock_id=match.track_id)
            stabilized.append(stable)
            used_track_ids.add(match.track_id)
        return stabilized

    def _nearest_synthetic_match(
        self,
        candidate: TrackingTarget,
        *,
        used_track_ids: set[Any],
    ) -> _TrackRecord | None:
        matches: list[tuple[float, _TrackRecord]] = []
        for record in self._tracks.values():
            if record.track_id in used_track_ids or record.miss_count > self.max_misses:
                continue
            if record.target.label != candidate.label:
                continue
            if not _is_synthetic_bbox_track_id(record.track_id, label=record.target.label):
                continue
            distance = _center_distance(record.target.bbox, candidate.bbox)
            threshold = max(_bbox_width(record.target.bbox), _bbox_height(record.target.bbox), _bbox_width(candidate.bbox), _bbox_height(candidate.bbox)) * 0.75
            if distance <= threshold:
                matches.append((distance, record))
        return min(matches, key=lambda item: item[0])[1] if matches else None

    def _activate(self, candidate: TrackingTarget, *, switched: bool) -> TrackingTarget:
        previous_active = self._active_track_id
        self._active_track_id = candidate.track_id
        self._pending_switch_id = None
        self._pending_switch_frames = 0
        if switched and previous_active != candidate.track_id:
            self._switch_count += 1
        return self._select_active(candidate, stable=not switched)

    def _select_active(self, candidate: TrackingTarget, *, stable: bool) -> TrackingTarget:
        record = self._tracks[candidate.track_id]
        was_lost = record.lost
        if was_lost:
            self._reacquired_count += 1
        record.miss_count = 0
        record.lost = False
        record.last_seen = candidate.frame_id if candidate.frame_id is not None else record.last_seen
        record.last_score = candidate.score
        if stable:
            self._stable_selection_count += 1
        return _copy_target_with_tracking_state(
            candidate,
            age=record.frame_count,
            frame_count=record.frame_count,
            last_seen=record.last_seen,
            miss_count=record.miss_count,
            lost=record.lost,
            reacquired=was_lost,
        )

    def _hold_missing_active_or_switch(self, best_candidate: TrackingTarget) -> TrackingTarget | None:
        active = self._tracks.get(self._active_track_id)
        active_score = 0.0 if active is None else active.last_score
        if best_candidate.score < active_score + self.switch_score_margin:
            self._suppressed_reason = "active_lost"
            self._clear_expired_active_track()
            return None
        if not self._confirm_pending_switch(best_candidate.track_id):
            self._suppressed_reason = "switch_hold"
            self._clear_expired_active_track()
            return None
        return self._activate(best_candidate, switched=True)

    def _confirm_pending_switch(self, track_id: Any) -> bool:
        if self._pending_switch_id == track_id:
            self._pending_switch_frames += 1
        else:
            self._pending_switch_id = track_id
            self._pending_switch_frames = 1
        return self._pending_switch_frames >= self.switch_hold_frames

    def _clear_expired_active_track(self) -> None:
        active = self._tracks.get(self._active_track_id)
        if active is not None and active.miss_count > self.max_misses:
            self._active_track_id = None


def select_tracking_target(
    detections: Iterable[Mapping[str, Any]],
    *,
    frame_width: float,
    frame_height: float,
    frame_id: Any | None = None,
    preferred_labels: Sequence[str] = DEFAULT_TRACKING_LABELS,
) -> TrackingTarget | None:
    """Pick one detector target for visual following.

    The selector prefers human/face-like labels, then confidence, area, and
    center bias. That keeps the neck planner from hopping to irrelevant high
    confidence objects at frame edges.
    """

    frame_width = float(frame_width)
    frame_height = float(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    preferred = {label.lower() for label in preferred_labels}
    candidates: list[tuple[tuple[float, float, float, float], Mapping[str, Any], float, float, float, float]] = []
    has_preferred = False

    for detection in detections:
        bbox = _coerce_bbox(detection, frame_width=frame_width, frame_height=frame_height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        label = _detection_label(detection)
        score = _as_float(detection.get("score", detection.get("confidence", 0.0)), default=0.0)
        center_x = x1 + width / 2.0
        horizontal_error = (center_x - frame_width / 2.0) / (frame_width / 2.0)
        area_ratio = min(1.0, (width * height) / (frame_width * frame_height))
        center_bias = 1.0 - min(1.0, abs(horizontal_error))
        has_preferred = has_preferred or label in preferred
        candidates.append((bbox, detection, score, area_ratio, center_bias, horizontal_error))

    if has_preferred:
        candidates = [candidate for candidate in candidates if _detection_label(candidate[1]) in preferred]
    if not candidates:
        return None

    bbox, detection, score, _area_ratio, _center_bias, horizontal_error = max(
        candidates,
        key=lambda candidate: (
            candidate[2],
            candidate[3],
            candidate[4],
            -abs(candidate[5]),
        ),
    )
    x1, y1, x2, y2 = bbox
    track_id = detection.get("trackId", detection.get("track_id", detection.get("id")))
    return TrackingTarget(
        bbox=bbox,
        center_x=x1 + (x2 - x1) / 2.0,
        center_y=y1 + (y2 - y1) / 2.0,
        horizontal_error=horizontal_error,
        score=score,
        label=_detection_label(detection),
        track_id=track_id,
        lock_id=detection.get("lockId", detection.get("lock_id", track_id)),
        frame_id=detection.get("frameId", detection.get("frame_id", frame_id)),
    )


def _ranked_tracking_candidates(
    detections: Iterable[Mapping[str, Any]],
    *,
    frame_width: float,
    frame_height: float,
    frame_id: Any | None,
    preferred_labels: Sequence[str],
) -> list[TrackingTarget]:
    frame_width = float(frame_width)
    frame_height = float(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    preferred = {label.lower() for label in preferred_labels}
    candidates: list[tuple[TrackingTarget, float, float]] = []
    has_preferred = False
    for detection in detections:
        bbox = _coerce_bbox(detection, frame_width=frame_width, frame_height=frame_height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        label = _detection_label(detection)
        score = _as_float(detection.get("score", detection.get("confidence", 0.0)), default=0.0)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        horizontal_error = (center_x - frame_width / 2.0) / (frame_width / 2.0)
        area_ratio = min(1.0, (width * height) / (frame_width * frame_height))
        center_bias = 1.0 - min(1.0, abs(horizontal_error))
        raw_track_id = detection.get("trackId", detection.get("track_id", detection.get("id")))
        track_id = _stable_track_id(raw_track_id, label=label, bbox=bbox)
        has_preferred = has_preferred or label in preferred
        candidates.append(
            (
                TrackingTarget(
                    bbox=bbox,
                    center_x=center_x,
                    center_y=center_y,
                    horizontal_error=horizontal_error,
                    score=score,
                    label=label,
                    track_id=track_id,
                    lock_id=detection.get("lockId", detection.get("lock_id", track_id)),
                    frame_id=detection.get("frameId", detection.get("frame_id", frame_id)),
                    last_seen=detection.get("frameId", detection.get("frame_id", frame_id)),
                ),
                area_ratio,
                center_bias,
            )
        )

    if has_preferred:
        candidates = [candidate for candidate in candidates if candidate[0].label in preferred]
    candidates.sort(
        key=lambda candidate: (
            candidate[0].score,
            candidate[1],
            candidate[2],
            -abs(candidate[0].horizontal_error),
        ),
        reverse=True,
    )
    return [candidate[0] for candidate in candidates]


def _copy_target_with_tracking_state(
    target: TrackingTarget,
    *,
    age: int,
    frame_count: int,
    last_seen: Any | None,
    miss_count: int,
    lost: bool,
    reacquired: bool,
) -> TrackingTarget:
    return TrackingTarget(
        bbox=target.bbox,
        center_x=target.center_x,
        center_y=target.center_y,
        horizontal_error=target.horizontal_error,
        score=target.score,
        label=target.label,
        track_id=target.track_id,
        lock_id=target.lock_id,
        frame_id=target.frame_id,
        age=age,
        frame_count=frame_count,
        last_seen=last_seen,
        miss_count=miss_count,
        lost=lost,
        reacquired=reacquired,
    )


def _copy_target_identity(target: TrackingTarget, *, track_id: Any, lock_id: Any) -> TrackingTarget:
    return TrackingTarget(
        bbox=target.bbox,
        center_x=target.center_x,
        center_y=target.center_y,
        horizontal_error=target.horizontal_error,
        score=target.score,
        label=target.label,
        track_id=track_id,
        lock_id=lock_id,
        frame_id=target.frame_id,
        age=target.age,
        frame_count=target.frame_count,
        last_seen=target.last_seen,
        miss_count=target.miss_count,
        lost=target.lost,
        reacquired=target.reacquired,
    )


def _stable_track_id(raw_track_id: Any, *, label: str, bbox: tuple[float, float, float, float]) -> Any:
    if raw_track_id is not None:
        try:
            hash(raw_track_id)
            return raw_track_id
        except TypeError:
            return str(raw_track_id)
    rounded_bbox = tuple(round(value, 1) for value in bbox)
    return f"{label}:{rounded_bbox}"


def _is_synthetic_bbox_track_id(track_id: Any, *, label: str) -> bool:
    return isinstance(track_id, str) and track_id.startswith(f"{label}:(")


def _coerce_bbox(
    detection: Mapping[str, Any],
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float, float, float] | None:
    raw_bbox = detection.get("bbox", detection.get("box", detection.get("xyxy")))
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, Mapping):
        bbox = _coerce_mapping_bbox(raw_bbox)
    elif isinstance(raw_bbox, Sequence) and not isinstance(raw_bbox, (str, bytes)) and len(raw_bbox) >= 4:
        bbox = _coerce_sequence_bbox(raw_bbox, format_hint=_bbox_format(detection))
    else:
        return None
    return _scale_normalized_bbox(bbox, frame_width=frame_width, frame_height=frame_height)


def _coerce_sequence_bbox(raw_bbox: Sequence[Any], *, format_hint: str = "") -> tuple[float, float, float, float]:
    x1, y1, third, fourth = (float(value) for value in raw_bbox[:4])
    if format_hint == "xyxy":
        return (x1, y1, third, fourth)
    if format_hint == "xywh":
        return (x1, y1, x1 + third, y1 + fourth)
    if max(abs(x1), abs(y1), abs(third), abs(fourth)) <= 1.0:
        # eiprotocol list boxes are normalized [x, y, w, h].
        return (x1, y1, x1 + third, y1 + fourth)
    if third <= x1 or fourth <= y1:
        # Pixel-space eiprotocol list boxes are also [x, y, w, h].
        return (x1, y1, x1 + third, y1 + fourth)
    return (x1, y1, third, fourth)


def _coerce_mapping_bbox(raw_bbox: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    for keys in (
        ("x1", "y1", "x2", "y2"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ):
        if all(key in raw_bbox for key in keys):
            return tuple(float(raw_bbox[key]) for key in keys)  # type: ignore[return-value]
    if all(key in raw_bbox for key in ("x", "y", "w", "h")):
        x = float(raw_bbox["x"])
        y = float(raw_bbox["y"])
        return (x, y, x + float(raw_bbox["w"]), y + float(raw_bbox["h"]))
    if all(key in raw_bbox for key in ("x", "y", "width", "height")):
        x = float(raw_bbox["x"])
        y = float(raw_bbox["y"])
        return (x, y, x + float(raw_bbox["width"]), y + float(raw_bbox["height"]))
    return None


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


def _scale_normalized_bbox(
    bbox: tuple[float, float, float, float],
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        return (x1 * frame_width, y1 * frame_height, x2 * frame_width, y2 * frame_height)
    return bbox


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _center_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x = float(first[0]) + _bbox_width(first) / 2.0
    first_y = float(first[1]) + _bbox_height(first) / 2.0
    second_x = float(second[0]) + _bbox_width(second) / 2.0
    second_y = float(second[1]) + _bbox_height(second) / 2.0
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5


def _detection_label(detection: Mapping[str, Any]) -> str:
    return str(detection.get("label", detection.get("name", detection.get("class", "")))).lower()


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
