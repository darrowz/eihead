from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


@dataclass(slots=True)
class VisualTargetLockConfig:
    min_confidence: float = 0.3
    switch_hysteresis: float = 0.5
    lost_timeout: float = 0.75
    target_age: float = 0.0
    max_lock_distance: float = 0.28
    confidence_weight: float = 2.0
    area_weight: float = 1.0
    center_weight: float = 0.5


@dataclass(slots=True)
class VisualTargetLockResult:
    track_id: str | None
    lock_id: str | None
    label: str | None
    bbox: dict[str, float] | None
    center: dict[str, float] | None
    confidence: float
    lock_state: str
    switch_reason: str
    is_locked: bool
    target: dict[str, object] | None
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "lock_id": self.lock_id,
            "label": self.label,
            "bbox": dict(self.bbox) if self.bbox is not None else None,
            "center": dict(self.center) if self.center is not None else None,
            "confidence": self.confidence,
            "lock_state": self.lock_state,
            "switch_reason": self.switch_reason,
            "is_locked": self.is_locked,
            "target": dict(self.target) if self.target is not None else None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(slots=True)
class _Candidate:
    key: str
    track_id: str
    label: str
    bbox: dict[str, float]
    center: dict[str, float]
    confidence: float
    area: float
    priority: int
    priority_name: str
    score: float
    identity_key: str | None
    target: dict[str, object]


class VisualTargetLockSelector:
    """Stable target selector for face/person detections and tracked objects."""

    def __init__(self, config: VisualTargetLockConfig | None = None) -> None:
        self.config = config or VisualTargetLockConfig()
        self._locked: _Candidate | None = None
        self._lock_id: str | None = None
        self._locked_at_ts: float | None = None
        self._last_seen_ts: float | None = None
        self._candidate_seen_since: dict[str, float] = {}

    def select(
        self,
        detections: Iterable[Mapping[str, Any]] | None,
        *,
        now_ts: float,
    ) -> VisualTargetLockResult:
        candidates, diagnostics = self._candidates_from(detections)
        self._update_candidate_ages(candidates, now_ts)

        locked_candidate = self._find_locked_candidate(candidates)
        if locked_candidate is not None:
            previous_track_id = self._locked.track_id if self._locked is not None else None
            best = self._best_candidate(candidates)
            if best is not None and best.key != locked_candidate.key and self._should_switch(best, locked_candidate):
                age = self._candidate_age(best, now_ts)
                if age < max(0.0, self.config.target_age):
                    self._maintain_lock(locked_candidate, now_ts)
                    return self._locked_result(
                        locked_candidate,
                        "locked",
                        "candidate_too_new",
                        diagnostics,
                        previous_track_id=previous_track_id,
                        competing_candidate=best,
                    )
                self._switch_lock(best, now_ts)
                return self._locked_result(
                    best,
                    "switched",
                    "stronger_target",
                    diagnostics,
                    previous_track_id=previous_track_id,
                )
            self._maintain_lock(locked_candidate, now_ts)
            return self._locked_result(locked_candidate, "locked", "maintained", diagnostics)

        if self._locked is not None and self._is_lock_recent(now_ts):
            best = self._best_candidate(candidates)
            if best is not None and self._should_switch(best, self._locked):
                age = self._candidate_age(best, now_ts)
                if age >= max(0.0, self.config.target_age):
                    previous_track_id = self._locked.track_id
                    self._switch_lock(best, now_ts)
                    return self._locked_result(
                        best,
                        "switched",
                        "stronger_target",
                        diagnostics,
                        previous_track_id=previous_track_id,
                    )
            return self._locked_result(self._locked, "lost_hold", "target_lost_hold", diagnostics)

        if not candidates:
            reason = "no_target" if int(diagnostics["raw_count"]) == 0 else "no_eligible_target"
            if self._locked is not None:
                reason = "target_lost_timeout"
            self._clear_lock()
            return self._unlocked_result(reason, diagnostics)

        best = self._best_candidate(candidates)
        if best is None:
            self._clear_lock()
            return self._unlocked_result("no_eligible_target", diagnostics)
        self._switch_lock(best, now_ts)
        return self._locked_result(best, "locked", "initial_lock", diagnostics)

    def reset(self) -> None:
        self._clear_lock()
        self._candidate_seen_since.clear()

    def _candidates_from(
        self,
        detections: Iterable[Mapping[str, Any]] | None,
    ) -> tuple[list[_Candidate], dict[str, object]]:
        diagnostics: dict[str, object] = {
            "raw_count": 0,
            "candidate_count": 0,
            "filtered_invalid": 0,
            "filtered_low_confidence": 0,
            "filtered_unsupported_label": 0,
        }
        candidates: list[_Candidate] = []
        for raw in detections or []:
            diagnostics["raw_count"] = int(diagnostics["raw_count"]) + 1
            candidate = self._candidate_from(raw)
            if candidate is None:
                diagnostics["filtered_invalid"] = int(diagnostics["filtered_invalid"]) + 1
                continue
            if candidate.confidence < self.config.min_confidence:
                diagnostics["filtered_low_confidence"] = int(diagnostics["filtered_low_confidence"]) + 1
                continue
            if candidate.priority <= 0:
                diagnostics["filtered_unsupported_label"] = int(diagnostics["filtered_unsupported_label"]) + 1
                continue
            candidates.append(candidate)

        diagnostics["candidate_count"] = len(candidates)
        return candidates, diagnostics

    def _candidate_from(self, raw: Mapping[str, Any]) -> _Candidate | None:
        label = str(raw.get("label") or raw.get("name") or raw.get("class") or "").strip().lower()
        bbox = _normalize_bbox(raw.get("bbox"))
        if not label or bbox is None:
            return None
        confidence = _coerce_float(raw.get("confidence", raw.get("score", 0.0)))
        center_tuple = _center(bbox)
        center = {"x": round(center_tuple[0], 3), "y": round(center_tuple[1], 3)}
        area = _area(bbox)
        identity_key = _identity_key(raw)
        priority, priority_name = _priority(label, identity_key)
        track_id = _track_id(raw, label, center)
        score = self._score(priority, confidence, area, center_tuple)
        target = {
            "track_id": track_id,
            "label": label,
            "bbox": dict(bbox),
            "center": dict(center),
            "confidence": round(confidence, 3),
        }
        _copy_optional(raw, target, "registered_identity")
        _copy_optional(raw, target, "person_id")
        _copy_optional(raw, target, "personId", target_key="person_id")
        return _Candidate(
            key=identity_key or track_id,
            track_id=track_id,
            label=label,
            bbox=bbox,
            center=center,
            confidence=confidence,
            area=area,
            priority=priority,
            priority_name=priority_name,
            score=score,
            identity_key=identity_key,
            target=target,
        )

    def _score(
        self,
        priority: int,
        confidence: float,
        area: float,
        center: tuple[float, float],
    ) -> float:
        center_distance = math.hypot(center[0] - 0.5, center[1] - 0.5)
        center_proximity = 1.0 - min(1.0, center_distance / math.sqrt(0.5))
        return (
            (priority * 10.0)
            + (confidence * self.config.confidence_weight)
            + (area * self.config.area_weight)
            + (center_proximity * self.config.center_weight)
        )

    def _update_candidate_ages(self, candidates: list[_Candidate], now_ts: float) -> None:
        active_keys = {candidate.key for candidate in candidates}
        for key in active_keys:
            self._candidate_seen_since.setdefault(key, now_ts)
        for key in list(self._candidate_seen_since):
            if key not in active_keys:
                del self._candidate_seen_since[key]

    def _find_locked_candidate(self, candidates: list[_Candidate]) -> _Candidate | None:
        if self._locked is None:
            return None
        if self._locked.identity_key is not None:
            for candidate in candidates:
                if candidate.identity_key == self._locked.identity_key:
                    return candidate
        for candidate in candidates:
            if candidate.track_id == self._locked.track_id:
                return candidate

        nearby: list[tuple[float, _Candidate]] = []
        locked_center = _center_from_dict(self._locked.center)
        for candidate in candidates:
            if candidate.label != self._locked.label:
                continue
            distance = _distance(locked_center, _center_from_dict(candidate.center))
            if distance <= self.config.max_lock_distance:
                nearby.append((distance, candidate))
        if not nearby:
            return None
        return sorted(nearby, key=lambda item: item[0])[0][1]

    def _should_switch(self, candidate: _Candidate, current: _Candidate) -> bool:
        return candidate.score >= current.score + max(0.0, self.config.switch_hysteresis)

    def _candidate_age(self, candidate: _Candidate, now_ts: float) -> float:
        return max(0.0, now_ts - self._candidate_seen_since.get(candidate.key, now_ts))

    def _is_lock_recent(self, now_ts: float) -> bool:
        if self._last_seen_ts is None:
            return False
        return now_ts - self._last_seen_ts <= max(0.0, self.config.lost_timeout)

    def _maintain_lock(self, candidate: _Candidate, now_ts: float) -> None:
        self._locked = candidate
        if self._locked_at_ts is None:
            self._locked_at_ts = now_ts
        self._last_seen_ts = now_ts

    def _switch_lock(self, candidate: _Candidate, now_ts: float) -> None:
        self._locked = candidate
        self._lock_id = candidate.identity_key or candidate.track_id
        self._locked_at_ts = now_ts
        self._last_seen_ts = now_ts

    def _clear_lock(self) -> None:
        self._locked = None
        self._lock_id = None
        self._locked_at_ts = None
        self._last_seen_ts = None

    def _best_candidate(self, candidates: list[_Candidate]) -> _Candidate | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                candidate.score,
                candidate.priority,
                candidate.confidence,
                candidate.area,
                candidate.track_id,
            ),
        )

    def _locked_result(
        self,
        candidate: _Candidate,
        lock_state: str,
        switch_reason: str,
        diagnostics: dict[str, object],
        *,
        previous_track_id: str | None = None,
        competing_candidate: _Candidate | None = None,
    ) -> VisualTargetLockResult:
        result_diagnostics = dict(diagnostics)
        target_payload = dict(candidate.target)
        if self._lock_id is not None:
            target_payload["lock_id"] = self._lock_id
        result_diagnostics.update(
            {
                "lock_id": self._lock_id,
                "lock_track_id": self._lock_id,
                "selected_priority": candidate.priority_name,
                "selected_score": round(candidate.score, 4),
                "locked_at_ts": self._locked_at_ts,
                "last_seen_ts": self._last_seen_ts,
            }
        )
        if previous_track_id is not None:
            result_diagnostics["previous_track_id"] = previous_track_id
        if competing_candidate is not None:
            result_diagnostics["competing_track_id"] = competing_candidate.track_id
            result_diagnostics["competing_score"] = round(competing_candidate.score, 4)
        return VisualTargetLockResult(
            track_id=candidate.track_id,
            lock_id=self._lock_id,
            label=candidate.label,
            bbox=dict(candidate.bbox),
            center=dict(candidate.center),
            confidence=round(candidate.confidence, 3),
            lock_state=lock_state,
            switch_reason=switch_reason,
            is_locked=True,
            target=target_payload,
            diagnostics=result_diagnostics,
        )

    def _unlocked_result(
        self,
        switch_reason: str,
        diagnostics: dict[str, object],
    ) -> VisualTargetLockResult:
        result_diagnostics = dict(diagnostics)
        result_diagnostics.update(
            {
                "lock_id": None,
                "lock_track_id": None,
                "selected_priority": None,
                "selected_score": 0.0,
                "locked_at_ts": None,
                "last_seen_ts": None,
            }
        )
        return VisualTargetLockResult(
            track_id=None,
            lock_id=None,
            label=None,
            bbox=None,
            center=None,
            confidence=0.0,
            lock_state="unlocked",
            switch_reason=switch_reason,
            is_locked=False,
            target=None,
            diagnostics=result_diagnostics,
        )


def _priority(label: str, identity_key: str | None) -> tuple[int, str]:
    if identity_key:
        return 3, "identity"
    if label == "face":
        return 2, "face"
    if label == "person":
        return 1, "person"
    return 0, "unsupported"


def _identity_key(raw: Mapping[str, Any]) -> str | None:
    registered = raw.get("registered_identity", raw.get("registeredIdentity"))
    if isinstance(registered, Mapping):
        value = (
            registered.get("id")
            or registered.get("person_id")
            or registered.get("personId")
            or registered.get("name")
        )
    else:
        value = registered
    value = value or raw.get("person_id") or raw.get("personId")
    if value is None:
        return None
    text = str(value).strip()
    return f"identity:{text}" if text else None


def _track_id(raw: Mapping[str, Any], label: str, center: Mapping[str, float]) -> str:
    value = raw.get("track_id") or raw.get("trackId") or raw.get("id")
    if value is not None and str(value).strip():
        return str(value).strip()
    return f"{label}:{float(center['x']):.3f}:{float(center['y']):.3f}"


def _normalize_bbox(raw: Any) -> dict[str, float] | None:
    if isinstance(raw, Mapping):
        try:
            x_min = _coerce_float(raw.get("x_min", raw.get("xmin", raw.get("left", raw.get("x", 0.0)))))
            y_min = _coerce_float(raw.get("y_min", raw.get("ymin", raw.get("top", raw.get("y", 0.0)))))
            if "x_max" in raw or "xmax" in raw or "right" in raw:
                x_max = _coerce_float(raw.get("x_max", raw.get("xmax", raw.get("right"))))
            else:
                x_max = x_min + _coerce_float(raw.get("width", raw.get("w", 0.0)))
            if "y_max" in raw or "ymax" in raw or "bottom" in raw:
                y_max = _coerce_float(raw.get("y_max", raw.get("ymax", raw.get("bottom"))))
            else:
                y_max = y_min + _coerce_float(raw.get("height", raw.get("h", 0.0)))
        except (TypeError, ValueError):
            return None
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 4:
        try:
            x_min = _coerce_float(raw[0])
            y_min = _coerce_float(raw[1])
            x_max = _coerce_float(raw[2])
            y_max = _coerce_float(raw[3])
        except (TypeError, ValueError):
            return None
    else:
        return None

    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    if x_max == x_min or y_max == y_min:
        return None
    return {
        "x_min": _clip01(x_min),
        "y_min": _clip01(y_min),
        "x_max": _clip01(x_max),
        "y_max": _clip01(y_max),
    }


def _copy_optional(
    source: Mapping[str, Any],
    target: dict[str, object],
    source_key: str,
    *,
    target_key: str | None = None,
) -> None:
    if source_key in source and source[source_key] is not None:
        target[target_key or source_key] = source[source_key]


def _center(bbox: Mapping[str, float]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
    )


def _center_from_dict(center: Mapping[str, float]) -> tuple[float, float]:
    return (float(center["x"]), float(center["y"]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _area(bbox: Mapping[str, float]) -> float:
    return max(0.0, float(bbox["x_max"]) - float(bbox["x_min"])) * max(
        0.0,
        float(bbox["y_max"]) - float(bbox["y_min"]),
    )


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "VisualTargetLockConfig",
    "VisualTargetLockResult",
    "VisualTargetLockSelector",
]
