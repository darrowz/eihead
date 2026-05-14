from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class VisualFollowTuningConfig:
    deadband: float = 0.08
    step_gain: float = 24.0
    max_step: int = 8
    min_interval: float = 0.45
    hold_frames: int = 2
    max_target_freshness_s: float = 0.75
    min_fps: float = 12.0
    boundary_margin_degrees: float = 4.0
    high_error_threshold: float = 0.18
    no_motion_proof_threshold: float = 0.02
    overshoot_min_dx: float = 0.12
    overshoot_ratio: float = 1.25


@dataclass(frozen=True, slots=True)
class VisualFollowTuningTelemetry:
    filtered_error: float | None
    stable_error_count: int
    suppress_reason: str | None
    action_interval_s: float | None
    fps: float | None
    target_freshness_s: float | None
    pan_proof_dx: float | None
    pan_min: int
    pan_max: int
    current_angle: float


@dataclass(frozen=True, slots=True)
class VisualFollowTuningRecommendation:
    deadband: float
    step_gain: float
    max_step: int
    min_interval: float
    hold_frames: int
    reason: str
    safe_to_apply: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "deadband": self.deadband,
            "step_gain": self.step_gain,
            "max_step": self.max_step,
            "min_interval": self.min_interval,
            "hold_frames": self.hold_frames,
            "reason": self.reason,
            "safe_to_apply": self.safe_to_apply,
        }


def recommend_visual_follow_tuning(
    telemetry: VisualFollowTuningTelemetry | Mapping[str, Any],
    config: VisualFollowTuningConfig | None = None,
) -> VisualFollowTuningRecommendation:
    cfg = config or VisualFollowTuningConfig()
    sample = _telemetry_from(telemetry)
    error = _float_or_none(sample.filtered_error) or 0.0
    abs_error = abs(error)
    proof_dx = _float_or_none(sample.pan_proof_dx)
    proof_abs = abs(proof_dx) if proof_dx is not None else None

    if not _target_is_fresh(sample, cfg):
        return _base_recommendation(cfg, "target_stale", safe_to_apply=False)
    if _pan_range_invalid(sample):
        return _base_recommendation(cfg, "invalid_pan_range", safe_to_apply=False)
    if _near_pan_boundary(sample, cfg, error):
        return _recommend_boundary(cfg)
    if _fps_is_low(sample, cfg):
        return _recommend_low_fps(cfg, sample)
    if _is_overshoot(abs_error, error, proof_dx, cfg):
        return _recommend_overshoot(cfg)
    if _is_underresponsive_no_motion(sample, cfg, abs_error, proof_abs):
        return _recommend_no_motion(cfg)
    if _is_excessive_jitter(sample, cfg, abs_error):
        return _recommend_jitter(cfg)

    return _base_recommendation(cfg, "baseline", safe_to_apply=True)


def _recommend_jitter(cfg: VisualFollowTuningConfig) -> VisualFollowTuningRecommendation:
    return VisualFollowTuningRecommendation(
        deadband=_round_float(min(0.18, max(cfg.deadband + 0.04, cfg.deadband * 1.5))),
        step_gain=_round_float(max(10.0, cfg.step_gain * 0.75)),
        max_step=max(2, int(round(cfg.max_step * 0.75))),
        min_interval=_round_float(max(cfg.min_interval + 0.1, cfg.min_interval * 1.4)),
        hold_frames=max(3, cfg.hold_frames + 1),
        reason="jitter_too_sensitive",
        safe_to_apply=True,
    )


def _recommend_no_motion(cfg: VisualFollowTuningConfig) -> VisualFollowTuningRecommendation:
    return VisualFollowTuningRecommendation(
        deadband=_round_float(max(0.03, cfg.deadband * 0.75)),
        step_gain=_round_float(min(40.0, cfg.step_gain * 1.35)),
        max_step=min(12, cfg.max_step + 3),
        min_interval=_round_float(max(0.2, cfg.min_interval * 0.75)),
        hold_frames=1,
        reason="underresponsive_no_motion",
        safe_to_apply=True,
    )


def _recommend_overshoot(cfg: VisualFollowTuningConfig) -> VisualFollowTuningRecommendation:
    return VisualFollowTuningRecommendation(
        deadband=_round_float(min(0.14, max(cfg.deadband, cfg.deadband + 0.02))),
        step_gain=_round_float(max(10.0, cfg.step_gain * 0.65)),
        max_step=max(2, cfg.max_step - 3),
        min_interval=_round_float(max(cfg.min_interval + 0.1, cfg.min_interval * 1.4)),
        hold_frames=max(2, cfg.hold_frames),
        reason="overshoot",
        safe_to_apply=True,
    )


def _recommend_low_fps(
    cfg: VisualFollowTuningConfig,
    sample: VisualFollowTuningTelemetry,
) -> VisualFollowTuningRecommendation:
    frame_s = 1.0 / max(1.0, float(sample.fps or 1.0))
    return VisualFollowTuningRecommendation(
        deadband=cfg.deadband,
        step_gain=_round_float(max(10.0, cfg.step_gain * 0.75)),
        max_step=max(2, min(6, cfg.max_step - 2)),
        min_interval=_round_float(max(cfg.min_interval * 1.5, frame_s * 4.0)),
        hold_frames=max(3, cfg.hold_frames + 1),
        reason="fps_low",
        safe_to_apply=True,
    )


def _recommend_boundary(cfg: VisualFollowTuningConfig) -> VisualFollowTuningRecommendation:
    return VisualFollowTuningRecommendation(
        deadband=_round_float(max(cfg.deadband, cfg.deadband + 0.02)),
        step_gain=_round_float(max(8.0, cfg.step_gain * 0.5)),
        max_step=min(2, cfg.max_step),
        min_interval=_round_float(max(cfg.min_interval, cfg.min_interval * 1.2)),
        hold_frames=max(2, cfg.hold_frames),
        reason="near_pan_boundary",
        safe_to_apply=False,
    )


def _base_recommendation(
    cfg: VisualFollowTuningConfig,
    reason: str,
    *,
    safe_to_apply: bool,
) -> VisualFollowTuningRecommendation:
    return VisualFollowTuningRecommendation(
        deadband=cfg.deadband,
        step_gain=cfg.step_gain,
        max_step=cfg.max_step,
        min_interval=cfg.min_interval,
        hold_frames=cfg.hold_frames,
        reason=reason,
        safe_to_apply=safe_to_apply,
    )


def _target_is_fresh(
    sample: VisualFollowTuningTelemetry,
    cfg: VisualFollowTuningConfig,
) -> bool:
    if _normalize_reason(sample.suppress_reason) in {"target_missing", "target_stale"}:
        return False
    freshness = _float_or_none(sample.target_freshness_s)
    return freshness is not None and freshness <= cfg.max_target_freshness_s


def _pan_range_invalid(sample: VisualFollowTuningTelemetry) -> bool:
    return int(sample.pan_max) <= int(sample.pan_min)


def _near_pan_boundary(
    sample: VisualFollowTuningTelemetry,
    cfg: VisualFollowTuningConfig,
    error: float,
) -> bool:
    pan_min = float(sample.pan_min)
    pan_max = float(sample.pan_max)
    current = float(sample.current_angle)
    margin = max(1.0, cfg.boundary_margin_degrees)
    at_left = current <= pan_min + margin and error < 0.0
    at_right = current >= pan_max - margin and error > 0.0
    return at_left or at_right


def _fps_is_low(
    sample: VisualFollowTuningTelemetry,
    cfg: VisualFollowTuningConfig,
) -> bool:
    fps = _float_or_none(sample.fps)
    return fps is not None and 0.0 < fps < cfg.min_fps


def _is_overshoot(
    abs_error: float,
    error: float,
    proof_dx: float | None,
    cfg: VisualFollowTuningConfig,
) -> bool:
    if proof_dx is None or error == 0.0:
        return False
    if error * proof_dx >= 0.0:
        return False
    return abs(proof_dx) >= max(cfg.overshoot_min_dx, abs_error * cfg.overshoot_ratio)


def _is_underresponsive_no_motion(
    sample: VisualFollowTuningTelemetry,
    cfg: VisualFollowTuningConfig,
    abs_error: float,
    proof_abs: float | None,
) -> bool:
    if abs_error < cfg.high_error_threshold:
        return False
    if int(sample.stable_error_count) < max(4, cfg.hold_frames + 3):
        return False
    if proof_abs is None or proof_abs > cfg.no_motion_proof_threshold:
        return False
    interval = _float_or_none(sample.action_interval_s)
    if interval is not None and interval < cfg.min_interval:
        return False
    reason = _normalize_reason(sample.suppress_reason)
    return reason in {"", "rate_limited", "bias_not_confirmed", "min_interval"}


def _is_excessive_jitter(
    sample: VisualFollowTuningTelemetry,
    cfg: VisualFollowTuningConfig,
    abs_error: float,
) -> bool:
    reason = _normalize_reason(sample.suppress_reason)
    if reason not in {"inside_deadband", "within_hysteresis", "bias_not_confirmed"}:
        return False
    return abs_error <= cfg.deadband * 1.5 and int(sample.stable_error_count) <= 1


def _telemetry_from(
    value: VisualFollowTuningTelemetry | Mapping[str, Any],
) -> VisualFollowTuningTelemetry:
    if isinstance(value, VisualFollowTuningTelemetry):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("telemetry must be VisualFollowTuningTelemetry or a mapping")
    return VisualFollowTuningTelemetry(
        filtered_error=_float_or_none(value.get("filtered_error")),
        stable_error_count=max(0, int(value.get("stable_error_count") or 0)),
        suppress_reason=(
            None if value.get("suppress_reason") is None else str(value.get("suppress_reason"))
        ),
        action_interval_s=_float_or_none(
            value.get("action_interval_s", value.get("action_interval"))
        ),
        fps=_float_or_none(value.get("fps")),
        target_freshness_s=_float_or_none(
            value.get("target_freshness_s", value.get("target_freshness"))
        ),
        pan_proof_dx=_float_or_none(value.get("pan_proof_dx")),
        pan_min=int(value.get("pan_min", 40)),
        pan_max=int(value.get("pan_max", 140)),
        current_angle=float(value.get("current_angle", 90)),
    )


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _normalize_reason(value: str | None) -> str:
    return (value or "").strip().lower()


def _round_float(value: float) -> float:
    return round(float(value), 3)
