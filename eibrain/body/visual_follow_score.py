from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VisualFollowScoreConfig:
    max_target_age_s: float = 0.75
    settle_tolerance: float = 0.06
    min_error_reduction: float = 0.02
    min_error_reduction_ratio: float = 0.2
    overshoot_tolerance: float = 0.05
    max_action_elapsed_s: float = 0.5
    max_settle_time_s: float = 0.7
    command_epsilon_degrees: float = 0.001


@dataclass(frozen=True, slots=True)
class VisualFollowScore:
    success: bool
    score: float
    error_reduced: bool
    overshoot: bool
    settled: bool
    reason: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "error_reduced": self.error_reduced,
            "overshoot": self.overshoot,
            "settled": self.settled,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


def score_visual_follow(
    *,
    before_error: float | None,
    after_error: float | None,
    command_angle_delta: float | None,
    target_age_s: float | None,
    action_elapsed_s: float | None,
    settle_time_s: float | None,
    suppressed: bool = False,
    suppressed_reason: str | None = None,
    held: bool = False,
    config: VisualFollowScoreConfig | None = None,
) -> VisualFollowScore:
    cfg = config or VisualFollowScoreConfig()
    before = _float_or_none(before_error)
    after = _float_or_none(after_error)
    command_delta = abs(_float_or_none(command_angle_delta) or 0.0)
    target_age = _float_or_none(target_age_s)
    action_elapsed = _float_or_none(action_elapsed_s)
    settle_elapsed = _float_or_none(settle_time_s)
    commanded = command_delta > max(0.0, cfg.command_epsilon_degrees)
    held = bool(held or not commanded)

    if before is None or after is None:
        metrics = _base_metrics(
            before=before,
            after=after,
            command_delta=command_delta,
            target_age=target_age,
            action_elapsed=action_elapsed,
            settle_elapsed=settle_elapsed,
            commanded=commanded,
            held=held,
            suppressed=suppressed,
            suppressed_reason=suppressed_reason,
            cfg=cfg,
        )
        return _result(False, 0.0, False, False, False, "target_missing", metrics)

    metrics = _base_metrics(
        before=before,
        after=after,
        command_delta=command_delta,
        target_age=target_age,
        action_elapsed=action_elapsed,
        settle_elapsed=settle_elapsed,
        commanded=commanded,
        held=held,
        suppressed=suppressed,
        suppressed_reason=suppressed_reason,
        cfg=cfg,
    )
    error_reduced = bool(metrics["error_reduced"])
    overshoot = bool(metrics["overshoot"])
    settled = bool(metrics["settled"])

    if target_age is None:
        return _result(False, 0.0, error_reduced, overshoot, settled, "target_missing", metrics)
    if not bool(metrics["target_fresh"]):
        return _result(False, 0.0, error_reduced, overshoot, settled, "target_stale", metrics)
    if suppressed:
        reason_suffix = _reason_suffix(suppressed_reason)
        return _result(
            False,
            0.0,
            error_reduced,
            overshoot,
            settled,
            f"suppressed{reason_suffix}",
            metrics,
        )
    if held:
        score = 1.0 if settled else _score_unsettled_hold(metrics)
        return _result(
            settled,
            score,
            error_reduced,
            overshoot,
            settled,
            "held_settled" if settled else "held_unsettled",
            metrics,
        )

    score = _score_commanded_follow(metrics)
    if overshoot:
        return _result(False, score, error_reduced, overshoot, settled, "overshot_target", metrics)
    if not error_reduced:
        return _result(False, score, error_reduced, overshoot, settled, "error_not_reduced", metrics)
    if not settled:
        return _result(False, score, error_reduced, overshoot, settled, "not_settled", metrics)
    if not bool(metrics["action_time_ok"]):
        return _result(False, score, error_reduced, overshoot, settled, "action_slow", metrics)
    if not bool(metrics["settle_time_ok"]):
        return _result(False, score, error_reduced, overshoot, settled, "settle_timeout", metrics)
    return _result(True, score, error_reduced, overshoot, settled, "settled_reduced_error", metrics)


def _base_metrics(
    *,
    before: float | None,
    after: float | None,
    command_delta: float,
    target_age: float | None,
    action_elapsed: float | None,
    settle_elapsed: float | None,
    commanded: bool,
    held: bool,
    suppressed: bool,
    suppressed_reason: str | None,
    cfg: VisualFollowScoreConfig,
) -> dict[str, Any]:
    before_abs = abs(before) if before is not None else None
    after_abs = abs(after) if after is not None else None
    error_reduction = (
        max(0.0, before_abs - after_abs)
        if before_abs is not None and after_abs is not None
        else 0.0
    )
    error_reduction_ratio = (
        error_reduction / before_abs
        if before_abs is not None and before_abs > 0.0
        else 0.0
    )
    settled = after_abs is not None and after_abs <= cfg.settle_tolerance
    crossed_center = (
        before is not None
        and after is not None
        and before != 0.0
        and after != 0.0
        and (before * after) < 0.0
    )
    overshoot = bool(
        crossed_center
        and after_abs is not None
        and after_abs > cfg.overshoot_tolerance
    )
    target_fresh = target_age is not None and target_age <= cfg.max_target_age_s
    action_time_ok = action_elapsed is None or action_elapsed <= cfg.max_action_elapsed_s
    settle_time_ok = settle_elapsed is None or settle_elapsed <= cfg.max_settle_time_s
    error_reduced = (
        error_reduction >= cfg.min_error_reduction
        and error_reduction_ratio >= cfg.min_error_reduction_ratio
    )
    return {
        "before_error": before,
        "after_error": after,
        "before_abs_error": before_abs,
        "after_abs_error": after_abs,
        "error_reduction": error_reduction,
        "error_reduction_ratio": error_reduction_ratio,
        "error_reduced": error_reduced,
        "settled": settled,
        "overshoot": overshoot,
        "crossed_center": crossed_center,
        "command_angle_delta": command_delta,
        "commanded": commanded,
        "held": held,
        "suppressed": bool(suppressed),
        "suppressed_reason": suppressed_reason or "",
        "target_age_s": target_age,
        "target_fresh": target_fresh,
        "action_elapsed_s": action_elapsed,
        "action_time_ok": action_time_ok,
        "settle_time_s": settle_elapsed,
        "settle_time_ok": settle_time_ok,
        "max_target_age_s": cfg.max_target_age_s,
        "settle_tolerance": cfg.settle_tolerance,
        "overshoot_tolerance": cfg.overshoot_tolerance,
        "max_action_elapsed_s": cfg.max_action_elapsed_s,
        "max_settle_time_s": cfg.max_settle_time_s,
    }


def _score_commanded_follow(metrics: dict[str, Any]) -> float:
    improvement = _clip01(float(metrics["error_reduction_ratio"]))
    settle_credit = 1.0 if metrics["settled"] else 0.0
    action_credit = 1.0 if metrics["action_time_ok"] else 0.0
    settle_time_credit = 1.0 if metrics["settle_time_ok"] else 0.0
    score = (
        (0.5 * improvement)
        + (0.3 * settle_credit)
        + (0.1 * action_credit)
        + (0.1 * settle_time_credit)
    )
    if metrics["overshoot"]:
        score -= 0.35
    return _round_score(score)


def _score_unsettled_hold(metrics: dict[str, Any]) -> float:
    after_abs = metrics["after_abs_error"]
    tolerance = max(0.0001, float(metrics["settle_tolerance"]))
    if after_abs is None:
        return 0.0
    return _round_score(max(0.0, 1.0 - (float(after_abs) / (tolerance * 2.0))))


def _result(
    success: bool,
    score: float,
    error_reduced: bool,
    overshoot: bool,
    settled: bool,
    reason: str,
    metrics: dict[str, Any],
) -> VisualFollowScore:
    return VisualFollowScore(
        success=bool(success),
        score=_round_score(score),
        error_reduced=bool(error_reduced),
        overshoot=bool(overshoot),
        settled=bool(settled),
        reason=reason,
        metrics=metrics,
    )


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_score(value: float) -> float:
    return round(_clip01(float(value)), 3)


def _reason_suffix(value: str | None) -> str:
    reason = (value or "").strip()
    return "" if not reason else f"_{reason}"
