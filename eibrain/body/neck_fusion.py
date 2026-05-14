from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class NeckFusionConfig:
    home_angle: int = 90
    pan_min_angle: int = 40
    pan_max_angle: int = 140
    min_confidence: float = 0.3
    deadband: float = 0.08
    hysteresis: float = 0.03
    consecutive_bias_required: int = 2
    pan_step_gain: float = 24.0
    min_step_degrees: int = 2
    max_step_degrees: int = 8
    min_command_interval_s: float = 0.45
    cooldown_s: float = 0.8
    recenter_after_missing_s: float = 0.75


@dataclass(slots=True)
class NeckFusionActionState:
    action: str = "hold"
    target_angle: int | None = None
    acted_at_ts: float | None = None
    bias_direction: str = "center"
    bias_count: int = 0
    missing_since_ts: float | None = None
    last_seen_at_ts: float | None = None
    last_seen_target_x: float | None = None
    last_score: float = 0.0
    filtered_error: float = 0.0
    stable_error_count: int = 0
    target_jitter_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "target_angle": self.target_angle,
            "acted_at_ts": self.acted_at_ts,
            "bias_direction": self.bias_direction,
            "bias_count": self.bias_count,
            "missing_since_ts": self.missing_since_ts,
            "last_seen_at_ts": self.last_seen_at_ts,
            "last_seen_target_x": self.last_seen_target_x,
            "last_score": self.last_score,
            "filtered_error": self.filtered_error,
            "stable_error_count": self.stable_error_count,
            "target_jitter_reason": self.target_jitter_reason,
        }


@dataclass(slots=True)
class NeckFusionRecommendation:
    action: str
    target_angle: int
    reason: str
    last_action: dict[str, object]


class NeckFusionPolicy:
    def __init__(self, config: NeckFusionConfig | None = None) -> None:
        self.config = config or NeckFusionConfig()

    def recommend(
        self,
        *,
        target_x: float | None,
        score: float,
        current_angle: int,
        last_action: Mapping[str, Any] | NeckFusionActionState | None,
        now_ts: float,
    ) -> NeckFusionRecommendation:
        state = self._state_from(last_action)
        current = self._clip_angle(current_angle)
        home = self._clip_angle(self.config.home_angle)

        if target_x is None:
            return self._recommend_missing_target(state, current, home, now_ts)
        error = self._target_error(target_x)
        if score < self.config.min_confidence:
            next_state = self._state_for_hold(
                state,
                current,
                target_x,
                score,
                now_ts,
                filtered_error=error,
                target_jitter_reason="low_confidence",
            )
            return self._recommend("hold", current, "low_confidence", next_state)

        if abs(error) <= self.config.deadband:
            next_state = self._state_for_hold(
                state,
                current,
                target_x,
                score,
                now_ts,
                filtered_error=error,
                target_jitter_reason="inside_deadband",
            )
            return self._recommend("hold", current, "inside_deadband", next_state)

        activation_threshold = self.config.deadband + self.config.hysteresis
        if abs(error) <= activation_threshold:
            next_state = self._state_for_hold(
                state,
                current,
                target_x,
                score,
                now_ts,
                filtered_error=error,
                target_jitter_reason="within_hysteresis",
            )
            return self._recommend("hold", current, "within_hysteresis", next_state)

        direction = "right" if error > 0 else "left"
        stable_error_count = (
            state.stable_error_count + 1 if state.bias_direction == direction else 1
        )
        next_state = NeckFusionActionState(
            action="hold",
            target_angle=state.target_angle if state.target_angle is not None else current,
            acted_at_ts=state.acted_at_ts,
            bias_direction=direction,
            bias_count=stable_error_count,
            missing_since_ts=None,
            last_seen_at_ts=now_ts,
            last_seen_target_x=target_x,
            last_score=score,
            filtered_error=error,
            stable_error_count=stable_error_count,
            target_jitter_reason=None,
        )
        if stable_error_count < max(1, self.config.consecutive_bias_required):
            return self._recommend("hold", current, "bias_not_confirmed", next_state)

        desired_action = "pan_right" if direction == "right" else "pan_left"
        if self._is_hysteresis_cooldown_active(state, desired_action, error, now_ts):
            return self._recommend("hold", current, "cooldown_active", next_state)
        if self._is_rate_limited(state, now_ts):
            return self._recommend(
                "hold",
                state.target_angle or current,
                "rate_limited",
                next_state,
            )

        step = max(
            self.config.min_step_degrees,
            min(
                self.config.max_step_degrees,
                int(round(abs(error) * self.config.pan_step_gain)),
            ),
        )
        target_angle = self._clip_angle(current + step if direction == "right" else current - step)
        next_state.action = desired_action
        next_state.target_angle = target_angle
        next_state.acted_at_ts = now_ts
        return self._recommend(desired_action, target_angle, "offset_confirmed", next_state)

    def _recommend_missing_target(
        self,
        state: NeckFusionActionState,
        current: int,
        home: int,
        now_ts: float,
    ) -> NeckFusionRecommendation:
        missing_since = (
            state.missing_since_ts
            if state.missing_since_ts is not None
            else (state.last_seen_at_ts if state.last_seen_at_ts is not None else now_ts)
        )
        next_state = NeckFusionActionState(
            action="hold",
            target_angle=state.target_angle if state.target_angle is not None else current,
            acted_at_ts=state.acted_at_ts,
            bias_direction="center",
            bias_count=0,
            missing_since_ts=missing_since,
            last_seen_at_ts=state.last_seen_at_ts,
            last_seen_target_x=state.last_seen_target_x,
            last_score=0.0,
            filtered_error=state.filtered_error,
            stable_error_count=0,
            target_jitter_reason="target_missing",
        )
        if current == home:
            return self._recommend("hold", home, "target_missing_centered", next_state)
        if now_ts - missing_since < self.config.recenter_after_missing_s:
            return self._recommend("hold", current, "target_missing_hold", next_state)
        next_state.action = "recenter"
        next_state.target_angle = home
        next_state.acted_at_ts = now_ts
        return self._recommend("recenter", home, "target_missing_recenter", next_state)

    def _state_for_hold(
        self,
        state: NeckFusionActionState,
        current: int,
        target_x: float,
        score: float,
        now_ts: float,
        filtered_error: float = 0.0,
        stable_error_count: int = 0,
        target_jitter_reason: str | None = None,
    ) -> NeckFusionActionState:
        return NeckFusionActionState(
            action="hold",
            target_angle=state.target_angle if state.target_angle is not None else current,
            acted_at_ts=state.acted_at_ts,
            bias_direction="center",
            bias_count=0,
            missing_since_ts=None,
            last_seen_at_ts=now_ts,
            last_seen_target_x=target_x,
            last_score=score,
            filtered_error=filtered_error,
            stable_error_count=stable_error_count,
            target_jitter_reason=target_jitter_reason,
        )

    def _is_hysteresis_cooldown_active(
        self,
        state: NeckFusionActionState,
        desired_action: str,
        error: float,
        now_ts: float,
    ) -> bool:
        return (
            state.action in {"pan_left", "pan_right"}
            and state.action != desired_action
            and state.acted_at_ts is not None
            and now_ts - state.acted_at_ts < self.config.cooldown_s
            and abs(error) <= self.config.deadband + self.config.hysteresis
        )

    def _is_rate_limited(self, state: NeckFusionActionState, now_ts: float) -> bool:
        return (
            state.acted_at_ts is not None
            and state.action in {"pan_left", "pan_right", "recenter"}
            and now_ts - state.acted_at_ts < self.config.min_command_interval_s
        )

    def _recommend(
        self,
        action: str,
        target_angle: int,
        reason: str,
        state: NeckFusionActionState,
    ) -> NeckFusionRecommendation:
        return NeckFusionRecommendation(
            action=action,
            target_angle=self._clip_angle(target_angle),
            reason=reason,
            last_action=state.to_dict(),
        )

    def _clip_angle(self, angle: int | float) -> int:
        return int(
            max(
                self.config.pan_min_angle,
                min(self.config.pan_max_angle, int(round(angle))),
            )
        )

    @staticmethod
    def _state_from(value: Mapping[str, Any] | NeckFusionActionState | None) -> NeckFusionActionState:
        if isinstance(value, NeckFusionActionState):
            return value
        if isinstance(value, Mapping):
            return NeckFusionActionState(
                action=str(value.get("action") or "hold"),
                target_angle=_int_or_none(value.get("target_angle")),
                acted_at_ts=_float_or_none(value.get("acted_at_ts")),
                bias_direction=str(value.get("bias_direction") or "center"),
                bias_count=max(0, int(value.get("bias_count") or 0)),
                missing_since_ts=_float_or_none(value.get("missing_since_ts")),
                last_seen_at_ts=_float_or_none(value.get("last_seen_at_ts")),
                last_seen_target_x=_float_or_none(value.get("last_seen_target_x")),
                last_score=float(value.get("last_score") or 0.0),
                filtered_error=_float_or_none(value.get("filtered_error")) or 0.0,
                stable_error_count=max(
                    0,
                    int(
                        value.get("stable_error_count")
                        if value.get("stable_error_count") is not None
                        else (value.get("bias_count") or 0)
                    ),
                ),
                target_jitter_reason=(
                    None
                    if value.get("target_jitter_reason") is None
                    else str(value.get("target_jitter_reason"))
                ),
            )
        return NeckFusionActionState()

    @staticmethod
    def _target_error(target_x: float) -> float:
        return max(-0.5, min(0.5, float(target_x) - 0.5))


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
