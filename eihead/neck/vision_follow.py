"""Pan-only neck action planning for realtime visual target following."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from eihead.eye.tracking import TrackingTarget


@dataclass(frozen=True)
class VisionFollowConfig:
    deadband: float = 0.15
    max_step_deg: float = 8.0
    smoothing: float = 0.5
    min_angle_delta_deg: float = 0.5
    min_action_interval_s: float = 0.25
    pan_gain_deg: float = 20.0
    pan_min_deg: float = 0.0
    pan_max_deg: float = 180.0
    home_pan_deg: float = 90.0
    hold_frames: int = 2
    lost_hold_frames: int = 2
    lost_decay_step_deg: float = 2.0


@dataclass
class VisionFollowState:
    current_pan_deg: float
    last_commanded_pan_deg: float | None = None
    smoothed_error: float = 0.0
    bias_direction: int = 0
    bias_count: int = 0
    last_action_ts: float | None = None
    lost_frames: int = 0


@dataclass(frozen=True)
class PanFollowAction:
    mode: str
    pan_deg: float
    pan_delta_deg: float
    tilt_deg: None = None
    reason: str = ""
    target_label: str | None = None
    target_track_id: Any | None = None
    lock_id: Any | None = None
    frame_id: Any | None = None
    target_angle: float | None = None
    delta: float | None = None
    deadband_applied: bool = False
    suppressed_reason: str | None = None

    def __post_init__(self) -> None:
        if self.target_angle is None:
            object.__setattr__(self, "target_angle", float(self.pan_deg))
        if self.delta is None:
            object.__setattr__(self, "delta", float(self.pan_delta_deg))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pan_deg": self.pan_deg,
            "pan_delta_deg": self.pan_delta_deg,
            "target_angle": self.target_angle,
            "delta": self.delta,
            "tilt_deg": self.tilt_deg,
            "reason": self.reason,
            "deadband_applied": self.deadband_applied,
            "suppressed_reason": self.suppressed_reason,
            "target_label": self.target_label,
            "target_track_id": self.target_track_id,
            "lock_id": self.lock_id,
            "frame_id": self.frame_id,
        }


def plan_pan_follow_action(
    target: TrackingTarget | None,
    *,
    state: VisionFollowState,
    config: VisionFollowConfig | None = None,
    now_ts: float | None = None,
) -> PanFollowAction:
    """Plan one safe horizontal-only follow step and update planner state."""

    config = config or VisionFollowConfig()
    event_ts = time.monotonic() if now_ts is None else float(now_ts)
    if target is None:
        return _plan_lost_target(state=state, config=config)

    state.lost_frames = 0
    state.smoothed_error = _smooth_error(
        previous=state.smoothed_error,
        current=target.horizontal_error,
        smoothing=config.smoothing,
    )
    if abs(state.smoothed_error) < config.deadband:
        state.bias_direction = 0
        state.bias_count = 0
        return _hold_action(
            state=state,
            target=target,
            reason="deadband",
            suppressed_reason="inside_deadband",
            deadband_applied=True,
        )

    _update_bias_state(state=state, error=state.smoothed_error)
    if state.bias_count < max(1, int(config.hold_frames)):
        return _hold_action(
            state=state,
            target=target,
            reason="bias_not_confirmed",
            suppressed_reason="bias_not_confirmed",
        )

    raw_delta = state.smoothed_error * config.pan_gain_deg
    pan_delta = _clamp(raw_delta, -config.max_step_deg, config.max_step_deg)
    next_pan = _clamp(state.current_pan_deg + pan_delta, config.pan_min_deg, config.pan_max_deg)
    pan_delta = next_pan - state.current_pan_deg
    if _is_rate_limited(state=state, config=config, now_ts=event_ts):
        return _hold_action(
            state=state,
            target=target,
            reason="min_interval",
            suppressed_reason="min_interval",
        )
    last_commanded = state.last_commanded_pan_deg
    if last_commanded is not None and abs(next_pan - last_commanded) < config.min_angle_delta_deg:
        return _hold_action(
            state=state,
            target=target,
            reason="min_angle_delta",
            suppressed_reason="min_angle_delta",
        )

    state.current_pan_deg = next_pan
    state.last_commanded_pan_deg = next_pan
    state.last_action_ts = event_ts
    return PanFollowAction(
        mode="track",
        pan_deg=next_pan,
        pan_delta_deg=pan_delta,
        reason="tracking",
        target_label=target.label,
        target_track_id=target.track_id,
        lock_id=_target_lock_id(target),
        frame_id=target.frame_id,
        deadband_applied=False,
    )


def _plan_lost_target(*, state: VisionFollowState, config: VisionFollowConfig) -> PanFollowAction:
    state.lost_frames += 1
    state.smoothed_error = 0.0
    state.bias_direction = 0
    state.bias_count = 0
    if state.lost_frames <= config.lost_hold_frames:
        return PanFollowAction(
            mode="hold",
            pan_deg=state.current_pan_deg,
            pan_delta_deg=0.0,
            reason="target_lost_hold",
            suppressed_reason="target_missing",
        )

    if abs(state.current_pan_deg - config.home_pan_deg) < config.min_angle_delta_deg:
        state.current_pan_deg = config.home_pan_deg
        state.last_commanded_pan_deg = config.home_pan_deg
        return PanFollowAction(
            mode="hold",
            pan_deg=config.home_pan_deg,
            pan_delta_deg=0.0,
            reason="target_lost_home",
            suppressed_reason="target_missing",
        )

    direction = 1.0 if config.home_pan_deg > state.current_pan_deg else -1.0
    pan_delta = direction * min(config.lost_decay_step_deg, abs(config.home_pan_deg - state.current_pan_deg))
    next_pan = _clamp(state.current_pan_deg + pan_delta, config.pan_min_deg, config.pan_max_deg)
    state.current_pan_deg = next_pan
    state.last_commanded_pan_deg = next_pan
    return PanFollowAction(
        mode="decay",
        pan_deg=next_pan,
        pan_delta_deg=pan_delta,
        reason="target_lost_decay",
        suppressed_reason="target_missing",
    )


def _hold_action(
    *,
    state: VisionFollowState,
    target: TrackingTarget | None,
    reason: str,
    suppressed_reason: str,
    deadband_applied: bool = False,
) -> PanFollowAction:
    return PanFollowAction(
        mode="hold",
        pan_deg=state.current_pan_deg,
        pan_delta_deg=0.0,
        reason=reason,
        target_label=None if target is None else target.label,
        target_track_id=None if target is None else target.track_id,
        lock_id=None if target is None else _target_lock_id(target),
        frame_id=None if target is None else target.frame_id,
        deadband_applied=deadband_applied,
        suppressed_reason=suppressed_reason,
    )


def _update_bias_state(*, state: VisionFollowState, error: float) -> None:
    direction = 1 if error > 0.0 else -1
    if state.bias_direction != direction:
        state.bias_direction = direction
        state.bias_count = 1
        return
    state.bias_count += 1


def _is_rate_limited(
    *,
    state: VisionFollowState,
    config: VisionFollowConfig,
    now_ts: float,
) -> bool:
    if config.min_action_interval_s <= 0.0 or state.last_action_ts is None:
        return False
    return (now_ts - state.last_action_ts) < config.min_action_interval_s


def _target_lock_id(target: TrackingTarget) -> Any | None:
    return target.lock_id if target.lock_id is not None else target.track_id


def _smooth_error(*, previous: float, current: float, smoothing: float) -> float:
    smoothing = _clamp(smoothing, 0.0, 1.0)
    return previous * (1.0 - smoothing) + current * smoothing


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
