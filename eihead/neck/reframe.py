"""Human-like pan reframe planning for visual attention.

This module does not follow targets continuously and never touches hardware.
It plans occasional pan moves when the current view is too poor to recognize a
target, then observes, returns home, and cools down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReframeConfig:
    pan_min_deg: float = 40.0
    pan_max_deg: float = 140.0
    home_pan_deg: float = 90.0
    comfort_min_x: float = 0.18
    comfort_max_x: float = 0.82
    clear_min_x: float = 0.30
    clear_max_x: float = 0.70
    min_face_width_px: int = 80
    min_face_height_px: int = 80
    min_crop_aspect: float = 0.45
    max_crop_aspect: float = 1.40
    min_confidence: float = 0.45
    confirm_frames: int = 3
    reframe_step_deg: float = 5.0
    return_step_deg: float = 4.0
    min_command_interval_s: float = 1.0
    observe_hold_s: float = 1.5
    cooldown_s: float = 3.0
    anti_oscillation_cooldown_s: float = 3.0


@dataclass(slots=True)
class ReframeState:
    current_pan_deg: float = 90.0
    last_commanded_pan_deg: float | None = None
    phase: str = "idle"
    phase_started_at_ts: float | None = None
    last_command_at_ts: float | None = None
    last_direction: int = 0
    unclear_direction: int = 0
    unclear_count: int = 0
    suppressed_reason: str = ""
    command_count: int = 0
    suppressed_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "current_pan_deg": self.current_pan_deg,
            "last_commanded_pan_deg": self.last_commanded_pan_deg,
            "phase": self.phase,
            "phase_started_at_ts": self.phase_started_at_ts,
            "last_command_at_ts": self.last_command_at_ts,
            "last_direction": self.last_direction,
            "unclear_direction": self.unclear_direction,
            "unclear_count": self.unclear_count,
            "suppressed_reason": self.suppressed_reason,
            "command_count": self.command_count,
            "suppressed_count": self.suppressed_count,
        }


@dataclass(frozen=True, slots=True)
class VisualTarget:
    label: str
    target_x: float
    known: bool = False
    confidence: float = 1.0
    crop_width: int | None = None
    crop_height: int | None = None
    frame_id: Any | None = None
    track_id: Any | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ReframeAction:
    mode: str
    pan_deg: float
    pan_delta_deg: float
    will_move: bool
    reason: str
    target_x: float | None = None
    target_label: str | None = None
    target_known: bool | None = None
    frame_id: Any | None = None
    state: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pan_deg": self.pan_deg,
            "pan_delta_deg": self.pan_delta_deg,
            "will_move": self.will_move,
            "reason": self.reason,
            "target_x": self.target_x,
            "target_label": self.target_label,
            "target_known": self.target_known,
            "frame_id": self.frame_id,
            "state": dict(self.state or {}),
        }


def plan_reframe_action(
    target: VisualTarget | None,
    *,
    state: ReframeState,
    config: ReframeConfig | None = None,
    now_ts: float,
) -> ReframeAction:
    config = config or ReframeConfig()
    now = float(now_ts)

    if state.phase == "cooldown" and not _elapsed(state.phase_started_at_ts, now, config.cooldown_s):
        return _hold(state, target, reason="cooldown")
    if state.phase == "cooldown":
        _set_phase(state, "idle", now)

    if state.phase == "observe" and not _elapsed(state.phase_started_at_ts, now, config.observe_hold_s):
        return _observe(state, target)
    if state.phase in {"observe", "return_home"}:
        return _return_home(state=state, target=target, config=config, now_ts=now)

    unclear, reason, direction = _target_unclear(target, config=config)
    if not unclear:
        state.unclear_count = 0
        state.unclear_direction = 0
        return _hold(state, target, reason=reason)

    if state.unclear_direction != direction:
        state.unclear_direction = direction
        state.unclear_count = 1
    else:
        state.unclear_count += 1

    if state.unclear_count < max(1, int(config.confirm_frames)):
        return _hold(state, target, reason="needs_confirmation")
    if _rate_limited(state, config=config, now_ts=now):
        return _hold(state, target, reason="min_interval")
    if (
        state.last_direction
        and direction
        and state.last_direction != direction
        and state.last_command_at_ts is not None
        and now - state.last_command_at_ts < config.anti_oscillation_cooldown_s
    ):
        return _hold(state, target, reason="direction_flip_cooldown")

    delta = direction * float(config.reframe_step_deg)
    next_pan = _clamp(state.current_pan_deg + delta, config.pan_min_deg, config.pan_max_deg)
    delta = next_pan - state.current_pan_deg
    if abs(delta) <= 0.0:
        return _hold(state, target, reason="pan_limit")

    state.current_pan_deg = next_pan
    state.last_commanded_pan_deg = next_pan
    state.last_command_at_ts = now
    state.last_direction = direction
    state.command_count += 1
    _set_phase(state, "observe", now)
    return _action("reframe", state, target, pan_delta=delta, will_move=True, reason=reason)


def _target_unclear(target: VisualTarget | None, *, config: ReframeConfig) -> tuple[bool, str, int]:
    if target is None:
        return False, "target_missing", 0
    if target.confidence < config.min_confidence:
        return False, "low_confidence", 0
    target_x = _clamp(float(target.target_x), 0.0, 1.0)
    direction = 1 if target_x > 0.5 else -1
    if target.known and config.clear_min_x <= target_x <= config.clear_max_x and _crop_shape_ok(target, config=config):
        return False, "target_clear", 0
    if target_x < config.comfort_min_x or target_x > config.comfort_max_x:
        return True, "target_at_edge", direction
    if not _crop_shape_ok(target, config=config):
        return True, "crop_unclear", direction
    if not target.known and not (config.clear_min_x <= target_x <= config.clear_max_x):
        return True, "unknown_off_center", direction
    return False, "target_clear", 0


def _crop_shape_ok(target: VisualTarget, *, config: ReframeConfig) -> bool:
    width = target.crop_width
    height = target.crop_height
    if width is None or height is None:
        return True
    if width < config.min_face_width_px or height < config.min_face_height_px:
        return False
    aspect = width / max(1, height)
    return config.min_crop_aspect <= aspect <= config.max_crop_aspect


def _return_home(
    *,
    state: ReframeState,
    target: VisualTarget | None,
    config: ReframeConfig,
    now_ts: float,
) -> ReframeAction:
    if _rate_limited(state, config=config, now_ts=now_ts):
        return _hold(state, target, reason="min_interval")
    difference = config.home_pan_deg - state.current_pan_deg
    if abs(difference) <= config.return_step_deg:
        delta = difference
        state.current_pan_deg = config.home_pan_deg
        state.last_commanded_pan_deg = config.home_pan_deg
        state.last_command_at_ts = now_ts
        state.last_direction = 0
        state.command_count += 1
        _set_phase(state, "cooldown", now_ts)
        return _action("return_home", state, target, pan_delta=delta, will_move=abs(delta) > 0.0, reason="home_reached")
    direction = 1 if difference > 0 else -1
    delta = direction * config.return_step_deg
    state.current_pan_deg = _clamp(state.current_pan_deg + delta, config.pan_min_deg, config.pan_max_deg)
    state.last_commanded_pan_deg = state.current_pan_deg
    state.last_command_at_ts = now_ts
    state.last_direction = direction
    state.command_count += 1
    _set_phase(state, "return_home", now_ts)
    return _action("return_home", state, target, pan_delta=delta, will_move=True, reason="returning_home")


def _hold(state: ReframeState, target: VisualTarget | None, *, reason: str) -> ReframeAction:
    state.suppressed_reason = reason
    state.suppressed_count += 1
    return _action("hold", state, target, pan_delta=0.0, will_move=False, reason=reason)


def _observe(state: ReframeState, target: VisualTarget | None) -> ReframeAction:
    state.suppressed_reason = "observe"
    state.suppressed_count += 1
    return _action("observe", state, target, pan_delta=0.0, will_move=False, reason="observe")


def _action(
    mode: str,
    state: ReframeState,
    target: VisualTarget | None,
    *,
    pan_delta: float,
    will_move: bool,
    reason: str,
) -> ReframeAction:
    return ReframeAction(
        mode=mode,
        pan_deg=float(state.current_pan_deg),
        pan_delta_deg=float(pan_delta),
        will_move=bool(will_move),
        reason=reason,
        target_x=None if target is None else float(target.target_x),
        target_label=None if target is None else target.label,
        target_known=None if target is None else bool(target.known),
        frame_id=None if target is None else target.frame_id,
        state=state.as_dict(),
    )


def _set_phase(state: ReframeState, phase: str, now_ts: float) -> None:
    if state.phase != phase:
        state.phase = phase
        state.phase_started_at_ts = now_ts
    state.suppressed_reason = ""


def _elapsed(started_at_ts: float | None, now_ts: float, duration_s: float) -> bool:
    if started_at_ts is None:
        return True
    return now_ts - started_at_ts >= duration_s


def _rate_limited(state: ReframeState, *, config: ReframeConfig, now_ts: float) -> bool:
    if config.min_command_interval_s <= 0.0 or state.last_command_at_ts is None:
        return False
    return now_ts - state.last_command_at_ts < config.min_command_interval_s


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


__all__ = [
    "ReframeAction",
    "ReframeConfig",
    "ReframeState",
    "VisualTarget",
    "plan_reframe_action",
]
