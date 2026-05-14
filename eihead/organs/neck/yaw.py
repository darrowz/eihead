"""Pure yaw control policy for the honjia horizontal neck servo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class YawIntent:
    """A hardware-free request to orient the horizontal neck axis."""

    source: str
    target_name: str = ""
    target_x: float | None = None
    target_angle: int | None = None
    confidence: float = 1.0
    ttl_s: float = -1.0
    created_at_ts: float = 0.0
    reason: str = ""

    def expired(self, now_ts: float) -> bool:
        return self.ttl_s >= 0 and now_ts - self.created_at_ts > self.ttl_s


@dataclass(slots=True)
class YawControlConfig:
    pan_min: int = 40
    pan_max: int = 140
    home_angle: int = 90
    deadband: float = 0.16
    step_gain: float = 18.0
    max_step: int = 6
    min_command_interval_s: float = 0.75
    smoothing_alpha: float = 0.25
    min_confidence: float = 0.3
    invert: bool = False


@dataclass(slots=True)
class YawControlState:
    last_angle: int = 90
    desired_angle: int = 90
    last_commanded_angle: int | None = None
    last_command_at_ts: float | None = None
    last_target_x: float | None = None
    suppressed_reason: str = ""
    command_count: int = 0
    suppressed_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "last_angle": self.last_angle,
            "desired_angle": self.desired_angle,
            "last_commanded_angle": self.last_commanded_angle,
            "last_command_at_ts": self.last_command_at_ts,
            "last_target_x": self.last_target_x,
            "suppressed_reason": self.suppressed_reason,
            "command_count": self.command_count,
            "suppressed_count": self.suppressed_count,
        }


@dataclass(frozen=True, slots=True)
class YawDecision:
    should_command: bool
    angle: int
    reason: str = ""
    target_x: float | None = None

    @property
    def status(self) -> str:
        return "command" if self.should_command else "suppressed"


class YawController:
    """Map target observations to stable pan commands without touching hardware."""

    def __init__(self, config: YawControlConfig | None = None) -> None:
        self.config = config or YawControlConfig()

    def decide(self, *, intent: YawIntent, state: YawControlState, now_ts: float) -> YawDecision:
        if intent.expired(now_ts):
            return self._suppress(intent, state, "intent_expired", state.last_angle)
        if intent.confidence < self.config.min_confidence:
            return self._suppress(intent, state, "low_confidence", state.last_angle)

        desired, mapping_reason = self.map_intent_to_pan(intent=intent, state=state)
        if mapping_reason == "deadband":
            return self._suppress(intent, state, "deadband", desired)

        since_last = None
        if state.last_command_at_ts is not None:
            since_last = now_ts - state.last_command_at_ts
        if state.last_commanded_angle == desired:
            return self._suppress(intent, state, "same_angle", desired)
        if since_last is not None and since_last < self.config.min_command_interval_s:
            return self._suppress(intent, state, "min_interval", desired)

        state.desired_angle = desired
        state.last_angle = desired
        state.last_commanded_angle = desired
        state.last_command_at_ts = now_ts
        state.last_target_x = intent.target_x
        state.suppressed_reason = ""
        state.command_count += 1
        return YawDecision(True, desired, target_x=intent.target_x)

    def map_intent_to_pan(self, *, intent: YawIntent, state: YawControlState) -> tuple[int, str]:
        if intent.target_angle is not None:
            return self._clip_angle(int(intent.target_angle)), ""
        if intent.target_x is None:
            return self._clip_angle(self.config.home_angle), ""

        error = self._normalized_error(intent.target_x)
        if abs(error) <= self.config.deadband:
            return self._clip_angle(state.last_angle), "deadband"

        delta = int(round(error * self.config.step_gain))
        if delta == 0:
            delta = 1 if error > 0 else -1
        delta = max(-self.config.max_step, min(self.config.max_step, delta))

        stepped = self._clip_angle(state.last_angle + delta)
        smoothed = stepped
        if state.last_target_x is not None:
            smoothed = int(round(state.last_angle + ((stepped - state.last_angle) * self.config.smoothing_alpha)))
        if smoothed == state.last_angle:
            smoothed = state.last_angle + (1 if error > 0 else -1)
        return self._clip_angle(smoothed), ""

    def _normalized_error(self, target_x: float) -> float:
        centered = max(0.0, min(1.0, float(target_x))) - 0.5
        return -centered if self.config.invert else centered

    def _suppress(
        self,
        intent: YawIntent,
        state: YawControlState,
        reason: str,
        desired: int,
    ) -> YawDecision:
        angle = self._clip_angle(desired)
        state.desired_angle = angle
        state.last_target_x = intent.target_x
        state.suppressed_reason = reason
        state.suppressed_count += 1
        return YawDecision(False, angle, reason=reason, target_x=intent.target_x)

    def _clip_angle(self, angle: int) -> int:
        return int(max(self.config.pan_min, min(self.config.pan_max, angle)))


__all__ = [
    "YawControlConfig",
    "YawControlState",
    "YawController",
    "YawDecision",
    "YawIntent",
]
