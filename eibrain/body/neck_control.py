"""Unified neck control policy.

This module is intentionally pure: it decides whether a neck intent should
be sent to hardware, but it never touches I2C or servo drivers directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NECK_STATES = {
    "idle",
    "listening",
    "attending_face",
    "tracking_face",
    "holding",
    "recentering",
    "manual_override",
    "fault",
}


@dataclass(slots=True)
class NeckIntent:
    source: str
    target_name: str = ""
    target_x: float | None = None
    target_angle: int | None = None
    priority: int = 50
    confidence: float = 1.0
    ttl_s: float = 1.5
    created_at_ts: float = 0.0
    reason: str = ""

    def expired(self, now_ts: float) -> bool:
        return self.ttl_s >= 0 and now_ts - self.created_at_ts > self.ttl_s


@dataclass(slots=True)
class NeckControlConfig:
    pan_min: int = 40
    pan_max: int = 140
    home_angle: int = 90
    deadband: float = 0.16
    step_gain: float = 18.0
    max_step: int = 6
    min_command_interval_s: float = 0.75
    smoothing_alpha: float = 0.25
    hold_after_miss_s: float = 2.0
    min_confidence: float = 0.3
    allowed_tracking_labels: tuple[str, ...] = ("face", "person")
    invert: bool = False


@dataclass(slots=True)
class NeckControlState:
    state: str = "idle"
    last_angle: int = 90
    desired_angle: int = 90
    last_commanded_angle: int | None = None
    last_command_at_ts: float | None = None
    last_target_x: float | None = None
    active_intent: dict[str, Any] = field(default_factory=dict)
    suppressed_reason: str = ""
    last_command_status: str = "idle"
    intent_count: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "last_angle": self.last_angle,
            "desired_angle": self.desired_angle,
            "last_commanded_angle": self.last_commanded_angle,
            "last_command_at_ts": self.last_command_at_ts,
            "last_target_x": self.last_target_x,
            "active_intent": dict(self.active_intent),
            "suppressed_reason": self.suppressed_reason,
            "last_command_status": self.last_command_status,
            "intent_count": self.intent_count,
        }


@dataclass(slots=True)
class NeckDecision:
    should_command: bool
    angle: int
    state: str
    reason: str = ""
    intent: NeckIntent | None = None

    @property
    def status(self) -> str:
        return "command" if self.should_command else "suppressed"


class NeckPolicy:
    def __init__(self, config: NeckControlConfig | None = None) -> None:
        self.config = config or NeckControlConfig()

    def decide(self, *, intent: NeckIntent, state: NeckControlState, now_ts: float) -> NeckDecision:
        state.intent_count += 1
        if intent.expired(now_ts):
            return self._suppress(intent, state, "intent_expired", now_ts, next_state="holding")
        if intent.confidence < self.config.min_confidence:
            return self._suppress(intent, state, "low_confidence", now_ts, next_state="holding")
        label = intent.target_name.strip().lower()
        is_home = label == "recenter" or intent.source == "safety_home"
        if (
            intent.source == "eye.tracking"
            and not is_home
            and self.config.allowed_tracking_labels
            and label not in self.config.allowed_tracking_labels
        ):
            return self._suppress(intent, state, "label_not_trackable", now_ts, next_state="holding")

        desired = self._desired_angle(intent=intent, state=state)
        next_state = self._state_for_intent(intent)
        since_last = None
        if state.last_command_at_ts is not None:
            since_last = now_ts - state.last_command_at_ts
        if state.last_commanded_angle == desired:
            return self._suppress(intent, state, "same_angle", now_ts, desired=desired, next_state=next_state)
        if since_last is not None and since_last < self.config.min_command_interval_s:
            return self._suppress(intent, state, "min_interval", now_ts, desired=desired, next_state=next_state)
        state.state = next_state
        state.desired_angle = desired
        state.last_angle = desired
        state.last_commanded_angle = desired
        state.last_command_at_ts = now_ts
        state.last_target_x = intent.target_x
        state.active_intent = self._intent_snapshot(intent)
        state.suppressed_reason = ""
        state.last_command_status = "command"
        return NeckDecision(True, desired, next_state, intent=intent)

    def mark_command_status(self, state: NeckControlState, status: str) -> None:
        state.last_command_status = status or state.last_command_status
        if status not in {"ok", "command", "suppressed", "idle"}:
            state.state = "fault"

    def _desired_angle(self, *, intent: NeckIntent, state: NeckControlState) -> int:
        if intent.target_angle is not None:
            return self._clip_angle(int(intent.target_angle))
        if intent.target_x is None:
            return self._clip_angle(self.config.home_angle)
        error = max(0.0, min(1.0, float(intent.target_x))) - 0.5
        if self.config.invert:
            error = -error
        if abs(error) <= self.config.deadband:
            return self._clip_angle(state.last_angle)
        delta = int(round(error * self.config.step_gain))
        if delta == 0:
            delta = 1 if error > 0 else -1
        delta = max(-self.config.max_step, min(self.config.max_step, delta))
        raw = self._clip_angle(state.last_angle + delta)
        if state.last_target_x is not None:
            raw = int(round(state.last_angle + ((raw - state.last_angle) * self.config.smoothing_alpha)))
        if raw == state.last_angle and abs(error) > self.config.deadband:
            raw = state.last_angle + (1 if error > 0 else -1)
        return self._clip_angle(raw)

    def _state_for_intent(self, intent: NeckIntent) -> str:
        if intent.source == "manual_override":
            return "manual_override"
        if intent.target_name == "recenter" or intent.source == "safety_home":
            return "recentering"
        if intent.source == "voice.activity":
            return "listening"
        if intent.source == "eye.tracking":
            label = intent.target_name.strip().lower()
            return "tracking_face" if label == "face" else "attending_face"
        return "attending_face"

    def _suppress(
        self,
        intent: NeckIntent,
        state: NeckControlState,
        reason: str,
        now_ts: float,
        *,
        desired: int | None = None,
        next_state: str = "holding",
    ) -> NeckDecision:
        state.state = next_state if next_state in NECK_STATES else "holding"
        state.desired_angle = state.last_angle if desired is None else self._clip_angle(desired)
        state.active_intent = self._intent_snapshot(intent)
        state.suppressed_reason = reason
        state.last_command_status = "suppressed"
        if intent.target_x is not None:
            state.last_target_x = intent.target_x
        return NeckDecision(False, state.desired_angle, state.state, reason=reason, intent=intent)

    def _clip_angle(self, angle: int) -> int:
        return int(max(self.config.pan_min, min(self.config.pan_max, angle)))

    @staticmethod
    def _intent_snapshot(intent: NeckIntent) -> dict[str, Any]:
        return {
            "source": intent.source,
            "target_name": intent.target_name,
            "target_x": intent.target_x,
            "target_angle": intent.target_angle,
            "priority": intent.priority,
            "confidence": intent.confidence,
            "ttl_s": intent.ttl_s,
            "reason": intent.reason,
        }
