"""Neck organ implementation."""

from __future__ import annotations

import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.body.neck_control import NeckControlConfig, NeckControlState, NeckIntent, NeckPolicy
from eibrain.body.organs.base import BaseOrgan
from eibrain.protocol.actions import MoveHeadAction
from eibrain.protocol.outcomes import ActionExecuted


class NeckOrgan(BaseOrgan):
    name = "neck"
    subfunction_names = ("motor", "tracking")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._last_tracking: dict[str, object] | None = None
        self._neck_policy = NeckPolicy(self._neck_control_config())
        self._neck_state = NeckControlState(
            last_angle=self._neck_policy.config.home_angle,
            desired_angle=self._neck_policy.config.home_angle,
        )

    def passive_heartbeat(self) -> OrganHealth:
        motor_details = {"driver": self._driver_kind("motor"), "status": "live_probe_skipped"}
        tracking_details = {"driver": self._driver_kind("tracking"), "status": "live_probe_skipped"}
        if self._last_tracking is not None:
            tracking_details.update(self._last_tracking)
        subfunctions = {
            "motor": SubfunctionHealth(name="motor", health="healthy", details=motor_details),
            "tracking": SubfunctionHealth(name="tracking", health="healthy", details=tracking_details),
        }
        return OrganHealth(organ=self.name, health="healthy", subfunctions=subfunctions)

    def heartbeat(self) -> OrganHealth:
        if self._driver_kind("tracking") == "noop":
            return super().heartbeat()
        motor_state = self._subfunction_health("motor")
        tracking_state = self._tracking_health(motor_state=motor_state)
        subfunctions = {
            "motor": motor_state,
            "tracking": tracking_state,
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def supports_action(self, action) -> bool:
        return isinstance(action, MoveHeadAction)

    def handle_action(self, action):
        if not isinstance(action, MoveHeadAction):
            return None
        now_ts = time.time()
        intent = self._intent_from_action(action=action, now_ts=now_ts)
        decision = self._neck_policy.decide(intent=intent, state=self._neck_state, now_ts=now_ts)
        result = None
        details = {
            "target_id": action.target_id,
            "target_name": action.target_name,
            "target_x": action.target_x,
            "target_angle": decision.angle,
            "neck_control": self.neck_control_snapshot(),
            "neck_decision": {
                "status": decision.status,
                "state": decision.state,
                "reason": decision.reason,
                "should_command": decision.should_command,
            },
        }
        status = "ok"
        if decision.should_command:
            result = self.drivers["motor"].invoke(
                "move_head",
                {
                    "target_id": action.target_id,
                    "target_name": action.target_name,
                    "target_x": action.target_x,
                    "target_angle": decision.angle,
                },
            )
            status = result.status
            self._neck_policy.mark_command_status(self._neck_state, status)
            details.update(result.details)
            details["neck_control"] = self.neck_control_snapshot()
        self._last_tracking = {
            "target_id": action.target_id,
            "target_name": action.target_name,
            "target_x": action.target_x,
            "target_angle": decision.angle,
            "tracked_at_ts": action.ts or now_ts,
            "status": status,
            "neck_control_state": self._neck_state.state,
            "suppressed_reason": decision.reason,
        }
        return ActionExecuted(
            ts=action.ts,
            source="neck.motor",
            status=status,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            action_kind=action.kind,
            details=details,
        )

    def neck_control_snapshot(self) -> dict[str, object]:
        return self._neck_state.snapshot()

    def clear_neck_control(self, *, reason: str = "engagement_inactive") -> None:
        self._neck_state.state = "idle"
        self._neck_state.desired_angle = self._neck_state.last_angle
        self._neck_state.last_target_x = None
        self._neck_state.active_intent = {}
        self._neck_state.suppressed_reason = reason
        self._neck_state.last_command_status = "idle"
        if self._last_tracking is not None:
            self._last_tracking.update(
                {
                    "status": "idle",
                    "neck_control_state": "idle",
                    "suppressed_reason": reason,
                }
            )

    def _intent_from_action(self, *, action: MoveHeadAction, now_ts: float) -> NeckIntent:
        source = action.source or "unknown"
        if action.target_angle is not None and source not in {"eye.tracking", "safety_home"}:
            source = "manual_override"
        if action.target_name == "recenter":
            source = "safety_home"
        priority = 100 if source == "manual_override" else (80 if source == "safety_home" else 60)
        return NeckIntent(
            source=source,
            target_name=action.target_name or "target",
            target_x=action.target_x,
            target_angle=action.target_angle,
            priority=priority,
            confidence=1.0,
            ttl_s=1.5,
            created_at_ts=now_ts,
            reason=action.kind,
        )

    def _neck_control_config(self) -> NeckControlConfig:
        config = self.config.subfunctions.get("motor")
        extra = config.driver.extra if config is not None else {}
        return NeckControlConfig(
            pan_min=int(extra.get("pan_min", 40)),
            pan_max=int(extra.get("pan_max", 140)),
            home_angle=int(extra.get("home_angle", 90)),
            deadband=float(extra.get("tracking_deadband", 0.16)),
            step_gain=float(extra.get("tracking_step_gain", 18.0)),
            max_step=int(extra.get("tracking_max_step", 6)),
            min_command_interval_s=float(extra.get("tracking_min_interval_s", 0.75)),
            smoothing_alpha=float(extra.get("tracking_smoothing_alpha", 0.25)),
            allowed_tracking_labels=tuple(str(item) for item in extra.get("tracking_labels", ("face", "person"))),
            invert=self._extra_bool(extra.get("tracking_invert", False)),
        )

    def _tracking_health(self, *, motor_state: SubfunctionHealth) -> SubfunctionHealth:
        probe = self.drivers["tracking"].heartbeat()
        details = self._merge_probe_details(dict(probe.details))
        details["neck_control"] = self.neck_control_snapshot()
        if self._last_tracking is not None:
            details.update(self._last_tracking)
        if motor_state.health == "unavailable":
            health = "unavailable"
            details["status"] = "motor_unavailable"
            details["error"] = "motor_unavailable"
        elif motor_state.health == "degraded":
            health = "degraded"
            details["status"] = "motor_degraded"
            details["error"] = "motor_degraded"
        elif self._last_tracking is not None:
            last_status = str(self._last_tracking.get("status", probe.status))
            health = self._normalize_status(last_status)
            details["status"] = "tracking_target" if health == "healthy" else last_status
        else:
            health = self._normalize_status(probe.status)
            details["status"] = "tracking_ready"
        return SubfunctionHealth(name="tracking", health=health, details=details)

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    def _current_angle(self, *, default: int) -> int:
        if self._last_tracking is None:
            return default
        try:
            return int(self._last_tracking.get("target_angle", default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extra_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _merge_probe_details(probe: dict[str, object]) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged
