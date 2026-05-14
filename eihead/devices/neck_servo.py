"""Servo command adapter for the honjia neck yaw axis.

The adapter owns no hardware policy.  It only translates an already-made yaw
decision into the tiny driver call expected by the current honjia servo driver.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Protocol


class ServoDriver(Protocol):
    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> Any:
        """Move one servo and return the driver's payload/status."""


class NeckServoCommandAdapter:
    """Apply yaw decisions to an injected servo driver."""

    def __init__(self, driver: ServoDriver, *, servo_id: int = 1) -> None:
        self._driver = driver
        self.servo_id = int(servo_id)

    def apply_decision(self, decision: Any) -> dict[str, Any]:
        angle = int(decision.angle)
        if not bool(decision.should_command):
            return {
                "status": "suppressed",
                "reason": str(getattr(decision, "reason", "") or ""),
                "angle": angle,
            }

        payload = self._driver.ctrl_servo(angle, self.servo_id)
        return {
            "status": "ok",
            "servo_id": self.servo_id,
            "angle": angle,
            "payload": _json_safe(payload),
        }

    def apply_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a native pan plan, letting suppressed plans stop before hardware."""

        angle = _plan_angle(plan)
        status = str(plan.get("status", "") or "")
        reason = str(plan.get("reason", "") or "")

        if status in {"invalid", "unsupported"}:
            return {
                "status": status,
                "success": False,
                "reason": reason or status,
                "angle": angle,
            }
        if not bool(plan.get("will_move")):
            return {
                "status": "suppressed",
                "reason": reason,
                "angle": angle,
            }
        if angle is None:
            return {
                "status": "invalid",
                "success": False,
                "reason": "missing_target_angle",
                "angle": None,
            }

        payload = self._driver.ctrl_servo(angle, self.servo_id)
        return {
            "status": "ok",
            "servo_id": self.servo_id,
            "angle": angle,
            "payload": _json_safe(payload),
        }


class UnavailableNeckServoCommandAdapter:
    """Safe no-hardware adapter for non-honjia hosts or missing drivers."""

    def __init__(self, *, node_id: str, reason: str) -> None:
        self.node_id = str(node_id or "")
        self.reason = str(reason or "neck_servo_unavailable")

    def apply_decision(self, decision: Any) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "success": False,
            "reason": self.reason,
            "node_id": self.node_id,
            "angle": _optional_int(getattr(decision, "angle", None)),
        }

    def apply_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "success": False,
            "reason": self.reason,
            "node_id": self.node_id,
            "angle": _plan_angle(plan),
        }


def build_neck_servo_adapter(
    *,
    node_id: str,
    driver: ServoDriver | None = None,
    servo_id: int = 1,
) -> NeckServoCommandAdapter | UnavailableNeckServoCommandAdapter:
    """Return a narrow injected-driver adapter only on honjia."""

    normalized_node_id = str(node_id or "")
    if normalized_node_id != "honjia":
        return UnavailableNeckServoCommandAdapter(
            node_id=normalized_node_id,
            reason="neck_servo_unavailable_off_honjia",
        )
    if driver is None:
        return UnavailableNeckServoCommandAdapter(
            node_id=normalized_node_id,
            reason="neck_servo_driver_unavailable",
        )
    return NeckServoCommandAdapter(driver, servo_id=servo_id)


def _plan_angle(plan: Mapping[str, Any]) -> int | None:
    action = plan.get("action")
    if isinstance(action, Mapping):
        angle = _optional_int(action.get("target_angle", action.get("angle")))
        if angle is not None:
            return angle
    return _optional_int(plan.get("target_angle", plan.get("angle")))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "NeckServoCommandAdapter",
    "ServoDriver",
    "UnavailableNeckServoCommandAdapter",
    "build_neck_servo_adapter",
]
