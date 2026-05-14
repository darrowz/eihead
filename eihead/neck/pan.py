"""Pan-only neck protocol and state planning.

This module is intentionally hardware-free: it does not import legacy body
runtime code, open I2C devices, or call servo drivers. It only turns a requested
pan target into action/outcome-shaped dictionaries for a later hardware adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Any


PAN_PLAN_SCHEMA = "eihead.neck.pan_plan.v1"
ACTION_CONTENT_SCHEMA = "eiprotocol.head_action.content.v0.1"
OUTCOME_CONTENT_SCHEMA = "eiprotocol.execution_outcome.content.v0.1"


@dataclass(frozen=True, slots=True)
class PanNeckState:
    """Serializable neck pan state owned by native eihead code."""

    current_angle: float = 90.0
    target_angle: float = 90.0
    min_angle: float = 40.0
    max_angle: float = 140.0
    deadband: float = 2.0
    last_command_status: str = "idle"
    suppression_reason: str = ""
    last_target_x: float | None = None
    last_direction: str = "center"

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_angle": _json_number(self.current_angle),
            "target_angle": _json_number(self.target_angle),
            "min_angle": _json_number(self.min_angle),
            "max_angle": _json_number(self.max_angle),
            "deadband": _json_number(self.deadband),
            "last_command_status": self.last_command_status,
            "suppression_reason": self.suppression_reason,
            "last_target_x": _json_number(self.last_target_x),
            "last_direction": self.last_direction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PanNeckState":
        return cls(
            current_angle=_float_or_default(payload.get("current_angle"), cls.current_angle),
            target_angle=_float_or_default(payload.get("target_angle"), cls.target_angle),
            min_angle=_float_or_default(payload.get("min_angle"), cls.min_angle),
            max_angle=_float_or_default(payload.get("max_angle"), cls.max_angle),
            deadband=_float_or_default(payload.get("deadband"), cls.deadband),
            last_command_status=str(payload.get("last_command_status", "idle") or "idle"),
            suppression_reason=str(payload.get("suppression_reason", "") or ""),
            last_target_x=_optional_float(payload.get("last_target_x")),
            last_direction=str(payload.get("last_direction", "center") or "center"),
        )


@dataclass(frozen=True, slots=True)
class PanMoveCommand:
    """A hardware-free request to plan the next horizontal neck movement."""

    axis: str = "pan"
    target_angle: float | None = None
    target_x: float | None = None
    source: str = ""
    action_id: str = ""
    trace_id: str = ""
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "target_angle": _json_number(self.target_angle),
            "target_x": _json_number(self.target_x),
            "source": self.source,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
            "reason": self.reason,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PanMoveCommand":
        metadata = payload.get("metadata")
        return cls(
            axis=str(payload.get("axis", "pan") or "pan"),
            target_angle=_optional_float(payload.get("target_angle", payload.get("angle"))),
            target_x=_optional_float(payload.get("target_x")),
            source=str(payload.get("source", "") or ""),
            action_id=str(payload.get("action_id", payload.get("actionId", "")) or ""),
            trace_id=str(payload.get("trace_id", payload.get("traceId", "")) or ""),
            reason=str(payload.get("reason", "") or ""),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


class PanNeckPlanner:
    """Stateless facade for callers that prefer an object boundary."""

    def plan(
        self,
        command: PanMoveCommand | Mapping[str, Any],
        state: PanNeckState | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return plan_pan_move(command, state or PanNeckState())


def plan_pan_move(
    command: PanMoveCommand | Mapping[str, Any],
    state: PanNeckState | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the next pan plan without mutating input state or touching hardware."""

    pan_command = _coerce_command(command)
    pan_state = _coerce_state(state)
    axis = _normalize_axis(pan_command.axis)

    if axis == "tilt":
        return _unsupported_result(pan_command, pan_state, "tilt_not_supported")
    if axis != "pan":
        return _unsupported_result(pan_command, pan_state, "axis_not_supported")

    limits = _normalized_limits(pan_state)
    target = _resolve_target_angle(pan_command, limits)
    if target["status"] != "ok":
        next_state = _state_with_status(
            pan_state,
            status="invalid",
            suppression_reason=target["reason"],
        )
        return _result(
            command=pan_command,
            state=next_state,
            status="invalid",
            success=False,
            will_move=False,
            target_angle=pan_state.target_angle,
            target_x=None,
            direction="center",
            reason=target["reason"],
            details=target,
        )

    target_angle = _clamp(target["target_angle"], limits["min_angle"], limits["max_angle"])
    direction = _direction(pan_state.current_angle, target_angle)
    normalized_target_x = target["normalized_target_x"]

    if abs(target_angle - pan_state.current_angle) <= max(0.0, pan_state.deadband):
        next_state = replace(
            pan_state,
            target_angle=target_angle,
            last_command_status="suppressed",
            suppression_reason="deadband",
            last_target_x=normalized_target_x,
            last_direction=direction,
        )
        return _result(
            command=pan_command,
            state=next_state,
            status="suppressed",
            success=True,
            will_move=False,
            target_angle=target_angle,
            target_x=normalized_target_x,
            direction=direction,
            reason="deadband",
            details={
                "reason": "deadband",
                "current_angle": _json_number(pan_state.current_angle),
                "target_angle": _json_number(target_angle),
                "deadband": _json_number(pan_state.deadband),
                **_target_details(target),
            },
        )

    next_state = replace(
        pan_state,
        target_angle=target_angle,
        last_command_status="planned",
        suppression_reason="",
        last_target_x=normalized_target_x,
        last_direction=direction,
    )
    return _result(
        command=pan_command,
        state=next_state,
        status="planned",
        success=True,
        will_move=True,
        target_angle=target_angle,
        target_x=normalized_target_x,
        direction=direction,
        reason="",
        details={
            "reason": "",
            "current_angle": _json_number(pan_state.current_angle),
            "target_angle": _json_number(target_angle),
            **_target_details(target),
        },
    )


def _coerce_command(command: PanMoveCommand | Mapping[str, Any]) -> PanMoveCommand:
    if isinstance(command, PanMoveCommand):
        return command
    if isinstance(command, Mapping):
        return PanMoveCommand.from_dict(command)
    raise TypeError("command must be a PanMoveCommand or mapping")


def _coerce_state(state: PanNeckState | Mapping[str, Any]) -> PanNeckState:
    if isinstance(state, PanNeckState):
        return state
    if isinstance(state, Mapping):
        return PanNeckState.from_dict(state)
    raise TypeError("state must be a PanNeckState or mapping")


def _normalize_axis(axis: str) -> str:
    normalized = str(axis or "pan").strip().lower()
    if normalized == "yaw":
        return "pan"
    return normalized or "pan"


def _unsupported_result(command: PanMoveCommand, state: PanNeckState, reason: str) -> dict[str, Any]:
    next_state = _state_with_status(state, status="unsupported", suppression_reason=reason)
    return _result(
        command=command,
        state=next_state,
        status="unsupported",
        success=False,
        will_move=False,
        target_angle=state.target_angle,
        target_x=None,
        direction="center",
        reason=reason,
        details={"axis": command.axis, "reason": reason},
    )


def _state_with_status(state: PanNeckState, *, status: str, suppression_reason: str) -> PanNeckState:
    return replace(
        state,
        last_command_status=status,
        suppression_reason=suppression_reason,
    )


def _normalized_limits(state: PanNeckState) -> dict[str, float]:
    min_angle = _float_or_default(state.min_angle, 40.0)
    max_angle = _float_or_default(state.max_angle, 140.0)
    if min_angle > max_angle:
        min_angle, max_angle = max_angle, min_angle
    return {"min_angle": min_angle, "max_angle": max_angle}


def _resolve_target_angle(command: PanMoveCommand, limits: Mapping[str, float]) -> dict[str, Any]:
    if command.target_angle is not None:
        target_angle = _optional_float(command.target_angle)
        if target_angle is None:
            return {"status": "invalid", "reason": "invalid_target_angle"}
        return {
            "status": "ok",
            "source": "target_angle",
            "target_angle": target_angle,
            "normalized_target_x": None,
        }

    if command.target_x is None:
        return {"status": "invalid", "reason": "missing_pan_target"}

    normalized_target_x = _optional_float(command.target_x)
    if normalized_target_x is None:
        return {"status": "invalid", "reason": "invalid_target_x"}

    normalized_target_x = _clamp(normalized_target_x, 0.0, 1.0)
    min_angle = limits["min_angle"]
    max_angle = limits["max_angle"]
    return {
        "status": "ok",
        "source": "target_x",
        "target_angle": min_angle + ((max_angle - min_angle) * normalized_target_x),
        "normalized_target_x": normalized_target_x,
    }


def _target_details(target: Mapping[str, Any]) -> dict[str, Any]:
    details = {
        "target_source": target.get("source", ""),
        "normalized_target_x": _json_number(target.get("normalized_target_x")),
    }
    return details


def _result(
    *,
    command: PanMoveCommand,
    state: PanNeckState,
    status: str,
    success: bool,
    will_move: bool,
    target_angle: float,
    target_x: float | None,
    direction: str,
    reason: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    action = {
        "schema": ACTION_CONTENT_SCHEMA,
        "action_id": command.action_id,
        "action_type": "move_head",
        "target": "neck.pan",
        "axis": "pan",
        "target_angle": _json_number(target_angle),
        "target_x": _json_number(target_x),
        "direction": direction,
        "source": command.source,
        "trace_id": command.trace_id,
        "params": {
            "axis": "pan",
            "target_angle": _json_number(target_angle),
            "target_x": _json_number(target_x),
            "direction": direction,
        },
        "metadata": _json_safe(command.metadata),
    }
    outcome = {
        "schema": OUTCOME_CONTENT_SCHEMA,
        "outcome_id": f"{command.action_id}:pan-plan" if command.action_id else "",
        "action_id": command.action_id,
        "action_type": "move_head",
        "success": success,
        "status": status,
        "did_what": _did_what(status, will_move),
        "errors": [] if success else [{"code": reason or status, "message": reason or status}],
        "details": _json_safe(details),
    }
    return {
        "schema": PAN_PLAN_SCHEMA,
        "status": status,
        "success": success,
        "will_move": will_move,
        "reason": reason,
        "trace_id": command.trace_id,
        "action": action,
        "outcome": outcome,
        "state": state.to_dict(),
    }


def _did_what(status: str, will_move: bool) -> list[str]:
    if will_move:
        return ["planned_pan_move"]
    if status == "suppressed":
        return ["suppressed_pan_move"]
    return []


def _direction(current_angle: float, target_angle: float) -> str:
    if target_angle > current_angle:
        return "right"
    if target_angle < current_angle:
        return "left"
    return "center"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _float_or_default(value: Any, default: float) -> float:
    result = _optional_float(value)
    return default if result is None else result


def _json_number(value: Any) -> int | float | None:
    number = _optional_float(value)
    if number is None:
        return None
    if number.is_integer():
        return int(number)
    return number


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return _json_number(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "ACTION_CONTENT_SCHEMA",
    "OUTCOME_CONTENT_SCHEMA",
    "PAN_PLAN_SCHEMA",
    "PanMoveCommand",
    "PanNeckPlanner",
    "PanNeckState",
    "plan_pan_move",
]
