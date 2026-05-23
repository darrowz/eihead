"""Truthful neck diagnostics for the eihead native monitor."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
import math
from typing import Any


NECK_MONITOR_SCHEMA = "eihead.monitor.neck.v1"
PAN_PLAN_SCHEMA = "eihead.neck.pan_plan.v1"

NATIVE_NECK_ATTRS = (
    "neck_realtime",
    "neck_status",
    "latest_neck_realtime",
    "latest_neck_status",
    "neck",
)
ACTION_OUTCOME_ATTRS = (
    "last_action_outcome",
    "latest_action_outcome",
    "action_outcome",
)
RECENT_ACTION_ATTRS = (
    "recent_actions",
    "action_log",
    "actions_log",
    "recent_action_log",
    "execution_log",
)


JsonObject = dict[str, Any]


def build_neck_diagnostics_from_app(app: Any, *, timestamp: float) -> JsonObject:
    """Build the monitor payload from native neck hooks, snapshots, or outcomes."""

    native = _from_native_hooks(app, timestamp=timestamp)
    if native is not None:
        return native

    snapshot = _from_snapshot(app, timestamp=timestamp)
    if snapshot is not None:
        return snapshot

    direct_plan = _from_direct_plan(app, timestamp=timestamp)
    if direct_plan is not None:
        return direct_plan

    outcome = _from_action_outcome_attrs(app, timestamp=timestamp)
    if outcome is not None:
        return outcome

    recent = _from_recent_actions(app, timestamp=timestamp)
    if recent is not None:
        return recent

    return _not_wired_payload(timestamp)


def _from_native_hooks(app: Any, *, timestamp: float) -> JsonObject | None:
    for attr_name in NATIVE_NECK_ATTRS:
        if not hasattr(app, attr_name):
            continue
        raw = _read_attr_payload(app, attr_name)
        payload = _coerce_mapping(raw)
        if payload is None:
            continue
        return _payload_from_neck_data(payload, timestamp=timestamp, source=attr_name)
    return None


def _from_snapshot(app: Any, *, timestamp: float) -> JsonObject | None:
    snapshot_fn = getattr(app, "snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = _coerce_mapping(snapshot_fn())
    if snapshot is None:
        return None

    for key in ("neck", "neck_status", "neck_realtime"):
        nested = _coerce_mapping(snapshot.get(key))
        if nested is not None:
            return _payload_from_neck_data(nested, timestamp=timestamp, source=f"snapshot.{key}")

    for key in ("neck_plan", "latest_neck_plan"):
        plan = _coerce_mapping(snapshot.get(key))
        if _looks_like_neck_plan(plan):
            return _payload_from_plan(plan, timestamp=timestamp, source=f"snapshot.{key}")

    for key in ("neck_pan_state", "neck_state", "pan_neck_state"):
        state = _coerce_mapping(snapshot.get(key))
        if state is not None:
            return _payload_from_neck_data(
                {"pan": state},
                timestamp=timestamp,
                source=f"snapshot.{key}",
            )

    body_runtime = _coerce_mapping(snapshot.get("body_runtime"))
    if body_runtime is None:
        return None
    for key in ("neck", "neck_status", "neck_realtime"):
        nested = _coerce_mapping(body_runtime.get(key))
        if nested is not None:
            return _payload_from_neck_data(
                nested,
                timestamp=timestamp,
                source=f"snapshot.body_runtime.{key}",
            )

    organs = _coerce_mapping(body_runtime.get("organs"))
    neck_organ = _coerce_mapping(organs.get("neck")) if organs is not None else None
    body_neck = _neck_data_from_body_runtime_organ(neck_organ)
    if body_neck is not None:
        return _payload_from_neck_data(
            body_neck,
            timestamp=timestamp,
            source="snapshot.body_runtime.organs.neck",
        )
    return None


def _from_direct_plan(app: Any, *, timestamp: float) -> JsonObject | None:
    for attr_name in ("neck_plan", "latest_neck_plan"):
        if not hasattr(app, attr_name):
            continue
        plan = _coerce_mapping(_read_attr_payload(app, attr_name))
        if _looks_like_neck_plan(plan):
            return _payload_from_plan(plan, timestamp=timestamp, source=attr_name)
    return None


def _from_action_outcome_attrs(app: Any, *, timestamp: float) -> JsonObject | None:
    for attr_name in ACTION_OUTCOME_ATTRS:
        if not hasattr(app, attr_name):
            continue
        outcome = _coerce_mapping(_read_attr_payload(app, attr_name))
        extracted = _extract_from_action_outcome(outcome)
        if extracted is not None:
            plan, servo, source = extracted
            return _payload_from_plan(plan, servo=servo, timestamp=timestamp, source=f"{attr_name}.{source}")
    return None


def _from_recent_actions(app: Any, *, timestamp: float) -> JsonObject | None:
    for attr_name in RECENT_ACTION_ATTRS:
        if not hasattr(app, attr_name):
            continue
        raw = _read_attr_payload(app, attr_name)
        actions = _action_items(raw)
        for action in actions:
            extracted = _extract_from_action_outcome(action)
            if extracted is None:
                continue
            plan, servo, source = extracted
            return _payload_from_plan(plan, servo=servo, timestamp=timestamp, source=f"{attr_name}.{source}")
    return None


def _payload_from_neck_data(payload: Mapping[str, Any], *, timestamp: float, source: str) -> JsonObject:
    if _looks_like_neck_plan(payload):
        return _payload_from_plan(payload, timestamp=timestamp, source=source)

    plan = _coerce_mapping(payload.get("neck_plan") or payload.get("plan"))
    servo = _coerce_mapping(payload.get("neck_servo") or payload.get("servo"))
    if _looks_like_neck_plan(plan):
        return _payload_from_plan(plan, servo=servo, timestamp=timestamp, source=f"{source}.neck_plan")

    pan = _coerce_mapping(payload.get("pan") or payload.get("yaw") or payload.get("neck_pan_state")) or payload
    neck_reframe = _coerce_mapping(payload.get("neck_reframe"))
    current_angle = _json_number(_first_present(pan, payload, keys=("current_angle", "current", "angle")))
    target_angle = _json_number(_first_present(pan, payload, keys=("target_angle", "target")))
    will_move = _optional_bool(_first_present(pan, payload, keys=("will_move", "moving")))
    raw_reason = _string_or_default(
        _first_present(pan, payload, keys=("suppression_reason", "reason", "last_suppression_reason")),
        "",
    )
    last_status = _string_or_default(
        _first_present(pan, payload, keys=("last_command_status", "command_status", "status")),
        "",
    )
    suppressed = _optional_bool(_first_present(pan, payload, keys=("suppressed", "is_suppressed")))
    if suppressed is None:
        suppressed = last_status == "suppressed" or bool(raw_reason and raw_reason != "none")
    suppression_reason = raw_reason or ("none" if suppressed is False else "unknown")

    normalized_servo = _normalize_servo(servo)
    axis_support = _axis_support_from_payload(payload, native_data=True)
    status = _status_for_neck_data(
        requested=_string_or_default(payload.get("status"), ""),
        servo_status=normalized_servo["status"],
        suppressed=suppressed,
        will_move=will_move,
        has_angles=current_angle is not None or target_angle is not None,
    )
    return _base_payload(
        timestamp=timestamp,
        source=source,
        status=status,
        current_angle=current_angle,
        target_angle=target_angle,
        will_move=will_move,
        suppressed=suppressed,
        suppression_reason=suppression_reason,
        servo=normalized_servo,
        axis_support=axis_support,
        plan=plan,
        neck_reframe=neck_reframe,
    )


def _payload_from_plan(
    plan: Mapping[str, Any],
    *,
    timestamp: float,
    source: str,
    servo: Mapping[str, Any] | None = None,
) -> JsonObject:
    state = _coerce_mapping(plan.get("state")) or {}
    action = _coerce_mapping(plan.get("action")) or {}
    params = _coerce_mapping(action.get("params")) or {}
    outcome = _coerce_mapping(plan.get("outcome")) or {}
    outcome_details = _coerce_mapping(outcome.get("details")) or {}

    current_angle = _json_number(_first_present(state, outcome_details, keys=("current_angle", "current")))
    target_angle = _json_number(
        _first_present(action, params, state, plan, keys=("target_angle", "angle", "target"))
    )
    plan_status = _string_or_default(plan.get("status"), "unknown")
    will_move = _optional_bool(plan.get("will_move"))
    if will_move is None:
        will_move = plan_status == "planned"

    raw_reason = _string_or_default(
        plan.get("reason")
        or state.get("suppression_reason")
        or outcome_details.get("reason")
        or outcome.get("status"),
        "",
    )
    suppressed = plan_status == "suppressed"
    suppression_reason = raw_reason or ("none" if not suppressed else "unknown")
    normalized_servo = _normalize_servo(servo)
    axis_support = _axis_support_for_plan(plan)
    status = _status_for_plan(plan_status, normalized_servo["status"])

    payload = _base_payload(
        timestamp=timestamp,
        source=source,
        status=status,
        current_angle=current_angle,
        target_angle=target_angle,
        will_move=will_move,
        suppressed=suppressed,
        suppression_reason=suppression_reason,
        servo=normalized_servo,
        axis_support=axis_support,
        plan=plan,
    )
    payload["plan_status"] = plan_status
    return payload


def _neck_data_from_body_runtime_organ(organ: Mapping[str, Any] | None) -> JsonObject | None:
    if organ is None:
        return None

    subfunctions = _coerce_mapping(organ.get("subfunctions")) or {}
    motor = _coerce_mapping(subfunctions.get("motor")) or {}
    tracking = _coerce_mapping(subfunctions.get("tracking")) or {}
    motor_details = _coerce_mapping(motor.get("details")) or {}
    tracking_details = _coerce_mapping(tracking.get("details")) or {}
    neck_control = _coerce_mapping(tracking_details.get("neck_control"))

    if neck_control is None and not motor and not tracking:
        return None

    status = _string_or_default(
        organ.get("health")
        or tracking_details.get("status")
        or tracking.get("health")
        or motor_details.get("status")
        or motor.get("health"),
        "unknown",
    )
    pan: JsonObject = {}
    if neck_control is not None:
        pan = {
            "current_angle": _first_present(neck_control, keys=("last_angle", "current_angle", "angle")),
            "target_angle": _first_present(
                neck_control,
                keys=("desired_angle", "target_angle", "last_commanded_angle"),
            ),
            "moving": _string_or_default(neck_control.get("state"), "") not in {"", "idle", "stable"},
            "suppression_reason": _string_or_default(
                _first_present(
                    neck_control,
                    keys=("last_suppression_reason", "suppression_reason", "reason"),
                ),
                "none",
            ),
            "state": neck_control.get("state"),
        }

    motor_status = _string_or_default(
        motor_details.get("status") or motor.get("health") or organ.get("health"),
        "unknown",
    ).strip().lower()
    motor_device_details = _coerce_mapping(motor_details.get("details")) or {}
    servo: JsonObject = {
        "status": motor_status,
        "available": motor_status in {"ok", "healthy", "ready", "wired", "online"},
        "reason": _string_or_default(motor_details.get("reason"), ""),
    }
    for key in ("device", "device_exists"):
        if key in motor_device_details:
            servo[key] = motor_device_details.get(key)

    return {
        "status": status,
        "pan": pan,
        "servo": servo,
        "axis_support": {
            "pan": {"supported": True, "status": "supported"},
            "tilt": {
                "supported": False,
                "status": "unsupported",
                "reason": "tilt_not_supported",
            },
        },
    }


def _not_wired_payload(timestamp: float) -> JsonObject:
    axis_support = _default_axis_support(native_data=False)
    return _base_payload(
        timestamp=timestamp,
        source=None,
        status="not_wired",
        current_angle=None,
        target_angle=None,
        will_move=None,
        suppressed=None,
        suppression_reason="unknown",
        servo=_normalize_servo(None),
        axis_support=axis_support,
        plan=None,
        not_wired=True,
        message="runtime app does not expose native neck diagnostics",
    )


def _base_payload(
    *,
    timestamp: float,
    source: str | None,
    status: str,
    current_angle: int | float | None,
    target_angle: int | float | None,
    will_move: bool | None,
    suppressed: bool | None,
    suppression_reason: str,
    servo: JsonObject,
    axis_support: JsonObject,
    plan: Mapping[str, Any] | None,
    neck_reframe: Mapping[str, Any] | None = None,
    not_wired: bool = False,
    message: str | None = None,
) -> JsonObject:
    pan_support = _coerce_mapping(axis_support.get("pan")) or {}
    tilt_support = _coerce_mapping(axis_support.get("tilt")) or {}
    wired = _wired_for_status(status, not_wired=not_wired)
    motion_evidence = _motion_evidence_from_servo(servo)
    payload: JsonObject = {
        "schema": NECK_MONITOR_SCHEMA,
        "runtime": "eihead",
        "status": status,
        "wired": wired,
        "not_wired": not_wired,
        "source": source,
        "captured_at_ts": timestamp,
        "angle_state": "known" if current_angle is not None or target_angle is not None else "unknown",
        "current_angle": current_angle,
        "target_angle": target_angle,
        "will_move": will_move,
        "suppressed": suppressed,
        "suppression_reason": suppression_reason,
        "servo_status": servo["status"],
        "servo": servo,
        "motion_verified": motion_evidence["verified"],
        "motion_evidence": motion_evidence,
        "axis_support": axis_support,
        "pan": {
            "supported": pan_support.get("supported"),
            "status": pan_support.get("status", "unknown"),
            "current_angle": current_angle,
            "target_angle": target_angle,
            "will_move": will_move,
            "suppressed": suppressed,
            "suppression_reason": suppression_reason,
        },
        "tilt": tilt_support,
        "readiness_message": message or _readiness_message(status, servo, not_wired=not_wired),
    }
    if plan is not None:
        payload["neck_plan"] = _json_ready(plan)
    if neck_reframe is not None:
        payload["neck_reframe"] = _json_ready(neck_reframe)
    return payload


def _status_for_neck_data(
    *,
    requested: str,
    servo_status: str,
    suppressed: bool | None,
    will_move: bool | None,
    has_angles: bool,
) -> str:
    requested = requested.strip().lower()
    if requested in {"not_wired", "unknown", "unsupported", "invalid", "degraded", "suppressed"}:
        return requested
    if requested in {"wired", "healthy", "ready", "tracking_ready", "online"} and (
        has_angles or servo_status in {"ok", "healthy", "ready", "wired", "online"}
    ):
        return "wired"
    if servo_status in {"unavailable", "error", "invalid", "unsupported"}:
        return "degraded"
    if suppressed is True:
        return "suppressed"
    if will_move is True:
        return "planned"
    if has_angles:
        return "wired"
    return "unknown"


def _status_for_plan(plan_status: str, servo_status: str) -> str:
    normalized = plan_status.strip().lower() or "unknown"
    if normalized in {"unsupported", "invalid", "suppressed"}:
        return normalized
    if servo_status in {"unavailable", "error", "invalid", "unsupported"}:
        return "degraded"
    if normalized == "planned":
        return "planned"
    if normalized in {"ok", "accepted"}:
        return "wired"
    return normalized


def _wired_for_status(status: str, *, not_wired: bool) -> bool:
    if not_wired:
        return False
    return status in {"wired", "planned", "suppressed"}


def _readiness_message(status: str, servo: Mapping[str, Any], *, not_wired: bool) -> str:
    if not_wired:
        return "native neck diagnostics are not wired"
    servo_status = _string_or_default(servo.get("status"), "unknown")
    servo_reason = _string_or_default(servo.get("reason"), "")
    if servo_status == "unavailable":
        return servo_reason or "neck servo is unavailable"
    if status == "unsupported":
        return "tilt axis is unsupported by pan-only neck"
    if status == "suppressed":
        return "neck pan move suppressed before servo command"
    if status == "degraded":
        return "native neck state is present but servo is not available"
    return f"neck status is {status}"


def _motion_evidence_from_servo(servo: Mapping[str, Any]) -> JsonObject:
    driver = _coerce_mapping(servo.get("driver"))
    verified = _optional_bool(servo.get("motion_verified"))
    if verified is None and driver is not None:
        verified = _optional_bool(driver.get("motion_verified"))
    if verified is None:
        verified = _optional_bool(servo.get("hardware_verified"))
    if verified is None and driver is not None:
        verified = _optional_bool(driver.get("hardware_verified"))

    evidence = _string_or_default(servo.get("motion_evidence"), "")
    if not evidence and driver is not None:
        evidence = _string_or_default(driver.get("motion_evidence"), "")
    servo_id = _first_present(servo, driver or {}, keys=("servo_id", "servoId"))
    axis = "pan" if verified is True else "unknown"
    source = "operator_observed" if evidence.startswith("operator_observed") else ("status" if evidence else "unknown")
    summary = (
        "S1 horizontal pan servo was observed moving"
        if evidence == "operator_observed_s1_pan_servo"
        else evidence
    )
    return {
        "verified": verified,
        "status": "verified" if verified is True else ("unverified" if verified is False else "unknown"),
        "source": source,
        "axis": axis,
        "servo_id": _json_number(servo_id),
        "evidence": evidence or None,
        "summary": summary or ("no motion evidence reported" if verified is not True else "motion verified"),
    }


def _axis_support_from_payload(payload: Mapping[str, Any], *, native_data: bool) -> JsonObject:
    raw = _coerce_mapping(payload.get("axis_support") or payload.get("axes"))
    if raw is None:
        return _default_axis_support(native_data=native_data)

    axis_support = _default_axis_support(native_data=native_data)
    for axis in ("pan", "yaw", "tilt"):
        value = raw.get(axis)
        normalized_axis = "pan" if axis == "yaw" else axis
        normalized = _normalize_axis_support(value, axis=normalized_axis, native_data=native_data)
        axis_support[normalized_axis] = normalized

    tilt = _coerce_mapping(axis_support.get("tilt")) or {}
    if tilt.get("supported") is not True:
        axis_support["tilt"] = {
            "supported": False,
            "status": "unsupported",
            "reason": _string_or_default(tilt.get("reason"), "tilt_not_supported"),
        }
    return axis_support


def _axis_support_for_plan(plan: Mapping[str, Any]) -> JsonObject:
    axis_support = _default_axis_support(native_data=True)
    status = _string_or_default(plan.get("status"), "")
    reason = _string_or_default(plan.get("reason"), "")
    if status == "unsupported" and reason == "tilt_not_supported":
        axis_support["tilt"] = {
            "supported": False,
            "status": "unsupported",
            "reason": "tilt_not_supported",
        }
    return axis_support


def _default_axis_support(*, native_data: bool) -> JsonObject:
    pan_status = "supported" if native_data else "unknown"
    pan_supported = True if native_data else None
    return {
        "pan": {"supported": pan_supported, "status": pan_status},
        "tilt": {
            "supported": False,
            "status": "unsupported",
            "reason": "tilt_not_supported",
        },
    }


def _normalize_axis_support(value: Any, *, axis: str, native_data: bool) -> JsonObject:
    if isinstance(value, Mapping):
        supported = _optional_bool(value.get("supported"))
        status = _string_or_default(value.get("status"), "")
        reason = _string_or_default(value.get("reason"), "")
    else:
        supported = _optional_bool(value)
        status = ""
        reason = ""

    if axis == "tilt":
        if supported is True:
            return {"supported": True, "status": status or "supported"}
        return {"supported": False, "status": "unsupported", "reason": reason or "tilt_not_supported"}

    if supported is None and native_data:
        supported = True
    return {
        "supported": supported,
        "status": status or ("supported" if supported is True else "unknown"),
    }


def _normalize_servo(value: Mapping[str, Any] | None) -> JsonObject:
    if value is None:
        return {"status": "unknown", "available": None, "reason": "unknown"}

    status = _string_or_default(value.get("status"), "unknown").strip().lower() or "unknown"
    available = _optional_bool(value.get("available"))
    if available is None:
        if status in {"ok", "healthy", "ready", "wired", "online", "suppressed", "planned", "accepted"}:
            available = True
        elif status in {"unavailable", "error", "invalid", "unsupported"}:
            available = False
    reason = _string_or_default(value.get("reason"), "")
    servo: JsonObject = {"status": status, "available": available, "reason": reason}
    for key, item in value.items():
        key_text = str(key)
        if key_text in servo:
            continue
        servo[key_text] = _json_ready(item)
    return servo


def _extract_from_action_outcome(outcome: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, str] | None:
    if outcome is None:
        return None

    plan = _coerce_mapping(outcome.get("neck_plan"))
    servo = _coerce_mapping(outcome.get("neck_servo"))
    if _looks_like_neck_plan(plan):
        return plan, servo, "neck_plan"

    details = _coerce_mapping(outcome.get("details"))
    if details is not None:
        plan = _coerce_mapping(details.get("neck_plan"))
        servo = _coerce_mapping(details.get("neck_servo"))
        if _looks_like_neck_plan(plan):
            return plan, servo, "details.neck_plan"

    nested = _coerce_mapping(outcome.get("action_outcome"))
    extracted = _extract_from_action_outcome(nested)
    if extracted is None:
        return None
    plan, servo, source = extracted
    return plan, servo, f"action_outcome.{source}"


def _action_items(raw: Any) -> list[Mapping[str, Any]]:
    payload = _coerce_mapping(raw)
    if payload is not None:
        for key in ("actions", "recent_actions", "items", "events"):
            items = payload.get(key)
            if items is not None:
                return _mapping_items(items)
        return [payload]
    return _mapping_items(raw)


def _mapping_items(raw: Any) -> list[Mapping[str, Any]]:
    if raw is None or isinstance(raw, (str, bytes, Mapping)):
        item = _coerce_mapping(raw)
        return [item] if item is not None else []
    if not isinstance(raw, Iterable):
        item = _coerce_mapping(raw)
        return [item] if item is not None else []
    return [item for item in (_coerce_mapping(entry) for entry in raw) if item is not None]


def _read_attr_payload(app: Any, attr_name: str) -> Any:
    source = getattr(app, attr_name)
    return source() if callable(source) else source


def _looks_like_neck_plan(value: Mapping[str, Any] | None) -> bool:
    if value is None:
        return False
    if value.get("schema") == PAN_PLAN_SCHEMA:
        return True
    return "will_move" in value and ("action" in value or "state" in value)


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return payload
    if is_dataclass(value):
        return asdict(value)
    return None


def _first_present(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping.get(key) is not None:
                return mapping.get(key)
    return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on", "supported"}:
            return True
        if normalized in {"false", "no", "0", "off", "unsupported"}:
            return False
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    return None


def _json_number(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _json_ready(value: Any) -> Any:
    mapping = _coerce_mapping(value)
    if mapping is not None:
        return {str(key): _json_ready(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


__all__ = [
    "NECK_MONITOR_SCHEMA",
    "build_neck_diagnostics_from_app",
]
