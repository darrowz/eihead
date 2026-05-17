"""Servo command adapter for the honjia neck yaw axis.

The adapter owns no hardware policy.  It only translates an already-made yaw
decision into the tiny driver call expected by the current honjia servo driver.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any, Protocol


class ServoDriver(Protocol):
    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> Any:
        """Move one servo and return the driver's payload/status."""


class NeckServoCommandAdapter:
    """Apply yaw decisions to an injected servo driver."""

    def __init__(self, driver: ServoDriver, *, servo_id: int = 1) -> None:
        self._driver = driver
        self.servo_id = int(servo_id)

    def status(self) -> dict[str, Any]:
        driver_status = _driver_status(self._driver)
        driver_available = driver_status.get("available")
        driver_status_text = str(driver_status.get("status", "") or "").strip().lower()
        available = driver_available is not False and driver_status_text not in {"unavailable", "error", "missing"}
        motion_verified = _optional_bool(driver_status.get("motion_verified"))
        return {
            "status": "ready" if available else "unavailable",
            "available": available,
            "reason": "neck_servo_adapter_ready" if available else str(driver_status.get("reason") or "driver_unavailable"),
            "servo_id": self.servo_id,
            "hardware_verified": _optional_bool(driver_status.get("hardware_verified")) is True,
            "motion_verified": motion_verified is True,
            "motion_evidence": str(driver_status.get("motion_evidence") or ""),
            "driver": driver_status,
        }

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

    def status(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "available": False,
            "reason": self.reason,
            "node_id": self.node_id,
            "hardware_verified": False,
        }


class RaspbotServoDriver:
    """Minimal native Raspbot I2C servo driver for honjia pan control."""

    def __init__(
        self,
        *,
        bus: int = 1,
        addr: int = 0x2B,
        servo_id: int = 1,
        enabled: bool = True,
        mock: bool = False,
        hardware_verified: bool = False,
        motion_verified: bool = False,
        motion_evidence: str = "",
    ) -> None:
        self.bus = int(bus)
        self.addr = int(addr)
        self.servo_id = int(servo_id)
        self.enabled = bool(enabled)
        self.mock = bool(mock)
        self.hardware_verified = bool(hardware_verified)
        self.motion_verified = bool(motion_verified)
        self.motion_evidence = str(motion_evidence or "")
        self.device_path = f"/dev/i2c-{self.bus}"
        self.last_command: tuple[int, int] | None = None

    def status(self) -> dict[str, Any]:
        device_exists = os.path.exists(self.device_path)
        return {
            "status": "ready" if self.enabled and (device_exists or self.mock) else "unavailable",
            "available": self.enabled and (device_exists or self.mock),
            "reason": "raspbot_i2c_ready"
            if self.enabled and (device_exists or self.mock)
            else "missing_i2c_device",
            "bus": self.bus,
            "addr": self.addr,
            "servo_id": self.servo_id,
            "device": self.device_path,
            "device_exists": device_exists,
            "mock": self.mock,
            "hardware_verified": self.hardware_verified,
            "motion_verified": self.motion_verified,
            "motion_evidence": self.motion_evidence,
        }

    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> list[int]:
        target_servo = self.servo_id if servo_id is None else int(servo_id)
        clipped_angle = max(0, min(180, int(angle)))
        if target_servo == 2 and clipped_angle > 110:
            clipped_angle = 110
        self.last_command = (target_servo, clipped_angle)
        payload = [target_servo & 0xFF, clipped_angle & 0xFF]
        if self.mock or not self.enabled:
            return payload
        if not os.path.exists(self.device_path):
            raise RuntimeError(f"missing i2c device: {self.device_path}")
        bus = _open_smbus(self.bus)
        try:
            bus.write_i2c_block_data(self.addr, 0x02, payload)
        finally:
            close = getattr(bus, "close", None)
            if callable(close):
                close()
        return payload


def build_neck_servo_adapter(
    *,
    node_id: str,
    driver: ServoDriver | None = None,
    servo_id: int = 1,
    bus: int = 1,
    addr: int = 0x2B,
    enabled: bool = True,
    mock: bool = False,
    hardware_verified: bool = False,
    motion_verified: bool = False,
    motion_evidence: str = "",
) -> NeckServoCommandAdapter | UnavailableNeckServoCommandAdapter:
    """Return a narrow injected-driver adapter only on honjia."""

    normalized_node_id = str(node_id or "")
    if normalized_node_id != "honjia":
        return UnavailableNeckServoCommandAdapter(
            node_id=normalized_node_id,
            reason="neck_servo_unavailable_off_honjia",
        )
    if driver is None:
        driver = RaspbotServoDriver(
            bus=bus,
            addr=addr,
            servo_id=servo_id,
            enabled=enabled,
            mock=mock,
            hardware_verified=hardware_verified,
            motion_verified=motion_verified,
            motion_evidence=motion_evidence,
        )
    return NeckServoCommandAdapter(driver, servo_id=servo_id)


def _open_smbus(bus: int) -> Any:
    try:
        import smbus2  # type: ignore

        return smbus2.SMBus(bus)
    except Exception as smbus2_exc:  # pragma: no cover - depends on honjia packages.
        try:
            import smbus  # type: ignore

            return smbus.SMBus(bus)
        except Exception as smbus_exc:  # pragma: no cover - depends on honjia packages.
            raise RuntimeError(f"smbus2/smbus unavailable: smbus2={smbus2_exc}; smbus={smbus_exc}") from smbus_exc


def _driver_status(driver: ServoDriver) -> dict[str, Any]:
    status = getattr(driver, "status", None)
    if callable(status):
        payload = status()
        if isinstance(payload, Mapping):
            return {str(key): _json_safe(value) for key, value in payload.items()}
    return {"status": "ready", "driver": driver.__class__.__name__}


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


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


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
    "RaspbotServoDriver",
    "ServoDriver",
    "UnavailableNeckServoCommandAdapter",
    "build_neck_servo_adapter",
]
