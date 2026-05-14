"""Minimal Raspbot servo driver for honjia gimbal control."""

from __future__ import annotations

import os
from typing import Any


def _open_smbus(bus: int) -> Any:
    try:
        import smbus2  # type: ignore

        return smbus2.SMBus(bus)
    except Exception as smbus2_exc:  # pragma: no cover - depends on honjia packages
        try:
            import smbus  # type: ignore

            return smbus.SMBus(bus)
        except Exception as smbus_exc:  # pragma: no cover - depends on honjia packages
            raise RuntimeError(f"smbus2/smbus unavailable: smbus2={smbus2_exc}; smbus={smbus_exc}") from smbus_exc


class RaspbotDriver:
    def __init__(self, *, bus: int = 1, addr: int = 0x2B, servo_id: int = 1, enabled: bool = True, mock: bool = False) -> None:
        self.bus = bus
        self.addr = addr
        self.servo_id = servo_id
        self.enabled = enabled
        self.mock = mock
        self.device_path = f"/dev/i2c-{self.bus}"
        self.last_command: tuple[int, int] | None = None

    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> list[int]:
        target_servo = self.servo_id if servo_id is None else servo_id
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
