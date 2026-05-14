"""Subprocess-backed driver."""

from __future__ import annotations

import json
import subprocess
import time

from eibrain.infra.config import DriverConfig

from .base import DriverResult


class CommandDriver:
    def __init__(self, config: DriverConfig) -> None:
        self.config = config

    def heartbeat(self) -> DriverResult:
        started = time.perf_counter()
        if not self.config.command:
            return DriverResult(
                status="unavailable",
                details={"reason": "missing_command", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
            )
        health_command = self.config.extra.get("health_command")
        if not health_command:
            return DriverResult(
                status="healthy",
                details={"driver": "command", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
            )
        command = [str(part) for part in health_command]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )
        except FileNotFoundError:
            return DriverResult(
                status="unavailable",
                details={
                    "reason": "command_not_found",
                    "command": command,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        except subprocess.TimeoutExpired:
            return DriverResult(
                status="degraded",
                details={
                    "reason": "command_timeout",
                    "command": command,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        details = {
            "driver": "command",
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "returncode": completed.returncode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        parsed = self._parse_json_payload(completed.stdout)
        if isinstance(parsed, dict):
            details.update(parsed)
        status = str(details.get("status", "healthy" if completed.returncode == 0 else "degraded"))
        return DriverResult(status=status, details=details)

    def invoke(self, operation: str, payload: dict[str, object]) -> DriverResult:
        started = time.perf_counter()
        if not self.config.command:
            return DriverResult(
                status="unavailable",
                details={"reason": "missing_command", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)},
            )
        try:
            completed = subprocess.run(
                self.config.command,
                input=json.dumps({"operation": operation, "payload": payload}),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )
        except FileNotFoundError:
            return DriverResult(
                status="error",
                details={
                    "reason": "command_not_found",
                    "command": self.config.command,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        except subprocess.TimeoutExpired:
            return DriverResult(
                status="error",
                details={
                    "reason": "command_timeout",
                    "command": self.config.command,
                    "operation": operation,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        details = {
            "operation": operation,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "returncode": completed.returncode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        parsed = self._parse_json_payload(completed.stdout)
        if isinstance(parsed, dict):
            details.update(parsed)
        status = "ok" if completed.returncode == 0 else "error"
        parsed_status = details.get("status")
        if isinstance(parsed_status, str):
            status = parsed_status
        return DriverResult(
            status=status,
            details=details,
        )

    @staticmethod
    def _parse_json_payload(stdout: str) -> dict[str, object] | None:
        payload = (stdout or "").strip()
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
