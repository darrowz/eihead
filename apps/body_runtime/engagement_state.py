"""Shared engagement gate for honjia body runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


DEFAULT_ENGAGEMENT_STATE_PATH = Path(tempfile.gettempdir()) / "eibrain-vision" / "engagement.json"


class EngagementStateWriter:
    def __init__(self, path: str | Path = DEFAULT_ENGAGEMENT_STATE_PATH) -> None:
        self.path = Path(path)

    def write(
        self,
        *,
        conversation_active: bool,
        phase: str,
        reason: str = "",
        security_mode: bool = False,
    ) -> Path:
        payload = {
            "conversation_active": bool(conversation_active),
            "phase": str(phase or "idle"),
            "reason": str(reason or ""),
            "security_mode": bool(security_mode),
            "updated_at_ts": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            Path(tmp_name).replace(self.path)
        except Exception:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            finally:
                raise
        return self.path


class EngagementStateReader:
    def __init__(
        self,
        path: str | Path = DEFAULT_ENGAGEMENT_STATE_PATH,
        *,
        default_active: bool = False,
        security_mode: bool = False,
    ) -> None:
        self.path = Path(path)
        self.default_active = default_active
        self.security_mode = security_mode

    def read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {
                "conversation_active": self.default_active,
                "phase": "unknown",
                "reason": "missing_engagement_state",
                "security_mode": self.security_mode,
            }
        if not isinstance(payload, dict):
            return {
                "conversation_active": self.default_active,
                "phase": "invalid",
                "reason": "invalid_engagement_state",
                "security_mode": self.security_mode,
            }
        payload.setdefault("conversation_active", self.default_active)
        payload["security_mode"] = bool(payload.get("security_mode") or self.security_mode)
        return payload

    def should_run_vision(self) -> bool:
        state = self.read()
        return bool(state.get("security_mode") or state.get("conversation_active"))
