"""Mouth organ implementation.

Boundary rule:
- keep playback execution and runtime telemetry here;
- keep conversation/session policy in `eibrain.eihead.eivoice_runtime`.

This organ only reports mouth heartbeat and executes playback actions.
"""

from __future__ import annotations

import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.body.organs.base import BaseOrgan
from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
from eibrain.protocol.outcomes import ActionExecuted, SpeechPlaybackCompleted


class MouthOrgan(BaseOrgan):
    name = "mouth"
    subfunction_names = ("tts_plan", "tts_playback")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._last_plan: dict[str, object] | None = None
        self._last_playback: dict[str, object] | None = None

    def passive_heartbeat(self) -> OrganHealth:
        plan_details = {"driver": self._driver_kind("tts_plan"), "status": "live_probe_skipped"}
        playback_details = {"driver": self._driver_kind("tts_playback"), "status": "live_probe_skipped"}
        if self._driver_kind("tts_plan") != "noop":
            plan_details.update(self._voice_config_details())
        if self._driver_kind("tts_playback") != "noop":
            playback_details.update(self._voice_config_details())
        if self._last_plan is not None:
            plan_details.update(self._last_plan)
        if self._last_playback is not None:
            playback_details.update(self._last_playback)
        subfunctions = {
            "tts_plan": SubfunctionHealth(name="tts_plan", health="healthy", details=plan_details),
            "tts_playback": SubfunctionHealth(name="tts_playback", health="healthy", details=playback_details),
        }
        return OrganHealth(organ=self.name, health="healthy", subfunctions=subfunctions)

    def heartbeat(self) -> OrganHealth:
        # Boundary: this health pass aggregates lower-level subfunction probes and
        # cached playback telemetry; it does not make orchestration decisions.
        plan_state = self._tts_plan_health()
        playback_state = self._tts_playback_health()
        subfunctions = {
            "tts_plan": plan_state,
            "tts_playback": playback_state,
        }
        health = self._derive_health(state.health for state in subfunctions.values())
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def supports_action(self, action) -> bool:
        return isinstance(action, (PlaySpeechAction, StopSpeechAction))

    def handle_action(self, action):
        # Boundary: session policy (cancelability, interruption sequencing) is not
        # owned by this organ; it only executes playback actions deterministically.
        if isinstance(action, PlaySpeechAction):
            started_at = time.time()
            base_details = {
                "session_id": action.session_id,
                "actor_id": action.actor_id,
                "target_id": action.target_id,
                "text_preview": self._preview_text(action.text),
                "text_char_count": len(action.text),
            }
            self._last_plan = {
                **base_details,
                "status": "planned",
                "busy": False,
                "started_at": started_at,
                "finished_at": started_at,
                "elapsed_ms": 0.0,
                "planned_at_ts": action.ts or time.time(),
            }
            self._last_playback = {
                **self._voice_config_details(),
                **base_details,
                "status": "starting",
                "busy": True,
                "reason": "play_requested",
                "last_error": None,
                "started_at": started_at,
                "finished_at": None,
                "elapsed_ms": None,
                "played_at_ts": action.ts or started_at,
                "details": {},
            }
            result = self.drivers["tts_playback"].invoke("play_speech", {"text": action.text})
            finished_at = time.time()
            result_details = dict(result.details)
            status = str(result_details.get("status") or result.status)
            busy = self._coerce_busy(result_details.get("busy"), default=self._status_indicates_busy(status))
            last_error = self._extract_last_error(status, result_details)
            reason = self._extract_reason(result_details, default=None)
            self._last_playback = {
                **self._voice_config_details(),
                **base_details,
                "status": result.status,
                "busy": busy,
                "reason": reason,
                "last_error": last_error,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_ms": round((finished_at - started_at) * 1000, 2),
                "played_at_ts": action.ts or time.time(),
                "details": result_details,
            }
            self._last_playback.update(self._selected_driver_runtime_details(result_details))
            self._last_playback["status"] = status
            self._last_playback["busy"] = busy
            self._last_playback["reason"] = reason
            self._last_playback["last_error"] = last_error
            return SpeechPlaybackCompleted(
                ts=action.ts,
                source="mouth.tts_playback",
                status=status,
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
            )
        if isinstance(action, StopSpeechAction):
            started_at = time.time()
            action_reason = self._extract_action_reason(action)
            result = self.drivers["tts_playback"].invoke("stop_speech", {"reason": action_reason})
            finished_at = time.time()
            result_details = dict(result.details)
            stop_succeeded = self._stop_succeeded(result.status, result_details)
            status = "stopped" if stop_succeeded else "stop_failed"
            previous_busy = self._last_playback.get("busy", False) if self._last_playback is not None else False
            busy = False if stop_succeeded else self._coerce_busy(result_details.get("busy"), default=bool(previous_busy))
            reason = self._extract_reason(result_details, default=action_reason)
            last_error = None if stop_succeeded else self._extract_last_error(status, result_details)
            playback_details = {
                **self._voice_config_details(),
                **self._previous_playback_identity(),
                "session_id": action.session_id,
                "actor_id": action.actor_id,
                "target_id": action.target_id,
                "status": status,
                "busy": busy,
                "reason": reason,
                "last_error": last_error,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_ms": round((finished_at - started_at) * 1000, 2),
                "stopped_at_ts": action.ts or finished_at,
                "details": result_details,
            }
            playback_details.update(self._selected_driver_runtime_details(result_details))
            playback_details["status"] = status
            playback_details["busy"] = busy
            playback_details["reason"] = reason
            playback_details["last_error"] = last_error
            self._last_playback = playback_details
            executed_details = dict(result_details)
            executed_details.update(
                {
                    "status": status,
                    "busy": busy,
                    "reason": reason,
                    "last_error": last_error,
                }
            )
            return ActionExecuted(
                ts=action.ts,
                source="mouth.tts_playback",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details=executed_details,
            )
        return None

    def _tts_plan_health(self) -> SubfunctionHealth:
        probe_health = self._driver_subfunction_health("tts_plan")
        details = dict(probe_health.details)
        if self._driver_kind("tts_plan") != "noop":
            details.update(self._voice_config_details())
        if self._last_plan is not None:
            details.update(self._last_plan)
            details["status"] = self._last_plan.get("status", "planned")
        else:
            details.setdefault("status", "ready")
        return SubfunctionHealth(
            name="tts_plan",
            health=probe_health.health,
            details=details,
        )

    def _tts_playback_health(self) -> SubfunctionHealth:
        probe_health = self._driver_subfunction_health("tts_playback")
        details = dict(probe_health.details)
        if self._driver_kind("tts_playback") != "noop":
            details.update(self._voice_config_details())
        health = probe_health.health
        if self._last_playback is not None:
            nested_details = self._last_playback.get("details", {})
            if isinstance(nested_details, dict):
                details.update(nested_details)
            details.update({key: value for key, value in self._last_playback.items() if key != "details"})
            health = self._normalize_status(str(self._last_playback.get("status", probe_health.health)))
        else:
            details.setdefault("status", "ready")
        return SubfunctionHealth(
            name="tts_playback",
            health=health,
            details=details,
        )

    def _voice_config_details(self) -> dict[str, object]:
        playback_cfg = self.config.subfunctions.get("tts_playback")
        if playback_cfg is None:
            return {}
        if playback_cfg.driver.kind == "noop":
            return {"backend": "noop", "provider": "noop"}
        extra = playback_cfg.driver.extra
        backend = extra.get("backend")
        voice_id = extra.get("voice_id")
        if backend is None:
            return {}
        return {
            "backend": backend,
            "provider": backend,
            "voice_id": voice_id,
            "voice": voice_id,
            "model": extra.get("model"),
            "output_device": extra.get("output_device"),
            "playback_backend": extra.get("playback_backend"),
        }

    def _driver_subfunction_health(self, name: str) -> SubfunctionHealth:
        if self._driver_kind(name) == "noop":
            return SubfunctionHealth(
                name=name,
                health="unavailable",
                details={"driver": "noop", "status": "not_wired", "not_wired": True},
            )
        probe = self.drivers[name].heartbeat()
        return SubfunctionHealth(
            name=name,
            health=self._normalize_status(probe.status),
            details=self._merge_probe_details(dict(probe.details)),
        )

    def _previous_playback_identity(self) -> dict[str, object]:
        if self._last_playback is None:
            return {}
        return {
            key: value
            for key, value in self._last_playback.items()
            if key in {"text_preview", "text_char_count", "played_at_ts"}
        }

    @staticmethod
    def _selected_driver_runtime_details(details: dict[str, object]) -> dict[str, object]:
        return {
            key: details[key]
            for key in (
                "provider",
                "backend",
                "model",
                "voice",
                "voice_id",
                "output_device",
                "playback_backend",
                "operation",
            )
            if key in details
        }

    @staticmethod
    def _coerce_busy(value: object, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "busy", "playing", "speaking"}:
                return True
            if normalized in {"0", "false", "no", "idle", "stopped", "done", "complete", "completed"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def _status_indicates_busy(status: str) -> bool:
        normalized = str(status or "").strip().lower()
        return normalized in {"busy", "playing", "queued", "speaking", "streaming", "starting", "in_progress"}

    @staticmethod
    def _stop_succeeded(status: str, details: dict[str, object]) -> bool:
        statuses = [str(status or "").strip().lower()]
        detail_status = details.get("status")
        if detail_status is not None:
            statuses.append(str(detail_status).strip().lower())
        if any("fail" in item or "error" in item or "timeout" in item for item in statuses):
            return False
        return any(item in {"ok", "stopped", "stop_ok", "success", "succeeded"} for item in statuses)

    @staticmethod
    def _extract_reason(details: dict[str, object], *, default: object = None) -> object:
        for key in ("reason", "stop_reason", "cause"):
            value = details.get(key)
            if value not in (None, ""):
                return value
        return default

    @staticmethod
    def _extract_last_error(status: str, details: dict[str, object]) -> object:
        for key in ("last_error", "error", "stderr"):
            value = details.get(key)
            if value not in (None, ""):
                return value
        normalized = str(status or "").strip().lower()
        if "fail" in normalized or "error" in normalized or "timeout" in normalized:
            return details.get("reason") or status
        return None

    @staticmethod
    def _extract_action_reason(action: StopSpeechAction) -> object:
        value = getattr(action, "reason", None)
        if value not in (None, ""):
            return value
        details = getattr(action, "details", None)
        if isinstance(details, dict):
            return MouthOrgan._extract_reason(details, default=None)
        return None

    @staticmethod
    def _merge_probe_details(probe: dict[str, object]) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged

    @staticmethod
    def _preview_text(text: str) -> str:
        collapsed = " ".join(text.split())
        return collapsed[:80]
