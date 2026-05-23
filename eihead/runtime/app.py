"""Head runtime scaffold with native-provider-first routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import os
import threading
import time
from typing import Any, Mapping

from eiprotocol.event_routing import classify_event
from eihead.monitoring import build_status_snapshot
from eihead.neck import PanMoveCommand, PanNeckState, ReframeConfig, ReframeState, VisualTarget, plan_pan_move
from eihead.neck import plan_reframe_action
from eihead.protocol import MoveHeadAction, PlaySpeechAction, StopSpeechAction, serialize_message
from eihead.services import CapabilityRegistry
from .event_journal import EventJournal
from .status_projection import runtime_check_summary
from .event_projection import event_outcome_common
from .native_providers import (
    NativeProviderProbe,
    build_native_provider_statuses,
    normalize_native_provider_statuses,
)
from .composition import build_native_capability_probe
from .native_services import (
    EyeAdapterFactory,
    build_native_neck_servo_adapter,
    build_native_provider_services,
    build_native_voice_runtime,
    build_native_voice_status,
)

DEFAULT_CONFIG_PATH = "config/eibrain.yaml"
DEFAULT_BODY_RUNTIME_DELEGATE = "eihead.native_runtime"
DEFAULT_REALTIME_VISION_MAX_AGE_SECONDS = 2.0
DEFAULT_PTZ_MIN_ANGLE_DELTA = 2.0
REALTIME_VISION_ATTRS = (
    "eye_realtime",
    "vision_realtime",
    "realtime_vision",
    "latest_eye_realtime",
    "latest_vision_realtime",
    "latest_realtime_vision",
)
APP_REALTIME_VISION_ATTRS = (
    "eye_realtime",
    "realtime_vision",
    "latest_eye_realtime",
    "latest_vision_realtime",
    "latest_realtime_vision",
)
REALTIME_VISION_CONTAINER_KEYS = (
    "eye",
    "vision",
    "realtime_vision",
    "body_runtime",
    "subfunctions",
    "camera",
    "detection",
    "identity",
)


@dataclass(slots=True)
class HeadRuntimeApp:
    """Standalone eihead runtime facade."""

    body_runtime: Any = None
    config_path: str = DEFAULT_CONFIG_PATH
    delegate_name: str = DEFAULT_BODY_RUNTIME_DELEGATE
    realtime_vision_max_age_seconds: float = DEFAULT_REALTIME_VISION_MAX_AGE_SECONDS
    ptz_min_angle_delta: float = DEFAULT_PTZ_MIN_ANGLE_DELTA
    event_journal: EventJournal = field(default_factory=EventJournal, repr=False)
    neck_servo_adapter: Any | None = field(default=None, repr=False)
    neck_pan_state: PanNeckState = field(default_factory=PanNeckState, repr=False)
    native_providers: Mapping[str, Any] | None = field(default=None, repr=False)
    native_voice_status: Mapping[str, Any] | None = field(default=None, repr=False)
    voice_runtime: Any | None = field(default=None, repr=False)
    neck_reframe_config: ReframeConfig = field(default_factory=ReframeConfig, repr=False)
    neck_reframe_state: ReframeState = field(default_factory=ReframeState, repr=False)
    neck_reframe_enabled: bool = False
    neck_reframe_interval_s: float = 0.5
    neck_reframe_require_voice_awake: bool = False
    _ptz_last_target_angle: int | None = field(default=None, init=False, repr=False)
    _last_neck_plan: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _last_neck_servo: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _last_neck_reframe: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _neck_reframe_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _neck_reframe_stop_event: threading.Event | None = field(default=None, init=False, repr=False)
    _native_provider_services: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        raw_native_providers = self.native_providers if isinstance(self.native_providers, Mapping) else {}
        self._native_provider_services = {
            str(name): provider
            for name, provider in raw_native_providers.items()
            if _is_native_provider_service(provider)
        }
        normalizable_providers = {
            str(name): _pending_native_provider_service_status(provider)
            if str(name) in self._native_provider_services
            else provider
            for name, provider in raw_native_providers.items()
        }
        self.native_providers = normalize_native_provider_statuses(
            normalizable_providers,
            neck_servo_adapter=self.neck_servo_adapter,
        )
        self._start_native_voice_runtime()

    @classmethod
    def from_config_path(
        cls,
        path: str = DEFAULT_CONFIG_PATH,
        *,
        body_runtime_factory: Any | None = None,
        native_provider_probe: NativeProviderProbe | None = None,
        native_environ: Mapping[str, str] | None = None,
        neck_servo_adapter: Any | None = None,
        native_eye_adapter_factory: EyeAdapterFactory | None = None,
    ) -> "HeadRuntimeApp":
        _ = body_runtime_factory
        native_config = _load_optional_eihead_config(str(path))
        if neck_servo_adapter is None:
            neck_servo_adapter = build_native_neck_servo_adapter(native_config)
        (
            reframe_config,
            reframe_enabled,
            reframe_interval_s,
            reframe_require_voice_awake,
        ) = _neck_reframe_settings_from_config(native_config)
        native_provider_statuses = build_native_provider_statuses(
            config=native_config,
            environ=native_environ,
            probe=native_provider_probe,
            neck_servo_adapter=neck_servo_adapter,
        )
        native_provider_services = build_native_provider_services(
            native_config,
            config_path=str(path),
            eye_adapter_factory=native_eye_adapter_factory,
        )
        native_providers: dict[str, Any] = {
            **native_provider_statuses,
            **native_provider_services,
        }
        return cls(
            body_runtime=None,
            config_path=str(path),
            delegate_name=DEFAULT_BODY_RUNTIME_DELEGATE,
            neck_servo_adapter=neck_servo_adapter,
            native_providers=native_providers,
            native_voice_status=build_native_voice_status(native_config),
            voice_runtime=build_native_voice_runtime(native_config),
            neck_reframe_config=reframe_config,
            neck_reframe_enabled=reframe_enabled,
            neck_reframe_interval_s=reframe_interval_s,
            neck_reframe_require_voice_awake=reframe_require_voice_awake,
        )

    def snapshot(self) -> dict[str, Any]:
        body_snapshot, body_snapshot_check = self._body_snapshot_or_error()
        native_providers = self._native_provider_statuses()
        checks, check_details, status = runtime_check_summary(
            delegate_name=self.delegate_name,
            native_providers=native_providers,
            body_snapshot_check=body_snapshot_check,
        )
        return {
            "runtime": "eihead",
            "node_role": "head",
            "ok": status == "ok",
            "status": status,
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "checks": checks,
            "check_details": check_details,
            "native_providers": native_providers,
            "body_runtime": body_snapshot,
        }

    def status(self) -> dict[str, Any]:
        return {
            "command": "status",
            **self.snapshot(),
        }

    def health(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        body_runtime = snapshot.get("body_runtime", {})
        payload: dict[str, Any] = {
            "ok": snapshot.get("ok") is True,
            "status": _string_or_default(snapshot.get("status"), "unknown"),
            "runtime": "eihead",
            "node_role": "head",
            "source": "snapshot",
            "checked_at_ts": float(time.time()),
            "checks": snapshot.get("checks", {}),
            "check_details": snapshot.get("check_details", {}),
            "native_providers": snapshot.get("native_providers", {}),
        }
        if isinstance(body_runtime, Mapping) and "node_id" in body_runtime:
            payload["node_id"] = body_runtime["node_id"]
        return payload

    def capabilities(self) -> dict[str, Any]:
        body_snapshot = self._body_snapshot()
        node_id = _string_or_default(body_snapshot.get("node_id"), "honjia")
        manifest = CapabilityRegistry(
            {"node_id": node_id},
            probe=build_native_capability_probe(self._native_provider_statuses()),
        ).manifest()
        status_snapshot = build_status_snapshot(manifest)
        return {
            "command": "capabilities",
            "runtime": "eihead",
            "node_role": "head",
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "body_runtime_node_id": node_id,
            "body_runtime_capabilities": body_snapshot.get("capabilities", {}),
            **status_snapshot,
        }

    def vision_realtime(self) -> Mapping[str, Any] | Any | None:
        """Return explicit realtime eye payloads only.

        Legacy body snapshots can contain static frame paths and image-derived
        detections. Those are intentionally not promoted here; the monitor
        should say not-wired until a realtime eye adapter exposes a live stream
        hook.
        """

        now_ts = float(time.time())
        for candidate in self._realtime_vision_candidates():
            payload = _resolve_realtime_payload_candidate(candidate)
            if _is_realtime_vision_payload(
                payload,
                now_ts=now_ts,
                max_age_seconds=self.realtime_vision_max_age_seconds,
            ):
                return _with_realtime_device_payload(payload)
        return None

    def voice_status(self) -> Mapping[str, Any] | Any | None:
        """Return native voice diagnostics when available, else a body snapshot fallback."""

        live_voice_runtime = self._voice_runtime_payload("voice_status")
        if live_voice_runtime is not None:
            return live_voice_runtime

        if isinstance(self.native_voice_status, Mapping):
            return dict(self.native_voice_status)

        for attr_name in (
            "voice_realtime",
            "voice_status",
            "latest_voice_realtime",
            "latest_voice_status",
        ):
            if not hasattr(self.body_runtime, attr_name):
                continue
            source = getattr(self.body_runtime, attr_name)
            payload = _resolve_realtime_payload_candidate(source() if callable(source) else source)
            if payload is not None:
                return payload

        body_snapshot = self._body_snapshot()
        voice_dialogue = body_snapshot.get("voice_dialogue")
        organs = body_snapshot.get("organs") if isinstance(body_snapshot.get("organs"), Mapping) else {}
        ear = organs.get("ear") if isinstance(organs, Mapping) and isinstance(organs.get("ear"), Mapping) else None
        mouth = organs.get("mouth") if isinstance(organs, Mapping) and isinstance(organs.get("mouth"), Mapping) else None
        if isinstance(voice_dialogue, Mapping) or ear is not None or mouth is not None:
            payload: dict[str, Any] = {}
            if isinstance(voice_dialogue, Mapping):
                payload["voice_dialogue"] = dict(voice_dialogue)
            if ear is not None:
                payload["ear"] = dict(ear)
            if mouth is not None:
                payload["mouth"] = dict(mouth)
            return payload
        return None

    def voice_realtime(self) -> Mapping[str, Any] | Any | None:
        return self.voice_status()

    def eivoice_runtime_status(self) -> Mapping[str, Any]:
        payload = self._voice_runtime_payload("status")
        return dict(payload) if isinstance(payload, Mapping) else {}

    def neck_status(self) -> Mapping[str, Any]:
        """Return native pan/servo diagnostics for the monitor."""

        servo_status = self._neck_servo_status()
        provider_status = self._native_provider_statuses().get("neck", {})
        requested_status = _string_or_default(provider_status.get("status"), "")
        if servo_status.get("status") in {"unavailable", "error", "invalid", "unsupported"}:
            status = "degraded"
        elif requested_status in {"wired", "degraded", "unavailable", "unknown"}:
            status = requested_status
        else:
            status = "wired" if self.neck_servo_adapter is not None else "degraded"

        payload: dict[str, Any] = {
            "status": status,
            "pan": self.neck_pan_state.to_dict(),
            "servo": servo_status,
            "axis_support": {
                "pan": {"supported": True, "status": "supported"},
                "yaw": {"supported": True, "status": "supported"},
                "tilt": {
                    "supported": False,
                    "status": "unsupported",
                    "reason": "tilt_not_supported",
                },
            },
        }
        if self._last_neck_plan is not None:
            payload["neck_plan"] = dict(self._last_neck_plan)
        if self._last_neck_servo is not None:
            payload["neck_servo"] = dict(self._last_neck_servo)
        if self._last_neck_reframe is not None:
            payload["neck_reframe"] = dict(self._last_neck_reframe)
        return payload

    def neck_realtime(self) -> Mapping[str, Any]:
        return self.neck_status()

    def tick_neck_reframe(self, now_ts: float | None = None, *, live: bool = True) -> dict[str, Any]:
        now = float(time.time() if now_ts is None else now_ts)
        self.neck_reframe_state.current_pan_deg = float(self.neck_pan_state.current_angle)
        voice_gate = _neck_reframe_voice_gate(self.voice_status()) if self.neck_reframe_require_voice_awake else None
        if voice_gate is not None and not bool(voice_gate.get("allowed")):
            reason = _string_or_default(voice_gate.get("reason"), "voice_not_awake")
            action_payload = _neck_reframe_hold_action(self.neck_reframe_state, reason=reason)
            result = _neck_reframe_tick_payload(
                now_ts=now,
                live=live,
                target=None,
                action=action_payload,
                dispatch=None,
                status="hold",
                reason=reason,
                voice_gate=voice_gate,
            )
            self._last_neck_reframe = dict(result)
            return result

        target = _extract_visual_reframe_target(self.vision_realtime())
        state_before_action = replace(self.neck_reframe_state)
        action = plan_reframe_action(
            target,
            state=self.neck_reframe_state,
            config=self.neck_reframe_config,
            now_ts=now,
        )
        action_payload = action.as_dict()
        dispatch: dict[str, Any] | None = None
        status = "observe" if action.mode == "observe" else "hold"
        reason = action.reason
        if action.will_move and live:
            dispatch = self.handle_action(
                {
                    "type": "move_head",
                    "axis": "pan",
                    "angle": action.pan_deg,
                    "target_x": action.target_x,
                    "metadata": {
                        "source": "neck_reframe",
                        "mode": action.mode,
                        "reason": action.reason,
                        "frame_id": action.frame_id,
                    },
                },
                trace_id=f"neck-reframe-{int(now * 1000)}",
            )
            status = _string_or_default(dispatch.get("status"), "accepted")
            if not bool(dispatch.get("success")):
                self.neck_reframe_state = state_before_action
                reason = _neck_reframe_dispatch_failure_reason(dispatch) or _action_outcome_reason(dispatch) or status
                action_payload = {
                    **action_payload,
                    "pan_deg": float(self.neck_reframe_state.current_pan_deg),
                    "pan_delta_deg": 0.0,
                    "will_move": False,
                    "state": self.neck_reframe_state.as_dict(),
                }
        elif action.will_move:
            status = "planned"

        result = _neck_reframe_tick_payload(
            now_ts=now,
            live=live,
            target=_visual_target_payload(target) if target is not None else None,
            action=action_payload,
            dispatch=dispatch,
            status=status,
            reason=reason,
            voice_gate=voice_gate,
        )
        self._last_neck_reframe = dict(result)
        return result

    def start_neck_reframe_loop(self, *, interval_s: float | None = None, live: bool = True) -> dict[str, Any]:
        if not self.neck_reframe_enabled:
            return {
                "status": "disabled",
                "started": False,
                "reason": "neck_reframe_disabled",
            }
        if self._neck_reframe_thread is not None and self._neck_reframe_thread.is_alive():
            return {
                "status": "running",
                "started": False,
                "reason": "neck_reframe_loop_already_running",
                "interval_s": float(self.neck_reframe_interval_s),
            }

        interval = max(0.05, float(self.neck_reframe_interval_s if interval_s is None else interval_s))
        self.neck_reframe_interval_s = interval
        stop_event = threading.Event()
        self._neck_reframe_stop_event = stop_event

        def run_loop() -> None:
            while not stop_event.is_set():
                try:
                    self.tick_neck_reframe(live=live)
                except Exception as exc:  # pragma: no cover - protects long-running hardware process.
                    self._last_neck_reframe = {
                        "schema": "eihead.neck.reframe_tick.v1",
                        "status": "error",
                        "reason": "neck_reframe_loop_error",
                        "error": str(exc),
                        "live": bool(live),
                        "ts": float(time.time()),
                    }
                stop_event.wait(interval)

        thread = threading.Thread(target=run_loop, name="eihead-neck-reframe", daemon=True)
        self._neck_reframe_thread = thread
        thread.start()
        return {
            "status": "running",
            "started": True,
            "interval_s": interval,
            "live": bool(live),
        }

    def stop_neck_reframe_loop(self, *, timeout_s: float = 1.0) -> dict[str, Any]:
        stop_event = self._neck_reframe_stop_event
        thread = self._neck_reframe_thread
        if stop_event is None or thread is None:
            return {"status": "stopped", "stopped": False, "reason": "neck_reframe_loop_not_started"}
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))
        stopped = not thread.is_alive()
        if stopped:
            self._neck_reframe_thread = None
            self._neck_reframe_stop_event = None
        return {"status": "stopped" if stopped else "stopping", "stopped": stopped}

    def recent_events(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.event_journal.recent(limit)

    def event_summary(self) -> dict[str, Any]:
        return self.event_journal.summary()

    def handle_event(self, event: Mapping[str, Any] | Any, trace_id: str | None = None) -> dict[str, Any]:
        route = classify_event(event)
        effective_trace_id = trace_id or _event_trace_id(event)
        common = event_outcome_common(route, trace_id=effective_trace_id)

        if route.get("status") == "invalid":
            outcome = {
                **common,
                "ok": False,
                "accepted": False,
                "processed": False,
                "status": "not_processed",
                "reason": "invalid_event",
                "errors": list(route.get("errors") or []),
            }
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        if route.get("status") == "not_processed":
            reason = _string_or_default(route.get("reason"), "unsupported_event_name")
            outcome = {
                **common,
                "ok": False,
                "accepted": False,
                "processed": False,
                "status": "not_processed",
                "reason": reason,
            }
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        route_name = _string_or_default(route.get("route"), "")
        if route_name == "action_request":
            action = _action_from_event_route(route)
            action_outcome = self.handle_action(action, trace_id=effective_trace_id)
            accepted = action_outcome.get("status") == "accepted" or action_outcome.get("success") is True
            outcome = {
                **common,
                "ok": bool(action_outcome.get("success")),
                "accepted": bool(accepted),
                "processed": True,
                "status": _string_or_default(action_outcome.get("status"), "unknown"),
                "route": route_name,
                "action_outcome": action_outcome,
            }
            reason = _action_outcome_reason(action_outcome)
            if reason:
                outcome["reason"] = reason
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        if route.get("status") == "routed":
            outcome = {
                **common,
                "ok": True,
                "accepted": True,
                "processed": False,
                "status": "recorded",
                "reason": "recorded_for_diagnostics",
                "route": route_name,
            }
            self.event_journal.append(event, outcome, trace_id=effective_trace_id)
            return outcome

        outcome = {
            **common,
            "ok": False,
            "accepted": False,
            "processed": False,
            "status": "not_processed",
            "reason": "unsupported_event_route",
            "route": route_name,
        }
        self.event_journal.append(event, outcome, trace_id=effective_trace_id)
        return outcome

    def handle_action(self, action: Mapping[str, Any] | Any, trace_id: str | None = None) -> dict[str, Any]:
        normalized, effective_trace_id = self._normalize_action(action, trace_id=trace_id)
        action_type = self._action_type(normalized)
        action_id = _string_or_default(normalized.get("action_id") or normalized.get("id"), "")

        if action_type == "speak":
            text = _string_or_default(_action_value(normalized, "text"), "")
            if not text.strip():
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=effective_trace_id,
                    status="skipped",
                    success=False,
                    details={"reason": "missing_text"},
                )
            protocol_action = PlaySpeechAction(
                ts=time.time(),
                source="eihead.runtime",
                text=text,
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
            )
            return self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
                details={"text_char_count": len(text)},
            )

        if action_type == "move_head":
            if self.neck_servo_adapter is not None:
                return self._handle_native_neck_action(
                    normalized,
                    action_id=action_id,
                    trace_id=effective_trace_id,
                )

            axis = _string_or_default(_action_value(normalized, "axis"), "yaw").strip().lower() or "yaw"
            if axis != "yaw":
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=effective_trace_id,
                    status="unsupported",
                    success=False,
                    details={"axis": axis, "reason": "honjia currently exposes yaw/pan only"},
                )
            target_angle = _action_value(normalized, "target_angle")
            if target_angle is None:
                target_angle = _action_value(normalized, "angle")
            target_angle = _optional_int(target_angle)

            ptz_suppressed_reason = self._maybe_suppress_ptz_jitter(target_angle)
            if ptz_suppressed_reason is not None:
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=effective_trace_id,
                    status="skipped",
                    success=False,
                    details=ptz_suppressed_reason,
                )

            protocol_action = MoveHeadAction(
                ts=time.time(),
                source="eihead.runtime",
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
                target_name=_string_or_default(_action_value(normalized, "target_name"), "manual"),
                target_x=_optional_float(_action_value(normalized, "target_x")),
                target_angle=target_angle,
            )
            outcome = self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
                details={"axis": "yaw"},
            )
            if outcome.get("success") and target_angle is not None:
                self._ptz_last_target_angle = target_angle
            return outcome

        if action_type == "stop_speech":
            protocol_action = StopSpeechAction(
                ts=time.time(),
                source="eihead.runtime",
                session_id=_optional_string(_action_value(normalized, "session_id")),
                actor_id=_optional_string(_action_value(normalized, "actor_id")),
                target_id=_optional_string(_action_value(normalized, "target_id")),
            )
            return self._dispatch_protocol_action(
                protocol_action,
                action_id=action_id,
                action_type=action_type,
                trace_id=effective_trace_id,
            )

        if action_type == "capture_frame":
            return self._capture_frame_outcome(
                action_id=action_id,
                trace_id=effective_trace_id,
            )

        return self._action_outcome(
            action_id=action_id,
            action_type=action_type or "unknown",
            trace_id=effective_trace_id,
            status="unsupported",
            success=False,
            details={"reason": "unsupported_action_type"},
        )

    def _handle_native_neck_action(
        self,
        action: Mapping[str, Any],
        *,
        action_id: str,
        trace_id: str | None,
    ) -> dict[str, Any]:
        axis = _string_or_default(_action_value(action, "axis"), "pan").strip().lower() or "pan"
        target_angle = _action_value(action, "target_angle")
        if target_angle is None:
            target_angle = _action_value(action, "angle")

        plan = plan_pan_move(
            PanMoveCommand(
                axis=axis,
                target_angle=_optional_float(target_angle),
                target_x=_optional_float(_action_value(action, "target_x")),
                source="eihead.runtime",
                action_id=action_id,
                trace_id=trace_id or "",
                metadata=_action_metadata(action),
            ),
            self.neck_pan_state,
        )
        self.neck_pan_state = PanNeckState.from_dict(plan.get("state", {}))
        self._last_neck_plan = dict(plan)

        if not bool(plan.get("success")):
            reason = _string_or_default(plan.get("reason"), _string_or_default(plan.get("status"), "invalid"))
            self._last_neck_servo = {
                "status": _string_or_default(plan.get("status"), "invalid"),
                "success": False,
                "reason": reason,
            }
            return self._action_outcome(
                action_id=action_id,
                action_type="move_head",
                trace_id=trace_id,
                status=_string_or_default(plan.get("status"), "invalid"),
                success=False,
                details={
                    "axis": axis,
                    "reason": reason,
                    "neck_plan": plan,
                },
            )

        apply_plan = getattr(self.neck_servo_adapter, "apply_plan", None)
        if not callable(apply_plan):
            self._last_neck_servo = {
                "status": "unavailable",
                "success": False,
                "reason": "neck_servo_adapter_unavailable",
            }
            return self._action_outcome(
                action_id=action_id,
                action_type="move_head",
                trace_id=trace_id,
                status="skipped",
                success=False,
                details={
                    "axis": "pan",
                    "reason": "neck_servo_adapter_unavailable",
                    "neck_plan": plan,
                },
            )

        try:
            servo_outcome = apply_plan(plan)
        except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
            self._last_neck_servo = {
                "status": "error",
                "success": False,
                "reason": "neck_servo_adapter_error",
                "error": str(exc),
            }
            return self._action_outcome(
                action_id=action_id,
                action_type="move_head",
                trace_id=trace_id,
                status="error",
                success=False,
                details={
                    "axis": "pan",
                    "reason": "neck_servo_adapter_error",
                    "error": str(exc),
                    "neck_plan": plan,
                },
            )

        if isinstance(servo_outcome, Mapping):
            servo_details = dict(servo_outcome)
        else:
            servo_details = _serialize_outcome(servo_outcome)
        self._last_neck_servo = dict(servo_details)
        servo_status = _string_or_default(servo_details.get("status"), "")

        if servo_status == "ok":
            target_angle_value = _optional_float(plan.get("action", {}).get("target_angle"))
            if target_angle_value is not None:
                self.neck_pan_state = replace(
                    self.neck_pan_state,
                    current_angle=target_angle_value,
                    target_angle=target_angle_value,
                )
            status = "accepted"
            success = True
        elif servo_status == "suppressed":
            status = "skipped"
            success = True
        elif servo_status == "unavailable":
            status = "skipped"
            success = False
        else:
            status = servo_status or _string_or_default(plan.get("status"), "skipped")
            success = False

        return self._action_outcome(
            action_id=action_id,
            action_type="move_head",
            trace_id=trace_id,
            status=status,
            success=success,
            delegated=True,
            details={
                "axis": "pan",
                "reason": _native_neck_reason(plan, servo_details),
                "neck_plan": plan,
                "neck_servo": servo_details,
            },
        )

    def _maybe_suppress_ptz_jitter(self, target_angle: int | None) -> dict[str, Any] | None:
        target_angle_int = target_angle
        if target_angle_int is None:
            return None
        min_angle_delta = float(self.ptz_min_angle_delta)
        previous_angle = self._ptz_last_target_angle
        if previous_angle is None:
            return None
        if min_angle_delta <= 0:
            return None
        if abs(target_angle_int - previous_angle) <= min_angle_delta:
            return {
                "axis": "yaw",
                "reason": "ptz_jitter_suppressed",
                "previous_target_angle": previous_angle,
                "target_angle": target_angle_int,
                "min_angle_delta": min_angle_delta,
            }
        return None

    def serve(self) -> dict[str, Any]:
        return {
            "command": "serve",
            "serve_mode": "compatibility_snapshot",
            **self.snapshot(),
        }

    def verify(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        body_runtime = snapshot.get("body_runtime", {})
        organ_count = body_runtime.get("organ_count") if isinstance(body_runtime, Mapping) else None
        return {
            "command": "verify",
            "runtime": "eihead",
            "ok": snapshot.get("ok") is True,
            "status": _string_or_default(snapshot.get("status"), "unknown"),
            "checks": snapshot.get("checks", {}),
            "check_details": snapshot.get("check_details", {}),
            "organ_count": organ_count,
            "config_path": self.config_path,
            "delegate": self.delegate_name,
            "native_providers": snapshot.get("native_providers", {}),
            "body_runtime": body_runtime,
        }

    def _body_snapshot_or_error(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return self._body_snapshot(), {"status": "ok"}
        except Exception as exc:
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            return (
                {
                    "status": "blocked",
                    "reason": "body_runtime_snapshot_failed",
                    "error": error,
                },
                {
                    "status": "blocked",
                    "reason": "body_runtime_snapshot_failed",
                    "error": error,
                },
            )

    def _body_snapshot(self) -> dict[str, Any]:
        if not hasattr(self.body_runtime, "snapshot"):
            return {}
        snapshot = self.body_runtime.snapshot()
        if not isinstance(snapshot, Mapping):
            return {}
        return dict(snapshot)

    def _realtime_vision_candidates(self) -> list[Any]:
        candidates: list[Any] = []
        candidates.extend(_attr_payload_candidates(self, APP_REALTIME_VISION_ATTRS))
        candidates.extend(_native_provider_service_realtime_candidates(self._native_provider_services))
        candidates.extend(_native_provider_realtime_candidates(self.native_providers or {}))
        candidates.extend(_attr_payload_candidates(self.body_runtime, REALTIME_VISION_ATTRS))
        try:
            body_snapshot = self._body_snapshot()
        except Exception:
            body_snapshot = {}
        candidates.extend(_mapping_realtime_candidates(body_snapshot))
        return candidates

    def _native_provider_statuses(self) -> dict[str, dict[str, Any]]:
        statuses = dict(self.native_providers or {})
        for provider_name, service in self._native_provider_services.items():
            service_status = _native_provider_service_status(service)
            if service_status is None:
                continue
            statuses[provider_name] = normalize_native_provider_statuses(
                {provider_name: service_status},
                neck_servo_adapter=self.neck_servo_adapter,
            )[provider_name]
        return statuses

    def _neck_servo_status(self) -> dict[str, Any]:
        adapter = self.neck_servo_adapter
        if adapter is None:
            return {
                "status": "unavailable",
                "available": False,
                "reason": "neck_servo_adapter_missing",
            }
        status_fn = getattr(adapter, "status", None)
        if callable(status_fn):
            try:
                payload = status_fn()
            except Exception as exc:  # pragma: no cover - hardware dependent.
                return {
                    "status": "error",
                    "available": False,
                    "reason": "neck_servo_status_error",
                    "error": str(exc),
                }
            if isinstance(payload, Mapping):
                return dict(payload)
        if callable(getattr(adapter, "apply_plan", None)):
            return {
                "status": "ready",
                "available": True,
                "reason": "neck_servo_adapter_ready",
                "hardware_verified": False,
            }
        return {
            "status": "unavailable",
            "available": False,
            "reason": "neck_servo_adapter_unavailable",
        }

    def _normalize_action(
        self,
        action: Mapping[str, Any] | Any,
        *,
        trace_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        if isinstance(action, Mapping):
            payload = dict(action)
        elif hasattr(action, "to_dict") and callable(action.to_dict):
            payload = dict(action.to_dict())
        elif is_dataclass(action):
            payload = asdict(action)
        else:
            return {"type": "unsupported", "raw_type": type(action).__name__}, trace_id

        nested = payload.get("action")
        if isinstance(nested, Mapping):
            effective_trace_id = trace_id or _optional_string(payload.get("trace_id"))
            return dict(nested), effective_trace_id
        return payload, trace_id or _optional_string(payload.get("trace_id"))

    def _action_type(self, action: Mapping[str, Any]) -> str:
        raw = _string_or_default(
            action.get("type") or action.get("action_type") or action.get("kind"),
            "",
        )
        normalized = raw.strip().lower()
        aliases = {
            "play_speech": "speak",
            "play_speech_action": "speak",
            "speech": "speak",
            "move_head_action": "move_head",
            "pan": "move_head",
            "stop_speech_action": "stop_speech",
            "stop_tts": "stop_speech",
        }
        return aliases.get(normalized, normalized)

    def _dispatch_protocol_action(
        self,
        protocol_action: Any,
        *,
        action_id: str,
        action_type: str,
        trace_id: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        dispatch_actions = getattr(self.body_runtime, "dispatch_actions", None)
        if not callable(dispatch_actions):
            native_voice_outcome = self._handle_native_voice_action(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                text=_string_or_default(getattr(protocol_action, "text", ""), ""),
            )
            if native_voice_outcome is not None:
                return native_voice_outcome
            return self._action_outcome(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                status="not_wired",
                success=False,
                details={
                    **dict(details or {}),
                    "reason": "native_provider_unavailable",
                    "protocol_action": protocol_action.kind,
                },
            )
        try:
            delegate_outcomes = dispatch_actions([protocol_action])
        except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
            return self._action_outcome(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                status="error",
                success=False,
                details={
                    **dict(details or {}),
                    "error": str(exc),
                    "protocol_action": protocol_action.kind,
                },
            )

        serialized = [_serialize_outcome(outcome) for outcome in delegate_outcomes or []]
        return self._action_outcome(
            action_id=action_id,
            action_type=action_type,
            trace_id=trace_id,
            status="accepted" if serialized else "skipped",
            success=bool(serialized),
            delegated=True,
            details={
                **dict(details or {}),
                "protocol_action": protocol_action.kind,
                "delegate_outcomes": serialized,
            },
        )

    def _start_native_voice_runtime(self) -> None:
        start = getattr(self.voice_runtime, "start", None)
        if callable(start):
            start()

    def _voice_runtime_payload(self, method_name: str) -> Mapping[str, Any] | Any | None:
        source = getattr(self.voice_runtime, method_name, None)
        if callable(source):
            return source()
        if method_name == "status" and isinstance(self.voice_runtime, Mapping):
            return dict(self.voice_runtime)
        return None

    def _handle_native_voice_action(
        self,
        *,
        action_id: str,
        action_type: str,
        trace_id: str | None,
        text: str = "",
    ) -> dict[str, Any] | None:
        if self.voice_runtime is None:
            return None
        if action_type == "speak":
            speak = getattr(self.voice_runtime, "speak", None)
            if not callable(speak):
                return None
            try:
                speech = speak(text)
            except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=trace_id,
                    status="error",
                    success=False,
                    details={"reason": "native_voice_runtime_error", "error": str(exc)},
                )
            details = _serialize_outcome(speech)
            status = _string_or_default(details.get("status"), "accepted")
            return self._action_outcome(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                status="accepted" if status == "ok" else status,
                success=bool(details.get("success")),
                delegated=True,
                details={"provider": "native_voice_runtime", **details},
            )
        if action_type == "stop_speech":
            stop_speech = getattr(self.voice_runtime, "stop_speech", None)
            if not callable(stop_speech):
                stop_speech = getattr(self.voice_runtime, "stop", None)
            if not callable(stop_speech):
                return None
            try:
                stopped = stop_speech()
            except Exception as exc:  # pragma: no cover - exercised by integration when hardware fails.
                return self._action_outcome(
                    action_id=action_id,
                    action_type=action_type,
                    trace_id=trace_id,
                    status="error",
                    success=False,
                    details={"reason": "native_voice_runtime_error", "error": str(exc)},
                )
            details = _serialize_outcome(stopped) if stopped is not None else {"status": "stopped", "success": True}
            return self._action_outcome(
                action_id=action_id,
                action_type=action_type,
                trace_id=trace_id,
                status=_string_or_default(details.get("status"), "stopped"),
                success=bool(details.get("success", True)),
                delegated=True,
                details={"provider": "native_voice_runtime", **details},
            )
        return None

    def _capture_frame_outcome(self, *, action_id: str, trace_id: str | None) -> dict[str, Any]:
        capture_frame = getattr(self.body_runtime, "capture_frame", None)
        if callable(capture_frame):
            try:
                frame = capture_frame()
            except Exception as exc:  # pragma: no cover - exercised by integration when camera fails.
                return self._action_outcome(
                    action_id=action_id,
                    action_type="capture_frame",
                    trace_id=trace_id,
                    status="error",
                    success=False,
                    details={"error": str(exc)},
                )
            return self._action_outcome(
                action_id=action_id,
                action_type="capture_frame",
                trace_id=trace_id,
                status="accepted",
                success=True,
                delegated=True,
                details={"frame": _serialize_outcome(frame)},
            )

        latest_visual_frame_path = getattr(self.body_runtime, "latest_visual_frame_path", None)
        if callable(latest_visual_frame_path):
            frame_path = latest_visual_frame_path()
            if frame_path:
                return self._action_outcome(
                    action_id=action_id,
                    action_type="capture_frame",
                    trace_id=trace_id,
                    status="accepted",
                    success=True,
                    delegated=True,
                    details={"frame_path": str(frame_path), "source": "latest_visual_frame_path"},
                )
            return self._action_outcome(
                action_id=action_id,
                action_type="capture_frame",
                trace_id=trace_id,
                status="skipped",
                success=False,
                delegated=True,
                details={"reason": "no_latest_visual_frame"},
            )

        return self._action_outcome(
            action_id=action_id,
            action_type="capture_frame",
            trace_id=trace_id,
            status="not_wired",
            success=False,
            details={"reason": "native_provider_unavailable"},
        )

    def _action_outcome(
        self,
        *,
        action_id: str,
        action_type: str,
        trace_id: str | None,
        status: str,
        success: bool,
        details: Mapping[str, Any] | None = None,
        delegated: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema": "eihead.execution_outcome.v1",
            "runtime": "eihead",
            "node_role": "head",
            "action_id": action_id,
            "action_type": action_type,
            "trace_id": trace_id or "",
            "status": status,
            "success": success,
            "delegated": delegated,
            "details": dict(details or {}),
        }


def _attr_payload_candidates(source_object: Any, attr_names: tuple[str, ...]) -> list[Any]:
    candidates: list[Any] = []
    for attr_name in attr_names:
        if not hasattr(source_object, attr_name):
            continue
        source = getattr(source_object, attr_name)
        candidates.append(source() if callable(source) else source)
    return candidates


def _neck_reframe_tick_payload(
    *,
    now_ts: float,
    live: bool,
    target: dict[str, Any] | None,
    action: Mapping[str, Any],
    dispatch: Mapping[str, Any] | None,
    status: str,
    reason: str,
    voice_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "eihead.neck.reframe_tick.v1",
        "ts": float(now_ts),
        "status": status,
        "reason": reason,
        "live": bool(live),
        "target": target,
        "action": dict(action),
        "will_move": bool(action.get("will_move")),
        "suppressed": not bool(action.get("will_move")),
        "suppression_reason": "" if action.get("will_move") else reason,
    }
    if dispatch is not None:
        payload["dispatch"] = dict(dispatch)
    if voice_gate is not None:
        payload["voice_gate"] = dict(voice_gate)
    return payload


def _neck_reframe_hold_action(state: ReframeState, *, reason: str) -> dict[str, Any]:
    state.suppressed_reason = reason
    state.suppressed_count += 1
    return {
        "mode": "hold",
        "pan_deg": float(state.current_pan_deg),
        "pan_delta_deg": 0.0,
        "will_move": False,
        "reason": reason,
        "target_x": None,
        "target_label": None,
        "target_known": None,
        "frame_id": None,
        "state": state.as_dict(),
    }


def _neck_reframe_voice_gate(payload: Any) -> dict[str, Any]:
    data = _payload_mapping(payload)
    if data is None:
        return {"allowed": False, "reason": "voice_unavailable"}

    observation = data.get("observation")
    if isinstance(observation, Mapping) and (
        "voice_dialogue" in observation or "voice_chain" in observation or "dialogue" in observation
    ):
        data = dict(observation)

    dialogue = (
        _mapping_from_any(data.get("voice_dialogue"))
        or _mapping_from_any(data.get("voice_chain"))
        or _mapping_from_any(data.get("dialogue"))
    )
    local_gate = (
        _mapping_from_any(dialogue.get("local_gate"))
        or _mapping_from_any(dialogue.get("localWakeGate"))
        or _mapping_from_any(data.get("local_gate"))
        or _mapping_from_any(data.get("localWakeGate"))
    )
    phase = _normalized_payload_text(dialogue.get("phase") or dialogue.get("state") or data.get("phase") or data.get("state"))
    gate_state = _normalized_payload_text(local_gate.get("state"))
    last_gate_status = _normalized_payload_text(
        dialogue.get("last_gate_status")
        or dialogue.get("lastGateStatus")
        or local_gate.get("lastStatus")
        or local_gate.get("last_status")
        or data.get("last_gate_status")
    )
    conversation_active = (
        _truthy_payload_flag(dialogue.get("conversation_active"))
        or _truthy_payload_flag(dialogue.get("conversationActive"))
        or _truthy_payload_flag(local_gate.get("conversationActive"))
        or _truthy_payload_flag(local_gate.get("conversation_active"))
    )
    if conversation_active:
        return _voice_gate_payload(True, "voice_awake", phase=phase, gate_state=gate_state, last_gate_status=last_gate_status)

    if phase in {"listening", "thinking", "responding", "speaking", "active", "conversation_active"}:
        return _voice_gate_payload(True, "voice_awake", phase=phase, gate_state=gate_state, last_gate_status=last_gate_status)

    wake_word_required = _truthy_payload_flag(dialogue.get("wake_word_required") or data.get("wake_word_required"))
    running_without_wake_word = _truthy_payload_flag(dialogue.get("running") or data.get("running")) and not wake_word_required
    if running_without_wake_word:
        return _voice_gate_payload(True, "voice_awake", phase=phase, gate_state=gate_state, last_gate_status=last_gate_status)

    if phase == "sleeping" or gate_state == "armed" or last_gate_status == "waiting_for_wake_word":
        return _voice_gate_payload(False, "voice_sleeping", phase=phase, gate_state=gate_state, last_gate_status=last_gate_status)

    return _voice_gate_payload(False, "voice_not_awake", phase=phase, gate_state=gate_state, last_gate_status=last_gate_status)


def _voice_gate_payload(
    allowed: bool,
    reason: str,
    *,
    phase: str,
    gate_state: str,
    last_gate_status: str,
) -> dict[str, Any]:
    return {
        "allowed": bool(allowed),
        "reason": reason,
        "phase": phase,
        "gate_state": gate_state,
        "last_gate_status": last_gate_status,
    }


def _visual_target_payload(target: VisualTarget) -> dict[str, Any]:
    return {
        "label": target.label,
        "target_x": target.target_x,
        "known": target.known,
        "confidence": target.confidence,
        "crop_width": target.crop_width,
        "crop_height": target.crop_height,
        "frame_id": target.frame_id,
        "track_id": target.track_id,
        "reason": target.reason,
    }


def _extract_visual_reframe_target(payload: Any) -> VisualTarget | None:
    data = _payload_mapping(payload)
    if data is None:
        return None
    nested_observation = data.get("observation")
    if isinstance(nested_observation, Mapping) and "detections" not in data:
        data = dict(nested_observation)
    detections = _list_payload(data.get("detections"))
    face_detections = [item for item in detections if _is_face_detection(item)]
    candidates = face_detections or _identity_observation_candidates(data) or _face_crop_candidates(data) or detections
    if not candidates:
        overlay = data.get("overlay")
        if isinstance(overlay, Mapping):
            top_target = overlay.get("top_target")
            if isinstance(top_target, Mapping):
                candidates = [dict(top_target)]
        tracked_target = data.get("tracked_target") or data.get("target")
        if not candidates and isinstance(tracked_target, Mapping):
            candidates = [dict(tracked_target)]
    if not candidates:
        return None

    frame_width = _optional_float(
        data.get("frame_width")
        or data.get("width")
        or _mapping_value(data.get("frame"), "width")
        or _mapping_value(data.get("stream"), "width")
        or _mapping_value(data.get("evidence"), "frame", "width")
    )
    frame_height = _optional_float(
        data.get("frame_height")
        or data.get("height")
        or _mapping_value(data.get("frame"), "height")
        or _mapping_value(data.get("stream"), "height")
        or _mapping_value(data.get("evidence"), "frame", "height")
    )
    best = _best_visual_target_candidate(candidates, frame_width=frame_width, frame_height=frame_height)
    if best is None:
        return None
    detection, target_x, crop_width, crop_height = best
    crop_width, crop_height = _face_crop_size(
        data,
        detection=detection,
        fallback_width=crop_width,
        fallback_height=crop_height,
    )
    return VisualTarget(
        label=_string_or_default(detection.get("label"), "face"),
        target_x=target_x,
        known=_candidate_has_known_identity(data, detection),
        confidence=_target_confidence(detection),
        crop_width=_optional_int(crop_width),
        crop_height=_optional_int(crop_height),
        frame_id=data.get("frame_id") or detection.get("frame_id"),
        track_id=detection.get("track_id") or detection.get("trackId") or detection.get("id"),
        reason=_string_or_default(
            detection.get("reason"),
            "face_detection" if _is_face_detection(detection) else "visual_detection",
        ),
    )


def _best_visual_target_candidate(
    candidates: list[dict[str, Any]],
    *,
    frame_width: float | None,
    frame_height: float | None,
) -> tuple[dict[str, Any], float, float | None, float | None] | None:
    best: tuple[float, dict[str, Any], float, float | None, float | None] | None = None
    for detection in candidates:
        target = _detection_target_x_and_size(detection, frame_width=frame_width, frame_height=frame_height)
        if target is None:
            continue
        target_x, width, height = target
        confidence = _optional_float(_first_present_value(detection, "confidence", "score")) or 0.0
        area = (width or 0.0) * (height or 0.0)
        score = confidence + min(area / 100_000.0, 1.0)
        if best is None or score > best[0]:
            best = (score, detection, target_x, width, height)
    if best is None:
        return None
    _, detection, target_x, width, height = best
    return detection, target_x, width, height


def _detection_target_x_and_size(
    detection: Mapping[str, Any],
    *,
    frame_width: float | None,
    frame_height: float | None,
) -> tuple[float, float | None, float | None] | None:
    center_x = _center_x(detection.get("center") or detection.get("target_center"))
    width: float | None = None
    height: float | None = None
    if center_x is None:
        center_x, width, height = _bbox_target_x_and_size(
            detection.get("bbox") or detection.get("box"),
            frame_width=frame_width,
            frame_height=frame_height,
            bbox_format=_string_or_default(detection.get("bboxFormat") or detection.get("bbox_format"), ""),
        )
    if center_x is None:
        center_x = _optional_float(detection.get("center_x") or detection.get("x"))
    if center_x is None:
        return None
    if frame_width and center_x > 1.0:
        center_x = center_x / frame_width
    return _clamp_float(center_x, 0.0, 1.0), width, height


def _bbox_target_x_and_size(
    bbox: Any,
    *,
    frame_width: float | None,
    frame_height: float | None,
    bbox_format: str,
) -> tuple[float | None, float | None, float | None]:
    if isinstance(bbox, Mapping):
        if all(key in bbox for key in ("x_min", "x_max")):
            x_min = _optional_float(bbox.get("x_min"))
            x_max = _optional_float(bbox.get("x_max"))
            y_min = _optional_float(bbox.get("y_min"))
            y_max = _optional_float(bbox.get("y_max"))
            if x_min is None or x_max is None:
                return None, None, None
            return _normalize_bbox_center(x_min, x_max, y_min, y_max, frame_width=frame_width, frame_height=frame_height)
        if all(key in bbox for key in ("x", "w")):
            x = _optional_float(bbox.get("x"))
            w = _optional_float(bbox.get("w"))
            h = _optional_float(bbox.get("h"))
            if x is None or w is None:
                return None, None, None
            center = x + (w / 2.0)
            if frame_width and center > 1.0:
                center = center / frame_width
            return center, _pixel_extent(w, frame_width), _pixel_extent(h, frame_height)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1 = _optional_float(bbox[0])
        y1 = _optional_float(bbox[1])
        x2_or_w = _optional_float(bbox[2])
        y2_or_h = _optional_float(bbox[3])
        if x1 is None or x2_or_w is None:
            return None, None, None
        if bbox_format.strip().lower() in {"xyxy", "x_min_y_min_x_max_y_max"}:
            return _normalize_bbox_center(x1, x2_or_w, y1, y2_or_h, frame_width=frame_width, frame_height=frame_height)
        if x1 <= 1.0 and x2_or_w <= 1.0:
            center = x1 + (x2_or_w / 2.0)
            return center, _pixel_extent(x2_or_w, frame_width), _pixel_extent(y2_or_h, frame_height)
        width = x2_or_w - x1 if x2_or_w > x1 else x2_or_w
        center = x1 + (width / 2.0)
        if frame_width and center > 1.0:
            center = center / frame_width
        height = None if y2_or_h is None else (y2_or_h - y1 if y1 is not None and y2_or_h > y1 else y2_or_h)
        return center, width, _pixel_extent(height, frame_height)
    return None, None, None


def _normalize_bbox_center(
    x_min: float,
    x_max: float,
    y_min: float | None,
    y_max: float | None,
    *,
    frame_width: float | None,
    frame_height: float | None,
) -> tuple[float, float | None, float | None]:
    center = (x_min + x_max) / 2.0
    if frame_width and center > 1.0:
        center = center / frame_width
    width = abs(x_max - x_min)
    height = None if y_min is None or y_max is None else abs(y_max - y_min)
    return center, _pixel_extent(width, frame_width), _pixel_extent(height, frame_height)


def _pixel_extent(value: float | None, frame_size: float | None) -> float | None:
    if value is None:
        return None
    if value <= 1.0:
        return value * frame_size if frame_size else None
    return value


def _center_x(center: Any) -> float | None:
    if isinstance(center, Mapping):
        return _optional_float(center.get("x") or center.get("center_x"))
    if isinstance(center, (list, tuple)) and center:
        return _optional_float(center[0])
    return None


def _face_crop_size(
    data: Mapping[str, Any],
    *,
    detection: Mapping[str, Any],
    fallback_width: float | None,
    fallback_height: float | None,
) -> tuple[float | None, float | None]:
    evidence = data.get("evidence")
    if not isinstance(evidence, Mapping):
        return fallback_width, fallback_height
    crops = _list_payload(evidence.get("face_crops") or evidence.get("faceCrops"))
    if not crops:
        return fallback_width, fallback_height
    crop = _matching_face_crop(crops, detection) or crops[0]
    return (
        _optional_float(crop.get("width") or crop.get("w")) or fallback_width,
        _optional_float(crop.get("height") or crop.get("h")) or fallback_height,
    )


def _matching_face_crop(crops: list[dict[str, Any]], detection: Mapping[str, Any]) -> dict[str, Any] | None:
    detection_track_id = _optional_string(detection.get("track_id") or detection.get("trackId") or detection.get("id"))
    if detection_track_id:
        for crop in crops:
            crop_track_id = _optional_string(crop.get("track_id") or crop.get("trackId") or crop.get("id"))
            if crop_track_id == detection_track_id:
                return crop

    detection_bbox = detection.get("bbox") or detection.get("box")
    if detection_bbox is None:
        return crops[0] if len(crops) == 1 else None
    best: tuple[float, dict[str, Any]] | None = None
    for crop in crops:
        crop_bbox = crop.get("bbox")
        if crop_bbox is None:
            continue
        distance = _bbox_center_distance(detection_bbox, crop_bbox)
        if distance is None:
            continue
        if best is None or distance < best[0]:
            best = (distance, crop)
    return None if best is None else best[1]


def _bbox_center_distance(left_bbox: Any, right_bbox: Any) -> float | None:
    left_x, _, _ = _bbox_target_x_and_size(left_bbox, frame_width=None, frame_height=None, bbox_format="")
    right_x, _, _ = _bbox_target_x_and_size(right_bbox, frame_width=None, frame_height=None, bbox_format="")
    if left_x is None or right_x is None:
        return None
    return abs(left_x - right_x)


def _identity_observation_candidates(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in _list_payload(data.get("identity_observations")) + _list_payload(data.get("identities")):
        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping):
            continue
        bbox = evidence.get("bbox")
        if bbox is None:
            continue
        candidates.append(
            {
                "label": "face",
                "known": item.get("known") if "known" in item else item.get("is_known"),
                "confidence": _first_present_value(item, "confidence", mapping=evidence, fallback_keys=("similarity", "confidence")),
                "bbox": bbox,
                "track_id": evidence.get("track_id") or item.get("track_id") or item.get("trackId"),
                "frame_id": item.get("frame_id") or evidence.get("frame_id"),
                "reason": "identity_observation",
            }
        )
    return candidates


def _face_crop_candidates(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = data.get("evidence")
    if not isinstance(evidence, Mapping):
        return []
    candidates: list[dict[str, Any]] = []
    for crop in _list_payload(evidence.get("face_crops") or evidence.get("faceCrops")):
        bbox = crop.get("bbox")
        if bbox is None:
            continue
        confidence = _first_present_value(crop, "confidence", "score")
        candidates.append(
            {
                "label": "face",
                "confidence": confidence if confidence is not None else 1.0,
                "bbox": bbox,
                "track_id": crop.get("track_id") or crop.get("trackId"),
                "frame_id": crop.get("frame_id"),
                "reason": "face_crop",
            }
        )
    return candidates


def _payload_has_known_identity(data: Mapping[str, Any]) -> bool:
    summary = str(data.get("identity_summary") or "").strip().lower()
    if summary and "unknown" not in summary and summary not in {"none", "missing"}:
        return True
    for key in ("identity_observations", "identities"):
        for item in _list_payload(data.get(key)):
            if _truthy_payload_flag(item.get("known") or item.get("is_known")):
                return True
    return False


def _candidate_has_known_identity(data: Mapping[str, Any], detection: Mapping[str, Any]) -> bool:
    if "known" in detection or "is_known" in detection:
        return _truthy_payload_flag(detection.get("known") if "known" in detection else detection.get("is_known"))

    candidate_track_id = _optional_string(detection.get("track_id") or detection.get("trackId") or detection.get("id"))
    identities = _list_payload(data.get("identity_observations")) + _list_payload(data.get("identities"))
    if candidate_track_id:
        for item in identities:
            evidence = item.get("evidence")
            evidence_payload = evidence if isinstance(evidence, Mapping) else {}
            identity_track_id = _optional_string(
                item.get("track_id")
                or item.get("trackId")
                or evidence_payload.get("track_id")
                or evidence_payload.get("trackId")
            )
            if identity_track_id == candidate_track_id:
                return _truthy_payload_flag(item.get("known") if "known" in item else item.get("is_known"))
        return False

    if len(identities) == 1 and len(_list_payload(data.get("detections"))) <= 1:
        item = identities[0]
        return _truthy_payload_flag(item.get("known") if "known" in item else item.get("is_known"))
    return _payload_has_known_identity(data) if not identities else False


def _target_confidence(detection: Mapping[str, Any]) -> float:
    value = _optional_float(_first_present_value(detection, "confidence", "score"))
    return 1.0 if value is None else value


def _first_present_value(
    primary: Mapping[str, Any],
    *keys: str,
    mapping: Mapping[str, Any] | None = None,
    fallback_keys: tuple[str, ...] = (),
) -> Any:
    for key in keys:
        if key in primary and primary.get(key) is not None:
            return primary.get(key)
    fallback = mapping or {}
    for key in fallback_keys:
        if key in fallback and fallback.get(key) is not None:
            return fallback.get(key)
    return None


def _is_face_detection(item: Mapping[str, Any]) -> bool:
    label = str(item.get("label") or item.get("class") or item.get("name") or "").strip().lower()
    return label in {"face", "person_face", "human_face"}


def _list_payload(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, (str, bytes)):
        return []
    try:
        raw_items = list(value)
    except TypeError:
        return []
    return [dict(item) for item in raw_items if isinstance(item, Mapping)]


def _mapping_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _native_provider_realtime_candidates(native_providers: Mapping[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    eye_provider = native_providers.get("eye") if isinstance(native_providers, Mapping) else None
    if isinstance(eye_provider, Mapping):
        candidates.extend(_mapping_realtime_candidates(eye_provider))
    return candidates


def _native_provider_service_realtime_candidates(native_provider_services: Mapping[str, Any]) -> list[Any]:
    eye_provider = native_provider_services.get("eye") if isinstance(native_provider_services, Mapping) else None
    return [eye_provider] if eye_provider is not None else []


def _mapping_realtime_candidates(payload: Any) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    candidates: list[Any] = []

    live_candidate = _live_vision_state_candidate(payload)
    if live_candidate is not None:
        candidates.append(live_candidate)

    simulator_candidate = _simulator_vision_state_candidate(payload.get("vision_state"))
    if simulator_candidate is not None:
        candidates.append(simulator_candidate)

    for attr_name in REALTIME_VISION_ATTRS:
        if attr_name in payload:
            candidates.append(payload[attr_name])
    if "realtime" in payload:
        candidates.append(payload["realtime"])

    details = payload.get("details")
    if isinstance(details, Mapping):
        candidates.append(details)
        candidates.extend(_mapping_realtime_candidates(details))

    for container_key in REALTIME_VISION_CONTAINER_KEYS:
        container = payload.get(container_key)
        if container_key == "realtime_vision" and container is not None:
            candidates.append(container)
        if isinstance(container, Mapping):
            candidates.append(container)
            candidates.extend(_mapping_realtime_candidates(container))

    organs = payload.get("organs")
    if isinstance(organs, Mapping):
        for organ_key in ("eye", "vision"):
            organ_payload = organs.get(organ_key)
            if isinstance(organ_payload, Mapping):
                candidates.append(organ_payload)
                candidates.extend(_mapping_realtime_candidates(organ_payload))
    return candidates


def _live_vision_state_candidate(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    source = str(payload.get("source") or "").strip().lower()
    driver = str(payload.get("driver") or "").strip().lower()
    backend = str(payload.get("backend") or "").strip()
    has_vision_fields = any(
        key in payload
        for key in (
            "frame_captured_at_ts",
            "frame_updated_at_ts",
            "state_age_s",
            "detections",
            "objects",
            "scene",
            "events",
        )
    )
    if not has_vision_fields:
        return None
    if source not in {"vision_state", "realtime_vision", "eye.realtime"} and driver != "vision_state" and not backend:
        return None

    raw_status = _normalized_payload_text(payload.get("status"))
    if raw_status in {"not_wired", "offline", "missing", "placeholder", "unavailable", "sleeping"}:
        return None
    status = "tracking" if raw_status in {"", "ok", "live", "ready", "running"} else raw_status
    captured_at = (
        payload.get("frame_captured_at_ts")
        or payload.get("state_updated_at_ts")
        or payload.get("frame_updated_at_ts")
        or payload.get("captured_at_ts")
        or payload.get("timestamp")
    )
    frame_id = str(payload.get("frame_id") or "")
    if not frame_id and captured_at not in (None, ""):
        captured_at_number = _optional_float(captured_at)
        if captured_at_number is not None:
            frame_id = f"vision-state-{int(captured_at_number * 1000)}"
    candidate = {
        str(key): value
        for key, value in payload.items()
        if str(key)
        not in {
            "kind",
            "mode",
            "primary_mode",
            "schema",
            "source",
            "driver",
            "status",
        }
    }
    if captured_at not in (None, ""):
        candidate["captured_at_ts"] = captured_at
        candidate["last_frame_captured_at_ts"] = captured_at
    if payload.get("state_age_s") not in (None, ""):
        candidate["last_frame_age_s"] = payload.get("state_age_s")
    if frame_id:
        candidate["frame_id"] = frame_id
    if payload.get("frame_path"):
        candidate["frame"] = {
            "path": payload.get("frame_path"),
            "captured_at_ts": captured_at,
        }
    return {
        "schema": "eihead.eye.realtime_status.v1",
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "primary_mode": True,
        "source": "vision_state_live",
        "status": status,
        "stream_ready": True,
        "not_wired": False,
        "backend": backend or source or "vision_state",
        **candidate,
    }


def _simulator_vision_state_candidate(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    if not (_truthy_payload_flag(payload.get("simulated")) or _truthy_payload_flag(payload.get("replay"))):
        return None
    source = str(payload.get("source") or "").strip()
    if not source or source == "vision_state":
        source = "vision_replay_simulator"
    candidate = {
        str(key): value
        for key, value in payload.items()
        if str(key) not in {"schema", "kind", "mode", "primary_mode", "source"}
    }
    return {
        "schema": "eihead.eye.realtime_status.v1",
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "primary_mode": True,
        "source": source,
        "status": payload.get("status") or "tracking",
        **candidate,
    }


def _is_native_provider_service(provider: Any) -> bool:
    if provider is None or isinstance(provider, Mapping):
        return False
    return any(callable(getattr(provider, method_name, None)) for method_name in ("poll", "status")) or hasattr(
        provider,
        "latest_status",
    )


def _pending_native_provider_service_status(provider: Any) -> dict[str, Any]:
    provider_name = provider.__class__.__name__ if provider is not None else "native_provider_service"
    return {
        "status": "unknown",
        "provider": provider_name,
        "reason": "native_provider_service_status_pending",
    }


def _native_provider_service_status(service: Any) -> Any | None:
    latest_status = getattr(service, "latest_status", None)
    if latest_status is not None:
        return latest_status

    for method_name in ("status", "poll"):
        method = getattr(service, method_name, None)
        if not callable(method):
            continue
        try:
            return method()
        except TypeError:
            continue
    return None


def _load_optional_eihead_config(path: str) -> Any | None:
    filename = path.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename.startswith("eihead"):
        return None
    try:
        from .config import EiheadConfigError, load_eihead_config

        return load_eihead_config(path)
    except (EiheadConfigError, OSError):
        return None


def _neck_reframe_settings_from_config(config: Any | None) -> tuple[ReframeConfig, bool, float, bool]:
    raw = getattr(config, "raw", None)
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    reframe_payload = _mapping_from_any(
        payload.get("neck_reframe")
        or payload.get("visual_reframe")
        or _mapping_value(payload.get("runtime"), "neck_reframe")
        or _mapping_value(payload.get("neck"), "reframe")
    )
    config_payload = _mapping_from_any(reframe_payload.get("config"))
    if not config_payload:
        config_payload = dict(reframe_payload)
    defaults = ReframeConfig()

    reframe_config = ReframeConfig(
        pan_min_deg=_env_float("EIHEAD_NECK_REFRAME_PAN_MIN_DEG", config_payload.get("pan_min_deg"), defaults.pan_min_deg),
        pan_max_deg=_env_float("EIHEAD_NECK_REFRAME_PAN_MAX_DEG", config_payload.get("pan_max_deg"), defaults.pan_max_deg),
        home_pan_deg=_env_float("EIHEAD_NECK_REFRAME_HOME_PAN_DEG", config_payload.get("home_pan_deg"), defaults.home_pan_deg),
        comfort_min_x=_env_float("EIHEAD_NECK_REFRAME_COMFORT_MIN_X", config_payload.get("comfort_min_x"), defaults.comfort_min_x),
        comfort_max_x=_env_float("EIHEAD_NECK_REFRAME_COMFORT_MAX_X", config_payload.get("comfort_max_x"), defaults.comfort_max_x),
        clear_min_x=_env_float("EIHEAD_NECK_REFRAME_CLEAR_MIN_X", config_payload.get("clear_min_x"), defaults.clear_min_x),
        clear_max_x=_env_float("EIHEAD_NECK_REFRAME_CLEAR_MAX_X", config_payload.get("clear_max_x"), defaults.clear_max_x),
        min_face_width_px=_env_int("EIHEAD_NECK_REFRAME_MIN_FACE_WIDTH_PX", config_payload.get("min_face_width_px"), defaults.min_face_width_px),
        min_face_height_px=_env_int("EIHEAD_NECK_REFRAME_MIN_FACE_HEIGHT_PX", config_payload.get("min_face_height_px"), defaults.min_face_height_px),
        min_crop_aspect=_env_float("EIHEAD_NECK_REFRAME_MIN_CROP_ASPECT", config_payload.get("min_crop_aspect"), defaults.min_crop_aspect),
        max_crop_aspect=_env_float("EIHEAD_NECK_REFRAME_MAX_CROP_ASPECT", config_payload.get("max_crop_aspect"), defaults.max_crop_aspect),
        min_confidence=_env_float("EIHEAD_NECK_REFRAME_MIN_CONFIDENCE", config_payload.get("min_confidence"), defaults.min_confidence),
        confirm_frames=_env_int("EIHEAD_NECK_REFRAME_CONFIRM_FRAMES", config_payload.get("confirm_frames"), defaults.confirm_frames),
        reframe_step_deg=_env_float("EIHEAD_NECK_REFRAME_STEP_DEG", config_payload.get("reframe_step_deg"), defaults.reframe_step_deg),
        return_step_deg=_env_float("EIHEAD_NECK_REFRAME_RETURN_STEP_DEG", config_payload.get("return_step_deg"), defaults.return_step_deg),
        min_command_interval_s=_env_float(
            "EIHEAD_NECK_REFRAME_MIN_COMMAND_INTERVAL_S",
            config_payload.get("min_command_interval_s"),
            defaults.min_command_interval_s,
        ),
        observe_hold_s=_env_float("EIHEAD_NECK_REFRAME_OBSERVE_HOLD_S", config_payload.get("observe_hold_s"), defaults.observe_hold_s),
        cooldown_s=_env_float("EIHEAD_NECK_REFRAME_COOLDOWN_S", config_payload.get("cooldown_s"), defaults.cooldown_s),
        anti_oscillation_cooldown_s=_env_float(
            "EIHEAD_NECK_REFRAME_ANTI_OSCILLATION_COOLDOWN_S",
            config_payload.get("anti_oscillation_cooldown_s"),
            defaults.anti_oscillation_cooldown_s,
        ),
    )
    enabled = _env_bool("EIHEAD_NECK_REFRAME_ENABLED", reframe_payload.get("enabled"), False)
    interval_s = _env_float("EIHEAD_NECK_REFRAME_INTERVAL_S", reframe_payload.get("interval_s"), 0.5)
    require_voice_awake = _env_bool(
        "EIHEAD_NECK_REFRAME_REQUIRE_VOICE_AWAKE",
        reframe_payload.get("require_voice_awake"),
        False,
    )
    return reframe_config, enabled, interval_s, require_voice_awake


def _mapping_from_any(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _env_bool(name: str, configured: Any, default: bool) -> bool:
    if name in os.environ:
        return _truthy_payload_flag(os.environ.get(name))
    if configured is None:
        return default
    return _truthy_payload_flag(configured)


def _env_float(name: str, configured: Any, default: float) -> float:
    raw = os.environ.get(name) if name in os.environ else configured
    value = _optional_float(raw)
    return float(default if value is None else value)


def _env_int(name: str, configured: Any, default: int) -> int:
    raw = os.environ.get(name) if name in os.environ else configured
    value = _optional_int(raw)
    return int(default if value is None else value)


def _event_trace_id(event: Mapping[str, Any] | Any) -> str | None:
    if isinstance(event, Mapping):
        return _optional_string(event.get("traceId") or event.get("trace_id"))

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return _optional_string(payload.get("traceId") or payload.get("trace_id"))

    return _optional_string(getattr(event, "trace_id", None) or getattr(event, "traceId", None))


def _action_from_event_route(route: Mapping[str, Any]) -> dict[str, Any]:
    action_id = _string_or_default(route.get("actionId"), "")
    action_type = _string_or_default(route.get("actionType"), "")
    target = _string_or_default(route.get("target"), "")
    params = _params_with_action_aliases(route.get("params"), action_type=action_type)
    action: dict[str, Any] = {
        "id": action_id,
        "action_id": action_id,
        "type": action_type,
        "action_type": action_type,
        "target": target,
        "params": params,
        "risk_level": _string_or_default(route.get("riskLevel"), ""),
        "idempotency_key": _string_or_default(route.get("idempotencyKey"), ""),
    }
    if target:
        action["target_name"] = target
    return action


def _params_with_action_aliases(params: Any, *, action_type: str) -> dict[str, Any]:
    normalized = dict(params) if isinstance(params, Mapping) else {}
    for key, value in list(normalized.items()):
        snake_key = _camel_to_snake(str(key))
        normalized.setdefault(snake_key, value)

    if action_type == "move_head" and "target_angle" in normalized:
        normalized.setdefault("angle", normalized["target_angle"])
    return normalized


def _camel_to_snake(text: str) -> str:
    result: list[str] = []
    for index, char in enumerate(text):
        if char.isupper() and index > 0 and text[index - 1] != "_":
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _action_outcome_reason(outcome: Mapping[str, Any]) -> str:
    details = outcome.get("details")
    if isinstance(details, Mapping):
        return _string_or_default(details.get("reason"), "")
    return ""


def _action_value(action: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in action:
        return action[key]
    params = action.get("params")
    if isinstance(params, Mapping) and key in params:
        return params[key]
    return default


def _action_metadata(action: Mapping[str, Any]) -> dict[str, Any]:
    metadata = action.get("metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    params = action.get("params")
    if isinstance(params, Mapping) and isinstance(params.get("metadata"), Mapping):
        return dict(params["metadata"])
    return {}


def _native_neck_reason(plan: Mapping[str, Any], servo_details: Mapping[str, Any]) -> str:
    servo_reason = _string_or_default(servo_details.get("reason"), "")
    if servo_reason:
        return servo_reason
    return _string_or_default(plan.get("reason"), "")


def _neck_reframe_dispatch_failure_reason(dispatch: Mapping[str, Any]) -> str:
    details = dispatch.get("details")
    if not isinstance(details, Mapping):
        return ""
    servo_details = details.get("neck_servo")
    if not isinstance(servo_details, Mapping):
        return ""
    reason = _string_or_default(servo_details.get("reason"), "")
    if reason:
        return reason
    status = _string_or_default(servo_details.get("status"), "")
    if status == "unavailable":
        return "neck_servo_adapter_unavailable"
    if status == "error":
        return "neck_servo_adapter_error"
    return status


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _string_or_default(value: Any, default: str) -> str:
    text = _optional_string(value)
    return text if text is not None else default


def _serialize_outcome(outcome: Any) -> dict[str, Any]:
    if isinstance(outcome, Mapping):
        return dict(outcome)
    if hasattr(outcome, "to_dict") and callable(outcome.to_dict):
        return dict(outcome.to_dict())
    if is_dataclass(outcome):
        return asdict(outcome)
    return {"value": outcome}


def _is_realtime_vision_payload(
    payload: Any,
    *,
    now_ts: float,
    max_age_seconds: float,
) -> bool:
    if payload is None:
        return False
    data = _payload_mapping(payload)
    if data is None:
        return False

    kind = _normalized_payload_text(data.get("kind"))
    mode = _normalized_payload_text(data.get("mode"))
    status = _normalized_payload_text(data.get("status"))
    schema = _normalized_payload_text(data.get("schema"))
    source = _normalized_payload_text(data.get("source"))

    if (
        _truthy_payload_flag(data.get("not_wired"))
        or _truthy_payload_flag(data.get("placeholder"))
        or status in {"not_wired", "offline", "missing", "placeholder", "unavailable"}
    ):
        return False
    if (
        kind == "vision_observation"
        or data.get("primary_mode") is False
        or _truthy_payload_flag(data.get("compatibility_mode"))
        or mode in {"compat", "compat/static", "static", "snapshot", "vision_state"}
        or "vision_state" in schema
        or source == "vision_state"
    ):
        return False
    if not (kind == "realtime_vision_observation" or mode in {"realtime", "realtime_stream"}):
        return False
    return _is_realtime_payload_fresh(data, now_ts=now_ts, max_age_seconds=max_age_seconds)


def _resolve_realtime_payload_candidate(payload: Any, *, seen: set[int] | None = None) -> Any:
    if payload is None:
        return None
    seen = seen or set()
    candidate_id = id(payload)
    if candidate_id in seen:
        return payload
    seen.add(candidate_id)

    for method_name in ("poll", "status"):
        method = getattr(payload, method_name, None)
        if not callable(method):
            continue
        try:
            resolved = _resolve_realtime_payload_candidate(method(), seen=seen)
        except TypeError:
            continue
        if resolved is not None:
            return resolved

    latest_status = getattr(payload, "latest_status", None)
    if latest_status is not None:
        resolved = _resolve_realtime_payload_candidate(latest_status, seen=seen)
        if resolved is not None:
            return resolved

    return payload


def _payload_mapping(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        data = payload.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    if is_dataclass(payload):
        return asdict(payload)
    try:
        serialized = serialize_message(payload)
    except TypeError:
        return None
    return dict(serialized) if isinstance(serialized, Mapping) else None


def _with_realtime_device_payload(payload: Any) -> Any:
    data = _payload_mapping(payload)
    if data is None:
        return payload
    device_keys = ("camera_device", "hailo_device")
    devices = data.get("devices")
    if not isinstance(devices, Mapping):
        devices = {}
    else:
        devices = dict(devices)
    for key in device_keys:
        if key in data and key not in devices:
            devices[key] = data[key]
    if not devices or data.get("devices") == devices:
        return payload
    return {**data, "devices": devices}


def _is_realtime_payload_fresh(data: Mapping[str, Any], *, now_ts: float, max_age_seconds: float) -> bool:
    if max_age_seconds <= 0:
        return True
    capture_ts = _extract_realtime_capture_timestamp(data)
    if capture_ts is None:
        return True
    if now_ts < capture_ts:
        return True
    return now_ts - capture_ts <= max_age_seconds


def _extract_realtime_capture_timestamp(data: Mapping[str, Any]) -> float | None:
    for key in (
        "last_frame_captured_at_ts",
        "captured_at_ts",
        "timestamp_ms",
        "timestamp",
    ):
        value = _coerce_realtime_timestamp(data.get(key))
        if value is not None:
            return value

    stream = data.get("stream")
    if isinstance(stream, Mapping):
        for key in ("last_frame_captured_at_ts", "captured_at_ts", "timestamp_ms", "timestamp"):
            value = _coerce_realtime_timestamp(stream.get(key))
            if value is not None:
                return value
    health = data.get("health")
    if isinstance(health, Mapping):
        for key in ("last_frame_captured_at_ts", "captured_at_ts", "timestamp_ms", "timestamp"):
            value = _coerce_realtime_timestamp(health.get(key))
            if value is not None:
                return value
    return None


def _coerce_realtime_timestamp(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        timestamp = float(raw)
    except (TypeError, ValueError):
        return None
    absolute_timestamp = abs(timestamp)
    if absolute_timestamp <= 2_000_000_000:
        return timestamp
    if absolute_timestamp <= 2_000_000_000_000:
        return timestamp / 1000.0
    return None


def _normalized_payload_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _truthy_payload_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
