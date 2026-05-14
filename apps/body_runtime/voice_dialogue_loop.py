"""Continuous honjia voice dialogue loop."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any
from typing import TYPE_CHECKING, Iterable

from apps.body_runtime.app import BodyRuntimeApp
from apps.body_runtime.voice_chain_benchmark import summarize_voice_chain
from eibrain.cognition.realtime import (
    FastThinkEngine,
    InterruptionController,
    MemoryOrchestrator,
    RealtimeCognitiveScheduler,
    RealtimeTurnManager,
    ResponseArbiter,
    SpeechActionPlanner,
    TurnBlackboard,
)
from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
from eibrain.protocol.observations import AudioTranscriptFinal

try:  # Task A may not be present on every branch yet.
    from eibrain.body.realtime_voice import RealtimeVoiceSession
except Exception:  # pragma: no cover - compatibility shim
    RealtimeVoiceSession = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp


_ACTION_CAPABILITY_BY_KIND = {
    "play_speech_action": "speech.play",
    "stop_speech_action": "speech.stop",
    "move_head_action": "head.move",
}
_VOICE_CHAIN_BENCHMARK_TRACE_LIMIT = 20
_VOICE_CHAIN_BENCHMARK_TERMINAL_STATUSES = {
    "reply_ready",
    "reply_degraded",
    "no_reply",
    "no_transcript",
    "waiting_for_wake_word",
    "wake_acknowledged",
    "sleep_acknowledged",
    "short_transcript_ignored",
    "interrupted",
    "stale_round_blocked",
}


class _FallbackRealtimeVoiceSession:
    """Small local snapshot model used when Task A's session object is absent."""

    def __init__(
        self,
        *,
        session_id: str,
        actor_id: str,
        round_id: str,
        cancellation_token: str,
    ) -> None:
        self.session_id = session_id
        self.actor_id = actor_id
        self.round_id = round_id
        self.cancellation_token = cancellation_token
        self.phase = "idle"
        self.status = "idle"
        self.transcript_final = ""
        self.reply_text = ""
        self.interrupted = False
        self.interrupt_reason = ""
        self.started_at_s: float | None = None
        self.final_asr_at_s: float | None = None
        self.first_reply_at_s: float | None = None
        self.first_speech_at_s: float | None = None
        self.completed_at_s: float | None = None
        self.events: list[dict[str, object]] = []

    def start_listening(self, **_: object) -> None:
        self.started_at_s = time.perf_counter()
        self.phase = "listening"
        self.status = "waiting_for_audio"
        self._record("listening", "waiting_for_audio", lane="listening", event_type="listening_started")

    def finalize_transcript(self, text: str, **_: object) -> None:
        self.final_asr_at_s = time.perf_counter()
        self.transcript_final = text.strip()
        self.phase = "thinking_stream"
        self.status = "final_transcript"
        self._record(
            "thinking_stream",
            "final_transcript",
            transcript=self.transcript_final,
            lane="listening",
            event_type="asr_final",
        )

    def update_microfeedback(self, text: str, **_: object) -> None:
        self._record(
            self.phase,
            "microfeedback",
            detail=text.strip(),
            lane="fast_think",
            event_type="microfeedback",
            payload={"text": text.strip()},
        )

    def append_reply_delta(self, delta: str, **_: object) -> None:
        if self.first_reply_at_s is None:
            self.first_reply_at_s = time.perf_counter()
        self.reply_text += delta
        self.phase = "thinking_stream"
        self.status = "reply_delta"
        self._record(
            "thinking_stream",
            "reply_delta",
            reply_delta=delta,
            lane="slow_thinking",
            event_type="agent_think",
        )

    def start_speaking(self, **_: object) -> None:
        if self.first_speech_at_s is None:
            self.first_speech_at_s = time.perf_counter()
        self.phase = "speaking_stream"
        self.status = "speech_started"
        self._record("speaking_stream", "speech_started", lane="speaking", event_type="tts_started")

    def complete(self, *, status: str = "ok", **_: object) -> None:
        self.completed_at_s = time.perf_counter()
        self.phase = "completed"
        self.status = status
        self._record("completed", status, lane="complete", event_type="complete")

    def fail(self, error: str, **_: object) -> None:
        self.completed_at_s = time.perf_counter()
        self.phase = "error"
        self.status = "error"
        self._record("error", "error", detail=error, lane="complete", event_type="error")

    def interrupt(self, *, reason: str = "user_barge_in", **_: object) -> None:
        self.interrupted = True
        self.interrupt_reason = reason
        self.phase = "barge_in"
        self.status = "interrupted"
        self._record(
            "barge_in",
            "interrupted",
            detail=reason,
            lane="interrupt",
            event_type="interrupt",
            payload={"reason": reason},
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "round_id": self.round_id,
            "roundId": self.round_id,
            "cancellation_token": self.cancellation_token,
            "cancellationToken": self.cancellation_token,
            "phase": self.phase,
            "status": self.status,
            "transcript_final": self.transcript_final,
            "reply_text": self.reply_text,
            "interrupted": self.interrupted,
            "interrupt_reason": self.interrupt_reason,
            "latency_ms": self.latency_ms(),
            "event_count": len(self.events),
            "events": [dict(event) for event in self.events],
        }

    def latency_ms(self) -> dict[str, float]:
        started = self.started_at_s
        if started is None:
            return {}
        result: dict[str, float] = {}
        if self.final_asr_at_s is not None:
            result["final_asr"] = self._elapsed_ms(started, self.final_asr_at_s)
        if self.first_reply_at_s is not None:
            result["first_reply_token"] = self._elapsed_ms(started, self.first_reply_at_s)
        if self.first_speech_at_s is not None:
            result["first_speech"] = self._elapsed_ms(started, self.first_speech_at_s)
        if self.completed_at_s is not None:
            result["total"] = self._elapsed_ms(started, self.completed_at_s)
        return result

    def _record(
        self,
        phase: str,
        status: str,
        *,
        transcript: str = "",
        reply_delta: str = "",
        detail: str = "",
        lane: str,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "phase": phase,
                "status": status,
                "lane": lane,
                "event_type": event_type,
                "transcript": transcript,
                "reply_delta": reply_delta,
                "detail": detail,
                "at_s": time.perf_counter(),
                "round_id": self.round_id,
                "roundId": self.round_id,
                "cancellation_token": self.cancellation_token,
                "cancellationToken": self.cancellation_token,
                "payload": dict(payload or {}),
            }
        )

    @staticmethod
    def _elapsed_ms(start_s: float, end_s: float) -> float:
        return round(max(0.0, end_s - start_s) * 1000, 2)


class VoiceDialogueLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        cognitive_runtime: CognitiveRuntimeApp,
        chunk_count: int = 2,
        max_chunk_count: int = 4,
        min_chunk_count: int = 1,
        idle_interval_s: float = 0.5,
        empty_interval_s: float = 0.25,
        session_id: str = "voice-dialogue-loop",
        actor_id: str = "darrow",
        wake_word: str = "\u9e3f\u9014",
        sleep_word: str = "\u7ed3\u675f\u5bf9\u8bdd",
        initial_conversation_active: bool = False,
        engagement_writer: object | None = None,
        waking_phrase: str = "\u6211\u5728\u3002",
        sleeping_phrase: str = "\u597d\u7684\uff0c\u5148\u4f11\u606f\u3002",
        realtime_turn_manager: RealtimeTurnManager | None = None,
        fast_think_engine: FastThinkEngine | None = None,
        response_arbiter: ResponseArbiter | None = None,
        interruption_controller: InterruptionController | None = None,
        speech_action_planner: SpeechActionPlanner | None = None,
        realtime_cognitive_scheduler: RealtimeCognitiveScheduler | None = None,
        realtime_wake_source: object | None = None,
    ) -> None:
        self.body_runtime = body_runtime
        self.cognitive_runtime = cognitive_runtime
        self.wake_word = wake_word
        self.sleep_word = sleep_word
        self.waking_phrase = waking_phrase
        self.sleeping_phrase = sleeping_phrase
        self.conversation_active = bool(initial_conversation_active)
        self.engagement_writer = engagement_writer
        self.chunk_count = max(1, int(chunk_count))
        self.max_chunk_count = max(1, int(max_chunk_count))
        self.min_chunk_count = max(1, int(min_chunk_count))
        if self.min_chunk_count > self.max_chunk_count:
            self.min_chunk_count = self.max_chunk_count
        if self.chunk_count > self.max_chunk_count:
            self.chunk_count = self.max_chunk_count
        if self.chunk_count < self.min_chunk_count:
            self.chunk_count = self.min_chunk_count
        self._rolling_chunk_count = self.chunk_count
        self.idle_interval_s = idle_interval_s
        self.empty_interval_s = empty_interval_s
        self.session_id = session_id
        self.actor_id = actor_id
        self._last_engagement_state = self.conversation_active
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.realtime_turn_manager = realtime_turn_manager or RealtimeTurnManager()
        self.fast_think_engine = fast_think_engine or FastThinkEngine()
        self.response_arbiter = response_arbiter or ResponseArbiter()
        self.interruption_controller = interruption_controller or InterruptionController()
        self.speech_action_planner = speech_action_planner or SpeechActionPlanner()
        memory_service = getattr(self.cognitive_runtime, "memory", None)
        self.realtime_cognitive_scheduler = realtime_cognitive_scheduler or RealtimeCognitiveScheduler(
            turn_manager=self.realtime_turn_manager,
            arbiter=self.response_arbiter,
            interruption_controller=self.interruption_controller,
            memory_orchestrator=MemoryOrchestrator(memory_service=memory_service) if memory_service is not None else None,
        )
        self._turn_lock = threading.RLock()
        self._interrupted_round_count = 0
        self._last_microfeedback: dict[str, object] | None = None
        self._realtime_session: object | None = None
        self.realtime_wake_source = realtime_wake_source
        self._realtime_audio_last_error = ""
        self._voice_chain_benchmark_traces: list[dict[str, object]] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        payload = {
            "enabled": True,
            "running": True,
            "phase": "starting",
            "last_status": "starting",
            "last_error": "",
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
            "conversation_active": self.conversation_active,
            "last_reply": "",
        }
        payload.update(self._round_state_payload())
        self._start_realtime_wake_source()
        payload["realtime_audio"] = self._realtime_audio_payload()
        self.body_runtime.update_voice_dialogue_state(**payload)
        self._publish_engagement_state(phase="running" if self.conversation_active else "sleeping")
        self._thread = threading.Thread(target=self._run, name="voice-dialogue-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
        self._stop_realtime_wake_source()
        payload = {
            "running": False,
            "phase": "stopped",
            "last_status": "stopped",
            "conversation_active": self.conversation_active,
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
        }
        payload.update(self._round_state_payload())
        payload["realtime_audio"] = self._realtime_audio_payload()
        self.body_runtime.update_voice_dialogue_state(**payload)
        self._publish_engagement_state(
            phase="stopped",
            conversation_active=self.conversation_active,
            reason="loop_stopped",
        )

    def _start_realtime_wake_source(self) -> None:
        source = self.realtime_wake_source
        if self.conversation_active:
            return
        starter = getattr(source, "start", None)
        if not callable(starter):
            return
        try:
            starter()
            self._realtime_audio_last_error = ""
        except Exception as exc:  # pragma: no cover - hardware/runtime boundary
            self._realtime_audio_last_error = str(exc)

    def _pause_realtime_wake_source(self) -> None:
        source = self.realtime_wake_source
        if source is None:
            return
        pauser = getattr(source, "pause", None)
        if callable(pauser):
            try:
                pauser()
                return
            except Exception as exc:  # pragma: no cover - hardware/runtime boundary
                self._realtime_audio_last_error = str(exc)
                return
        self._stop_realtime_wake_source()

    def _resume_realtime_wake_source(self) -> None:
        source = self.realtime_wake_source
        if source is None:
            return
        resumer = getattr(source, "resume", None)
        if callable(resumer):
            try:
                resumer()
                self._realtime_audio_last_error = ""
                return
            except Exception as exc:  # pragma: no cover - hardware/runtime boundary
                self._realtime_audio_last_error = str(exc)
                return
        self._start_realtime_wake_source()

    def _stop_realtime_wake_source(self) -> None:
        source = self.realtime_wake_source
        stopper = getattr(source, "stop", None)
        if not callable(stopper):
            return
        try:
            stopper()
        except Exception as exc:  # pragma: no cover - hardware/runtime boundary
            self._realtime_audio_last_error = str(exc)

    def _realtime_audio_payload(self) -> dict[str, object]:
        source = self.realtime_wake_source
        if source is None:
            return {"enabled": False, "running": False, "last_error": self._realtime_audio_last_error}
        payload: dict[str, object] = {"enabled": True, "last_error": self._realtime_audio_last_error}
        snapshot = getattr(source, "snapshot", None)
        if callable(snapshot):
            try:
                value = snapshot()
                if isinstance(value, dict):
                    payload.update(value)
            except Exception as exc:  # pragma: no cover - diagnostic boundary
                payload["last_error"] = str(exc)
        payload.setdefault("running", False)
        return payload

    def _poll_realtime_wake_source(self, *, timeout_s: float = 0.0) -> AudioTranscriptFinal | None:
        source = self.realtime_wake_source
        if source is None:
            return None
        getter = getattr(source, "next_transcript", None)
        if not callable(getter):
            return None
        try:
            value = getter(timeout_s=max(0.0, float(timeout_s)))
        except TypeError:
            value = getter()
        except Exception as exc:  # pragma: no cover - runtime boundary
            self._realtime_audio_last_error = str(exc)
            return None
        return self._coerce_audio_observation(value)

    def _read_audio_observation(self, *, chunk_count: int) -> AudioTranscriptFinal:
        if self.realtime_wake_source is not None and not self.conversation_active:
            observation = self._poll_realtime_wake_source(timeout_s=min(0.05, self.empty_interval_s))
            if observation is not None:
                return observation
            return AudioTranscriptFinal(
                ts=time.time(),
                source="ear.realtime_wake",
                text="",
                session_id=self.session_id,
                actor_id=self.actor_id,
            )
        return self.body_runtime.transcribe_audio_window(
            chunk_count=chunk_count,
            session_id=self.session_id,
            actor_id=self.actor_id,
        )

    def _coerce_audio_observation(self, value: object) -> AudioTranscriptFinal | None:
        if value is None:
            return None
        if isinstance(value, AudioTranscriptFinal):
            return value
        if isinstance(value, dict):
            text = str(value.get("text") or value.get("transcript") or "")
            if not text:
                return None
            return AudioTranscriptFinal(
                ts=float(value.get("ts") or time.time()),
                source=str(value.get("source") or "ear.realtime_wake"),
                text=text,
                language=str(value.get("language") or "und"),
                session_id=str(value.get("session_id") or self.session_id),
                actor_id=str(value.get("actor_id") or self.actor_id),
                target_id=str(value.get("target_id") or "") or None,
            )
        text = str(getattr(value, "text", "") or getattr(value, "transcript", "") or "")
        if not text:
            return None
        return AudioTranscriptFinal(
            ts=float(getattr(value, "ts", 0.0) or time.time()),
            source=str(getattr(value, "source", "") or "ear.realtime_wake"),
            text=text,
            language=str(getattr(value, "language", "") or "und"),
            session_id=str(getattr(value, "session_id", "") or self.session_id),
            actor_id=str(getattr(value, "actor_id", "") or self.actor_id),
            target_id=str(getattr(value, "target_id", "") or "") or None,
        )

    def _publish_engagement_state(
        self,
        *,
        phase: str,
        conversation_active: bool | None = None,
        reason: str = "",
    ) -> None:
        if self.engagement_writer is None:
            return
        next_conversation_active = self.conversation_active if conversation_active is None else conversation_active
        if next_conversation_active == self._last_engagement_state and phase != "stopped":
            return
        try:
            self.engagement_writer.write(
                conversation_active=next_conversation_active,
                phase=phase,
                reason=reason,
                security_mode=False,
            )
            self._last_engagement_state = next_conversation_active
        except Exception:
            self.body_runtime.update_voice_dialogue_state(
                last_status="engagement_writer_error",
                phase="error",
                last_error="engagement write failed",
            )

    def request_interrupt(self, *, reason: str = "user_barge_in") -> dict[str, object]:
        with self._turn_lock:
            old_turn = self.realtime_turn_manager.current_turn()
            old_realtime_session = self._realtime_session
            interruption = self.interruption_controller.interrupt_and_start_new_round(
                self.realtime_turn_manager,
                reason=reason,
            )
            new_turn = self.realtime_turn_manager.current_turn()
            if old_turn is not None and old_turn.state == "interrupted":
                self._interrupted_round_count += 1
            self._last_microfeedback = None

        self._call_realtime(old_realtime_session, "interrupt", reason=reason, turn=old_turn)
        stop_started_s = time.perf_counter()
        stop_status, stop_error = self._dispatch_stop_speech()
        stop_dispatch_elapsed_ms = round(max(0.0, time.perf_counter() - stop_started_s) * 1000, 2)
        tts_stop_confirmed = self._tts_stop_confirmed(stop_status)
        interruption["stop_dispatch_elapsed_ms"] = stop_dispatch_elapsed_ms
        interruption["interrupt_to_tts_stop_ms"] = stop_dispatch_elapsed_ms if tts_stop_confirmed else None
        interruption["tts_stop_confirmed"] = tts_stop_confirmed
        interruption["tts_stop_within_300ms"] = tts_stop_confirmed and stop_dispatch_elapsed_ms <= 300.0
        self._publish_state(
            phase="listening" if self.conversation_active else "idle",
            last_status="interrupted",
            turn=new_turn,
            realtime_voice_session=old_realtime_session,
            interruption=interruption,
            stop_speech_status=stop_status,
            stop_speech_error=stop_error,
            stop_dispatch_elapsed_ms=stop_dispatch_elapsed_ms,
            interrupt_to_tts_stop_ms=stop_dispatch_elapsed_ms if tts_stop_confirmed else None,
            tts_stop_confirmed=tts_stop_confirmed,
            tts_stop_within_300ms=tts_stop_confirmed and stop_dispatch_elapsed_ms <= 300.0,
            interrupt_active=True,
            last_interrupt=interruption,
            last_error="",
        )
        return interruption

    def _start_round(self, *, reason: str) -> TurnBlackboard:
        with self._turn_lock:
            self._last_microfeedback = None
            return self.realtime_turn_manager.start_round(reason=reason)

    def _finalize_asr(
        self,
        turn: TurnBlackboard,
        transcript: str,
    ) -> TurnBlackboard:
        with self._turn_lock:
            if transcript.strip():
                microfeedback_started_s = time.perf_counter()
                partial = self.realtime_cognitive_scheduler.observe_partial(
                    transcript,
                    round_id=turn.round_id,
                    cancellation_token=turn.cancellation_token,
                    session_id=self.session_id,
                    actor_id=self.actor_id,
                )
                microfeedback_elapsed_ms = round(max(0.0, time.perf_counter() - microfeedback_started_s) * 1000, 2)
                self.realtime_cognitive_scheduler.observe_final(
                    transcript,
                    round_id=turn.round_id,
                    cancellation_token=turn.cancellation_token,
                    session_id=self.session_id,
                    actor_id=self.actor_id,
                )
                finalized = self.realtime_turn_manager.current_turn() or turn
                fast_payload = partial.get("fast") if isinstance(partial, dict) else {}
                deadline_ms = int(fast_payload.get("deadline_ms") or 500)
                self._last_microfeedback = {
                    "text": str(fast_payload.get("microfeedback") or ""),
                    "deadline_ms": deadline_ms,
                    "elapsed_ms": microfeedback_elapsed_ms,
                    "within_deadline": microfeedback_elapsed_ms <= float(deadline_ms),
                    "source": str(fast_payload.get("source") or "fast_think"),
                    "stable": bool(fast_payload.get("stable") is True),
                }
            else:
                finalized = self.realtime_turn_manager.finalize_asr(
                    round_id=turn.round_id,
                    cancellation_token=turn.cancellation_token,
                    asr_text=transcript,
                )
            return finalized

    def _is_current_turn(self, turn: TurnBlackboard) -> bool:
        with self._turn_lock:
            return self.realtime_turn_manager.is_current(
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
            )

    def _start_realtime_session(self, turn: TurnBlackboard) -> object:
        session_cls = RealtimeVoiceSession or _FallbackRealtimeVoiceSession
        session = session_cls(
            session_id=self.session_id,
            actor_id=self.actor_id,
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
        )
        self._call_realtime(session, "start_listening", turn=turn)
        with self._turn_lock:
            self._realtime_session = session
        return session

    def _realtime_session_for_turn(self, turn: TurnBlackboard | None = None) -> object | None:
        with self._turn_lock:
            session = self._realtime_session
        if session is None or turn is None:
            return session
        snapshot = self._realtime_snapshot(session)
        if snapshot.get("round_id") != turn.round_id:
            return None
        return session

    def _call_realtime(
        self,
        session: object | None,
        method_name: str,
        *args: object,
        turn: TurnBlackboard | None = None,
        **kwargs: object,
    ) -> None:
        if session is None:
            return
        method = getattr(session, method_name, None)
        if not callable(method):
            return
        if turn is not None:
            kwargs.setdefault("round_id", turn.round_id)
            kwargs.setdefault("cancellation_token", turn.cancellation_token)
        try:
            method(*args, **kwargs)
        except TypeError:
            try:
                method(*args)
            except (RuntimeError, ValueError, TypeError):
                return
        except (RuntimeError, ValueError):
            return

    def _realtime_snapshot(self, session: object | None) -> dict[str, object]:
        if session is None:
            return {}
        snapshot = getattr(session, "snapshot", None)
        if callable(snapshot):
            try:
                value = snapshot()
                if isinstance(value, dict):
                    return value
            except (RuntimeError, ValueError, TypeError):
                return {}
        return {}

    def _realtime_updates(
        self,
        session: object | None,
        *,
        last_reply_delta: str | None = None,
    ) -> dict[str, object]:
        snapshot = self._realtime_snapshot(session)
        if not snapshot:
            return {}
        events = list(snapshot.get("events", []) or [])
        if last_reply_delta is None:
            for event in reversed(events):
                if isinstance(event, dict) and event.get("reply_delta"):
                    last_reply_delta = str(event.get("reply_delta") or "")
                    break
        latency_ms = dict(snapshot.get("latency_ms", {}) or {})
        return {
            "realtime_session": snapshot,
            "realtime_events": events,
            "last_reply_delta": last_reply_delta or "",
            "closed_loop_state": self._closed_loop_state(snapshot, events),
            "realtime_latency_ms": latency_ms,
        }

    def _voice_chain_benchmark_for_payload(self, payload: dict[str, object]) -> dict[str, object] | None:
        trace = self._voice_chain_trace_from_payload(payload)
        if trace is None:
            return None
        trace_key = self._voice_chain_trace_key(trace)
        if trace_key is None:
            self._voice_chain_benchmark_traces.append(trace)
        else:
            for index, existing in enumerate(self._voice_chain_benchmark_traces):
                if self._voice_chain_trace_key(existing) == trace_key:
                    self._voice_chain_benchmark_traces[index] = trace
                    break
            else:
                self._voice_chain_benchmark_traces.append(trace)
        if len(self._voice_chain_benchmark_traces) > _VOICE_CHAIN_BENCHMARK_TRACE_LIMIT:
            del self._voice_chain_benchmark_traces[:-_VOICE_CHAIN_BENCHMARK_TRACE_LIMIT]
        summary = summarize_voice_chain(self._voice_chain_benchmark_traces)
        summary["traceLimit"] = _VOICE_CHAIN_BENCHMARK_TRACE_LIMIT
        summary["recentTraces"] = [dict(item) for item in self._voice_chain_benchmark_traces]
        return summary

    @staticmethod
    def _voice_chain_trace_key(trace: dict[str, object]) -> tuple[str, str] | None:
        round_id = str(trace.get("roundId") or "")
        status = str(trace.get("status") or "")
        if not round_id or not status:
            return None
        return round_id, status

    def _voice_chain_trace_from_payload(self, payload: dict[str, object]) -> dict[str, object] | None:
        status = str(payload.get("last_status") or "").lower()
        if status not in _VOICE_CHAIN_BENCHMARK_TERMINAL_STATUSES:
            return None
        stage_latency_ms = self._numeric_mapping(payload.get("last_stage_latency_ms"))
        if not stage_latency_ms:
            stage_latency_ms = {
                key: round(value * 1000.0, 3)
                for key, value in self._numeric_mapping(payload.get("last_latency_s")).items()
            }
        realtime_latency_ms = self._numeric_mapping(payload.get("realtime_latency_ms"))
        event_sample_metrics = self._event_sample_metrics(payload)
        interruption = payload.get("interruption")
        interruption_payload = interruption if isinstance(interruption, dict) else {}
        tts_stop_confirmed = bool(payload.get("tts_stop_confirmed") or interruption_payload.get("tts_stop_confirmed"))
        interrupt_stop_ms = (
            self._first_number(
                payload.get("interrupt_to_tts_stop_ms"),
                interruption_payload.get("interrupt_to_tts_stop_ms"),
            )
            if tts_stop_confirmed
            else None
        )
        interrupted = status == "interrupted" or bool(payload.get("interrupt_active"))
        stale_round_payload = payload.get("stale_round")
        round_leak = status == "stale_round_blocked" or isinstance(stale_round_payload, dict)
        if not stage_latency_ms and event_sample_metrics:
            stage_latency_ms = self._stage_latency_from_event_metrics(event_sample_metrics)
        else:
            stage_latency_ms = self._merge_stage_latency_metrics(stage_latency_ms, event_sample_metrics)
        if (
            not stage_latency_ms
            and not realtime_latency_ms
            and not event_sample_metrics
            and interrupt_stop_ms is None
            and not round_leak
            and not interrupted
        ):
            return None
        first_asr_partial_ms = self._first_number(
            payload.get("firstAsrPartialMs"),
            realtime_latency_ms.get("first_partial_asr"),
            realtime_latency_ms.get("partial_asr"),
            stage_latency_ms.get("listen_asr_partial"),
            event_sample_metrics.get("firstAsrPartialMs"),
        )
        asr_final_ms = self._first_number(
            payload.get("asrFinalMs"),
            realtime_latency_ms.get("final_asr"),
            realtime_latency_ms.get("asr_final"),
            stage_latency_ms.get("listen_asr"),
            stage_latency_ms.get("asr"),
            event_sample_metrics.get("asrFinalMs"),
        )
        if first_asr_partial_ms is None:
            first_asr_partial_ms = asr_final_ms
        first_llm_delta_ms = self._first_number(
            payload.get("firstLlmDeltaMs"),
            realtime_latency_ms.get("first_reply_token"),
            realtime_latency_ms.get("first_llm_delta"),
            stage_latency_ms.get("llm_first_delta"),
            stage_latency_ms.get("llm_first_token"),
            event_sample_metrics.get("firstLlmDeltaMs"),
            payload.get("firstTokenMs"),
        )
        first_token_ms = self._first_number(payload.get("firstTokenMs"), first_llm_delta_ms)
        first_tts_chunk_ms = self._first_number(
            payload.get("firstTtsChunkMs"),
            realtime_latency_ms.get("first_tts_chunk"),
            stage_latency_ms.get("tts_first_chunk"),
            event_sample_metrics.get("firstTtsChunkMs"),
        )
        first_audio_ms = self._first_number(
            payload.get("firstAudioMs"),
            realtime_latency_ms.get("first_speech"),
            realtime_latency_ms.get("first_audio"),
            realtime_latency_ms.get("firstAudioMs"),
            stage_latency_ms.get("first_audio"),
            stage_latency_ms.get("firstAudioMs"),
            event_sample_metrics.get("firstAudioMs"),
        )
        if first_audio_ms is None and ("listen_asr" in stage_latency_ms or "think" in stage_latency_ms):
            first_audio_ms = stage_latency_ms.get("listen_asr", 0.0) + stage_latency_ms.get("think", 0.0)
        if first_audio_ms is None:
            first_audio_ms = self._first_number(stage_latency_ms.get("total"), event_sample_metrics.get("totalMs"))
        if first_tts_chunk_ms is None:
            first_tts_chunk_ms = first_audio_ms
        trace: dict[str, object] = {
            "roundId": payload.get("round_id") or payload.get("current_round_id") or "",
            "cancellationToken": payload.get("cancellation_token") or payload.get("current_cancellation_token") or "",
            "status": payload.get("last_status") or "",
            "interrupted": interrupted,
            "roundLeak": round_leak,
        }
        if stage_latency_ms:
            trace["stageLatencyMs"] = dict(stage_latency_ms)
        streaming = self._streaming_trace_from_payload(payload)
        if streaming:
            trace["streaming"] = streaming
        if first_asr_partial_ms is not None:
            trace["firstAsrPartialMs"] = first_asr_partial_ms
        if asr_final_ms is not None:
            trace["asrFinalMs"] = asr_final_ms
        if first_llm_delta_ms is not None:
            trace["firstLlmDeltaMs"] = first_llm_delta_ms
        if first_token_ms is not None:
            trace["firstTokenMs"] = first_token_ms
        if first_tts_chunk_ms is not None:
            trace["firstTtsChunkMs"] = first_tts_chunk_ms
        if first_audio_ms is not None:
            trace["firstAudioMs"] = first_audio_ms
        if interrupt_stop_ms is not None:
            trace["interruptStopMs"] = interrupt_stop_ms
        return trace

    def _streaming_trace_from_payload(self, payload: dict[str, object]) -> dict[str, bool]:
        explicit = payload.get("streaming")
        closed_loop = payload.get("closed_loop_state")
        closed = closed_loop if isinstance(closed_loop, dict) else {}
        events = payload.get("realtime_events")
        event_list = events if isinstance(events, list) else []
        event_types = {
            str(event.get("event_type") or "")
            for event in event_list
            if isinstance(event, dict)
        }
        stage_latency_ms = self._numeric_mapping(payload.get("last_stage_latency_ms"))
        realtime_latency_ms = self._numeric_mapping(payload.get("realtime_latency_ms"))
        event_sample_metrics = self._event_sample_metrics(payload)
        explicit_mapping = explicit if isinstance(explicit, dict) else {}
        streaming = {
            "asrPartial": bool(
                explicit_mapping.get("asrPartial")
                or explicit_mapping.get("asr_partial")
                or payload.get("firstAsrPartialMs")
                or realtime_latency_ms.get("first_partial_asr")
                or stage_latency_ms.get("listen_asr_partial")
                or event_sample_metrics.get("firstAsrPartialMs")
                or closed.get("final_asr")
                or "asr_partial" in event_types
                or "asr_final" in event_types
            ),
            "asrFinal": bool(
                explicit_mapping.get("asrFinal")
                or explicit_mapping.get("asr_final")
                or payload.get("asrFinalMs")
                or realtime_latency_ms.get("final_asr")
                or stage_latency_ms.get("listen_asr")
                or event_sample_metrics.get("asrFinalMs")
                or closed.get("final_asr")
                or "asr_final" in event_types
            ),
            "llmDelta": bool(
                explicit_mapping.get("llmDelta")
                or explicit_mapping.get("llm_delta")
                or payload.get("firstLlmDeltaMs")
                or payload.get("firstTokenMs")
                or realtime_latency_ms.get("first_reply_token")
                or stage_latency_ms.get("llm_first_delta")
                or stage_latency_ms.get("llm_first_token")
                or event_sample_metrics.get("firstLlmDeltaMs")
                or payload.get("last_reply_delta")
                or closed.get("reply_delta")
                or "agent_think" in event_types
            ),
            "ttsChunk": bool(
                explicit_mapping.get("ttsChunk")
                or explicit_mapping.get("tts_chunk")
                or payload.get("firstTtsChunkMs")
                or stage_latency_ms.get("tts_first_chunk")
                or event_sample_metrics.get("firstTtsChunkMs")
                or closed.get("speaking")
                or "tts_chunk" in event_types
                or "tts_started" in event_types
            ),
            "playback": bool(
                explicit_mapping.get("playback")
                or explicit_mapping.get("playback_started")
                or payload.get("firstAudioMs")
                or realtime_latency_ms.get("first_speech")
                or stage_latency_ms.get("first_audio")
                or event_sample_metrics.get("firstAudioMs")
                or closed.get("speaking")
                or "tts_started" in event_types
            ),
        }
        return streaming if any(streaming.values()) else {}

    @classmethod
    def _merge_stage_latency_metrics(
        cls,
        stage_latency_ms: dict[str, float],
        event_sample_metrics: dict[str, float],
    ) -> dict[str, float]:
        result = dict(stage_latency_ms)
        for field, key in (
            ("firstAsrPartialMs", "listen_asr_partial"),
            ("asrFinalMs", "listen_asr"),
            ("firstLlmDeltaMs", "llm_first_delta"),
            ("firstTokenMs", "llm_first_token"),
            ("firstTtsChunkMs", "tts_first_chunk"),
            ("firstAudioMs", "first_audio"),
        ):
            value = cls._number_or_none(event_sample_metrics.get(field))
            if value is not None:
                result.setdefault(key, value)
        total_ms = cls._number_or_none(event_sample_metrics.get("totalMs"))
        if total_ms is not None:
            result.setdefault("total", total_ms)
        elif "firstAudioMs" in event_sample_metrics:
            result.setdefault("total", float(event_sample_metrics["firstAudioMs"]))
        return result

    @classmethod
    def _stage_latency_from_event_metrics(cls, event_sample_metrics: dict[str, float]) -> dict[str, float]:
        return cls._merge_stage_latency_metrics({}, event_sample_metrics)

    @classmethod
    def _event_sample_metrics(cls, payload: dict[str, object]) -> dict[str, float]:
        events = payload.get("realtime_events")
        event_list = [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
        if not event_list:
            return {}
        typed_events: list[tuple[str, float]] = []
        for event in event_list:
            at_s = cls._number_or_none(event.get("at_s"))
            if at_s is None:
                continue
            typed_events.append((str(event.get("event_type") or ""), at_s))
        if not typed_events:
            return {}
        start_at_s = next((at_s for event_type, at_s in typed_events if event_type == "listening_started"), typed_events[0][1])
        result: dict[str, float] = {}
        for event_type, at_s in typed_events:
            elapsed_ms = round((at_s - start_at_s) * 1000.0, 3)
            if event_type == "asr_partial":
                result.setdefault("firstAsrPartialMs", elapsed_ms)
            elif event_type == "asr_final":
                result.setdefault("asrFinalMs", elapsed_ms)
            elif event_type == "agent_think":
                result.setdefault("firstLlmDeltaMs", elapsed_ms)
                result.setdefault("firstTokenMs", elapsed_ms)
            elif event_type == "tts_chunk":
                result.setdefault("firstTtsChunkMs", elapsed_ms)
            elif event_type == "tts_started":
                result.setdefault("firstAudioMs", elapsed_ms)
            elif event_type == "complete":
                result.setdefault("totalMs", elapsed_ms)
        return result

    @staticmethod
    def _numeric_mapping(value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, float] = {}
        for key, item in value.items():
            number = VoiceDialogueLoop._number_or_none(item)
            if number is not None:
                result[str(key)] = number
        return result

    @staticmethod
    def _first_number(*values: object) -> float | None:
        for value in values:
            number = VoiceDialogueLoop._number_or_none(value)
            if number is not None:
                return number
        return None

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _closed_loop_state(
        snapshot: dict[str, object],
        events: list[object],
    ) -> dict[str, bool]:
        event_types = {
            str(event.get("event_type", ""))
            for event in events
            if isinstance(event, dict)
        }
        phase = str(snapshot.get("phase", ""))
        return {
            "listening": "listening_started" in event_types or phase == "listening",
            "final_asr": "asr_final" in event_types or bool(snapshot.get("transcript_final")),
            "reply_delta": "agent_think" in event_types or bool(snapshot.get("reply_text")),
            "speaking": "tts_started" in event_types or bool(snapshot.get("first_speech_at_s")),
            "complete": "complete" in event_types or phase == "completed" or bool(snapshot.get("complete")),
            "error": "error" in event_types or phase == "error",
            "interrupted": "interrupt" in event_types or phase == "barge_in" or bool(snapshot.get("interrupted")),
        }

    def _round_state_payload(self, turn: TurnBlackboard | None = None) -> dict[str, object]:
        with self._turn_lock:
            current_turn = self.realtime_turn_manager.current_turn()
            selected_turn = turn or current_turn
            current_round_id = current_turn.round_id if current_turn is not None else ""
            current_cancellation_token = (
                current_turn.cancellation_token if current_turn is not None else ""
            )
            scheduler_state = self._scheduler_state_payload(selected_turn)
            interrupted_round_count = self._interrupted_round_count

        payload: dict[str, object] = {
            "current_round_id": current_round_id,
            "current_cancellation_token": current_cancellation_token,
            "scheduler_state": scheduler_state,
            "interrupted_round_count": interrupted_round_count,
            "microfeedback": dict(self._last_microfeedback or {}),
        }
        if selected_turn is not None:
            payload["round_id"] = selected_turn.round_id
            payload["cancellation_token"] = selected_turn.cancellation_token
        return payload

    def _scheduler_state_payload(self, selected_turn: TurnBlackboard | None) -> dict[str, object]:
        if selected_turn is None:
            return {
                "state": "idle",
                "status": "idle",
                "interrupted_round_count": self._interrupted_round_count,
                "lanes": {},
            }

        payload = selected_turn.to_dict()
        fast_hypotheses = list(payload.get("fast_hypotheses") or [])
        stable_decisions = list(payload.get("stable_decisions") or [])
        last_decision = stable_decisions[-1] if stable_decisions and isinstance(stable_decisions[-1], dict) else {}
        speech_plan = dict(payload.get("speech_plan") or {})
        if not speech_plan:
            speech_plan = {
                "round_id": selected_turn.round_id,
                "cancellation_token": selected_turn.cancellation_token,
                "stable": bool(payload.get("stable_speech_segments")),
                "speech_segments": list(payload.get("stable_speech_segments") or []),
                "action_plan": list(payload.get("action_plan") or []),
            }
        speech_plan.setdefault("plan_id", f"{selected_turn.round_id}:speech_action")
        speech_plan.setdefault("round_id", selected_turn.round_id)
        speech_plan.setdefault("cancellation_token", selected_turn.cancellation_token)
        speech_plan.setdefault("action_plan", list(payload.get("action_plan") or []))
        speech_plan.setdefault("actions", list(payload.get("action_plan") or []))

        scheduler_decision = selected_turn.safety_state.get("scheduler_decision")
        if isinstance(scheduler_decision, dict):
            proactive_activity = dict(scheduler_decision.get("proactive_activity") or {})
        else:
            proactive_activity = {}

        fast_state = "ready" if fast_hypotheses else "waiting"
        slow_state = "ready" if stable_decisions else "waiting"
        arbiter_verdict = selected_turn.safety_state.get("arbiter_verdict")
        if isinstance(arbiter_verdict, dict):
            arbiter_state = str(arbiter_verdict.get("state") or "approved")
            can_speak = bool(arbiter_verdict.get("can_speak"))
        else:
            arbiter_state = "ready" if speech_plan.get("speech_segments") else "waiting"
            can_speak = arbiter_state == "ready"
        lanes = {
            "fast_think": {
                "state": fast_state,
                "status": fast_state,
                "hypothesis_count": len(fast_hypotheses),
                "last_hypothesis": fast_hypotheses[-1] if fast_hypotheses else {},
            },
            "slow_reasoner": {
                "state": slow_state,
                "status": slow_state,
                "decision_count": len(stable_decisions),
                "last_decision": last_decision,
            },
            "arbiter": {
                "state": arbiter_state,
                "status": arbiter_state,
                "current_round": selected_turn.round_id,
                "speech_plan_stable": bool(speech_plan.get("stable")),
                "can_speak": can_speak,
            },
        }
        payload.update(
            {
                "state": selected_turn.state,
                "status": selected_turn.state,
                "lanes": lanes,
                "fast_think": lanes["fast_think"],
                "slow_reasoner": lanes["slow_reasoner"],
                "arbiter": lanes["arbiter"],
                "speech_action_plan": speech_plan,
                "proactive_activity": proactive_activity,
                "interrupted_round_count": self._interrupted_round_count,
            }
        )
        return payload

    def _publish_stale_round(
        self,
        turn: TurnBlackboard,
        *,
        reason: str,
        last_transcript: str = "",
        last_reply: str = "",
        last_error: str = "",
        realtime_voice_session: object | None = None,
    ) -> None:
        self._publish_state(
            phase="idle",
            last_status="stale_round_blocked",
            turn=turn,
            realtime_voice_session=realtime_voice_session,
            last_transcript=last_transcript,
            last_reply=last_reply,
            last_error=last_error,
            stale_round={
                "round_id": turn.round_id,
                "cancellation_token": turn.cancellation_token,
                "state": turn.state,
                "reason": reason,
            },
        )

    def _dispatch_stop_speech(self) -> tuple[str, str]:
        action = StopSpeechAction(
            ts=time.time(),
            source="voice_dialogue_loop",
            session_id=self.session_id,
            actor_id=self.actor_id,
            reason="voice_loop_interrupt",
            details={"reason": "voice_loop_interrupt"},
        )
        try:
            outcomes = self.body_runtime.dispatch_actions([action])
        except Exception as exc:
            return "unsupported", str(exc)
        if outcomes:
            statuses = [str(getattr(outcome, "status", "") or "") for outcome in outcomes]
            ok_statuses = {"ok", "healthy", "completed", "stopped"}
            if all(status in ok_statuses for status in statuses):
                return "ok", ""
            first_status = next((status for status in statuses if status), "failed")
            details = getattr(outcomes[0], "details", None)
            error = ""
            if isinstance(details, dict):
                error = str(details.get("last_error") or details.get("error") or details.get("reason") or "")
            return first_status, error
        return "not_supported", ""

    def _tts_stop_confirmed(self, stop_status: str) -> bool:
        if stop_status not in {"ok", "healthy", "completed", "stopped"}:
            return False
        is_speaking = getattr(self.body_runtime, "is_speaking", None)
        if callable(is_speaking):
            try:
                return not bool(is_speaking())
            except Exception:
                return stop_status in {"completed", "stopped"}
        return stop_status in {"completed", "stopped"}

    def _maybe_interrupt_during_playback(self) -> bool:
        probe = getattr(self.body_runtime, "probe_barge_in", None)
        if not callable(probe):
            return False
        result = probe(session_id=self.session_id, actor_id=self.actor_id)
        if isinstance(result, dict) and result.get("detected"):
            self.request_interrupt(reason=str(result.get("reason") or "playback_barge_in"))
            return True
        return False

    def _actions_allowed_for_turn(
        self,
        turn: TurnBlackboard,
        actions: list[object],
        reply: str,
    ) -> bool:
        if not self._is_current_turn(turn):
            return False
        if not actions or not reply:
            return True
        plan = self.speech_action_planner.plan(turn, speech_text=reply)
        action_segments = self._action_segments_for_gate(actions)
        plan["action_segments"] = action_segments
        plan["actions"] = action_segments
        plan["action_plan"] = action_segments
        plan["actionSegments"] = action_segments
        turn.speech_plan = plan
        turn.action_plan = action_segments
        with self._turn_lock:
            allowed = self.response_arbiter.allow_speaking(
                self.realtime_turn_manager,
                turn,
                plan,
            )
            turn.safety_state["arbiter_verdict"] = {
                "state": "approved" if allowed else "blocked",
                "status": "approved" if allowed else "blocked",
                "can_speak": allowed,
                "reason": "approved" if allowed else "response_arbiter_rejected_plan",
                "speech_segment_count": len(plan.get("speech_segments", []) or []),
                "action_segment_count": len(action_segments),
                "round_id": turn.round_id,
                "cancellation_token": turn.cancellation_token,
            }
            return allowed

    @staticmethod
    def _action_segments_for_gate(actions: list[object]) -> list[dict[str, object]]:
        segments: list[dict[str, object]] = []
        for index, action in enumerate(actions):
            payload = VoiceDialogueLoop._action_payload(action)
            kind = str(payload.get("kind") or payload.get("type") or action.__class__.__name__)
            capability_id = str(
                payload.get("capabilityId")
                or payload.get("capability_id")
                or _ACTION_CAPABILITY_BY_KIND.get(kind, kind)
            )
            segments.append(
                {
                    "capabilityId": capability_id,
                    "startOffsetMs": int(payload.get("startOffsetMs") or payload.get("start_offset_ms") or index * 120),
                    "durationMs": int(payload.get("durationMs") or payload.get("duration_ms") or 0),
                    "style": str(payload.get("style") or "default"),
                    "payload": payload,
                    "status": str(payload.get("status") or "ready"),
                }
            )
        return segments

    @staticmethod
    def _action_payload(action: object) -> dict[str, object]:
        if isinstance(action, dict):
            return dict(action)
        if hasattr(action, "to_dict"):
            try:
                value = action.to_dict()
                if isinstance(value, dict):
                    return dict(value)
            except Exception:
                pass
        if is_dataclass(action) and not isinstance(action, type):
            return dict(asdict(action))
        payload: dict[str, object] = {}
        for name in ("kind", "type", "source", "session_id", "actor_id", "target_id", "text", "reason"):
            if hasattr(action, name):
                value = getattr(action, name)
                if value is not None:
                    payload[name] = value
        return payload

    @staticmethod
    def _strip_trigger(text: str, trigger: str) -> tuple[str, bool]:
        value = text.strip()
        if not value or not trigger:
            return value, False
        if value == trigger:
            return "", True
        separators = r"[，,、\s:.!！?？:：]*"
        greeting_prefixes = ("你好", "您好", "嗨", "嘿", "哈喽", "hello")
        prefix_pattern = (
            r"(?:(?:"
            + "|".join(re.escape(prefix) for prefix in greeting_prefixes)
            + r")"
            + separators
            + r")?"
        )
        pattern = re.compile(
            r"^\s*" + prefix_pattern + re.escape(trigger) + separators,
            re.IGNORECASE,
        )
        match = pattern.match(value)
        if match is None:
            return value, False
        remainder = value[match.end() :].strip()
        return remainder, True

    def _publish_state(
        self,
        *,
        phase: str,
        last_status: str,
        turn: TurnBlackboard | None = None,
        realtime_voice_session: object | None = None,
        **updates: object,
    ) -> None:
        payload = {
            "phase": phase,
            "last_status": last_status,
            "running": True,
            "enabled": True,
            "wake_word": self.wake_word,
            "sleep_word": self.sleep_word,
            "conversation_active": self.conversation_active,
        }
        payload.update(self._round_state_payload(turn))
        session = realtime_voice_session or self._realtime_session_for_turn(turn)
        payload.update(self._realtime_updates(session))
        payload["realtime_audio"] = self._realtime_audio_payload()
        if last_status not in {"interrupted", "stale_round_blocked"}:
            payload.setdefault("interrupt_active", False)
        payload.update(updates)
        benchmark = self._voice_chain_benchmark_for_payload(payload)
        if benchmark is not None:
            payload["voice_chain_benchmark"] = benchmark
        payload.setdefault("last_reply", self.body_runtime.voice_dialogue_state.get("last_reply", ""))
        self.body_runtime.update_voice_dialogue_state(**payload)

    def _dispatch_ack_reply(self, text: str, turn: TurnBlackboard) -> bool:
        action = PlaySpeechAction(
            ts=time.time(),
            source="voice_dialogue_loop",
            session_id=self.session_id,
            actor_id=self.actor_id,
            text=text,
        )
        if not self._actions_allowed_for_turn(turn, [action], text):
            return False
        if not self._is_current_turn(turn):
            return False
        self.body_runtime.dispatch_actions([action])
        return True

    def _publish_stale_ack(
        self,
        turn: TurnBlackboard,
        *,
        reason: str,
        last_transcript: str,
    ) -> None:
        self._publish_stale_round(
            turn,
            reason=reason,
            last_transcript=last_transcript,
            last_reply="",
        )

    def _replace_transcript(
        self,
        observation: AudioTranscriptFinal,
        text: str,
    ) -> AudioTranscriptFinal:
        if observation.text == text:
            return observation
        return AudioTranscriptFinal(
            ts=observation.ts,
            source=observation.source,
            text=text,
            language=getattr(observation, "language", "und"),
            session_id=observation.session_id,
            actor_id=observation.actor_id,
            target_id=observation.target_id,
        )

    def _streaming_facade(self):
        for name in (
            "stream_observation",
            "handle_observation_stream",
            "stream_handle_observation",
            "stream_response",
        ):
            facade = getattr(self.cognitive_runtime, name, None)
            if callable(facade):
                return facade
        return None

    def _open_cognitive_stream(
        self,
        facade,
        observation: AudioTranscriptFinal,
        turn: TurnBlackboard,
        session: object,
    ) -> Iterable[object]:
        try:
            return facade(
                observation,
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
                realtime_session=session,
            )
        except TypeError:
            try:
                return facade(
                    observation,
                    round_id=turn.round_id,
                    cancellation_token=turn.cancellation_token,
                )
            except TypeError:
                return facade(observation)

    def _run_cognition_turn(
        self,
        observation: AudioTranscriptFinal,
        turn: TurnBlackboard,
        session: object,
    ) -> tuple[list[object], str, float, str, bool]:
        think_started = time.perf_counter()
        facade = self._streaming_facade()
        if facade is None:
            actions = list(self.cognitive_runtime.handle_observation(observation) or [])
            reply = self._reply_from_actions(actions)
            think_s = time.perf_counter() - think_started
            if not self._is_current_turn(turn):
                return actions, reply, think_s, "round_not_current_or_unstable", False
            self._record_scheduler_decision(observation, turn)
            return actions, reply, think_s, "", False

        actions: list[object] = []
        reply_parts: list[str] = []
        emitted_reply_delta = False
        for item in self._open_cognitive_stream(facade, observation, turn, session):
            if not self._is_current_turn(turn):
                return actions, "".join(reply_parts), time.perf_counter() - think_started, "streaming_round_not_current", emitted_reply_delta
            delta = self._reply_delta_from_stream_item(item)
            if delta:
                reply_parts.append(delta)
                emitted_reply_delta = True
                self._call_realtime(session, "append_reply_delta", delta, turn=turn)
                self._publish_state(
                    phase="thinking",
                    last_status="reply_delta",
                    turn=turn,
                    realtime_voice_session=session,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_reply="".join(reply_parts),
                    last_reply_delta=delta,
                    last_error="",
                )
            item_actions = self._actions_from_stream_item(item, observation)
            if item_actions:
                actions.extend(item_actions)

        reply = self._reply_from_actions(actions) or "".join(reply_parts)
        if not actions and reply:
            actions = [
                PlaySpeechAction(
                    ts=time.time(),
                    source="voice_dialogue_loop",
                    session_id=observation.session_id,
                    actor_id=observation.actor_id,
                    text=reply,
                )
            ]
        self._record_scheduler_decision(observation, turn)
        return actions, reply, time.perf_counter() - think_started, "", emitted_reply_delta

    def _record_scheduler_decision(self, observation: AudioTranscriptFinal, turn: TurnBlackboard) -> dict[str, object]:
        if not self._is_current_turn(turn):
            return {}
        try:
            result = self.realtime_cognitive_scheduler.decide(
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
                final_text=observation.text,
                session_id=observation.session_id or self.session_id,
                actor_id=observation.actor_id or self.actor_id,
            )
        except (RuntimeError, ValueError):
            return {}
        turn.safety_state["scheduler_decision"] = result
        return result

    @staticmethod
    def _reply_from_actions(actions: list[object]) -> str:
        for action in actions:
            if getattr(action, "kind", "") == "play_speech_action":
                return str(getattr(action, "text", "") or "")
        return ""

    @staticmethod
    def _reply_delta_from_stream_item(item: object) -> str:
        if isinstance(item, dict):
            item_type = str(item.get("type") or item.get("kind") or item.get("event") or item.get("event_type") or "")
            if item_type in {"reply_delta", "delta", "agent_think", "thinking_delta"}:
                return str(item.get("delta") or item.get("text") or item.get("reply_delta") or "")
            return ""
        item_type = str(
            getattr(item, "type", "")
            or getattr(item, "kind", "")
            or getattr(item, "event", "")
            or getattr(item, "event_type", "")
        )
        if item_type in {"reply_delta", "delta", "agent_think", "thinking_delta"}:
            return str(
                getattr(item, "delta", "")
                or getattr(item, "text", "")
                or getattr(item, "reply_delta", "")
                or ""
            )
        return ""

    @staticmethod
    def _actions_from_stream_item(item: object, observation: AudioTranscriptFinal | None = None) -> list[object]:
        if isinstance(item, list):
            return VoiceDialogueLoop._coerce_stream_actions(item, observation)
        if isinstance(item, dict):
            actions = item.get("actions")
            if isinstance(actions, list):
                return VoiceDialogueLoop._coerce_stream_actions(actions, observation)
            action = item.get("action")
            return VoiceDialogueLoop._coerce_stream_actions([action], observation) if action is not None else []
        if getattr(item, "kind", "") == "play_speech_action":
            return [item]
        actions = getattr(item, "actions", None)
        if isinstance(actions, list):
            return VoiceDialogueLoop._coerce_stream_actions(actions, observation)
        return []

    @staticmethod
    def _coerce_stream_actions(actions: list[object], observation: AudioTranscriptFinal | None) -> list[object]:
        coerced: list[object] = []
        for action in actions:
            if isinstance(action, dict):
                payload = action
                kind = str(payload.get("kind") or payload.get("type") or "")
                if kind != "play_speech_action":
                    continue
                text = str(payload.get("text") or payload.get("reply_text") or "")
                if not text:
                    continue
                coerced.append(
                    PlaySpeechAction(
                        ts=float(payload.get("ts") or time.time()),
                        source=str(payload.get("source") or "voice_dialogue_loop"),
                        session_id=str(payload.get("session_id") or getattr(observation, "session_id", "") or ""),
                        actor_id=str(payload.get("actor_id") or getattr(observation, "actor_id", "") or ""),
                        target_id=str(payload.get("target_id") or getattr(observation, "target_id", "") or ""),
                        text=text,
                    )
                )
                continue
            coerced.append(action)
        return coerced

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.body_runtime.is_speaking():
                    if self._maybe_interrupt_during_playback():
                        continue
                    self._publish_state(
                        phase="speaking",
                        last_status="playback_active",
                    )
                    self._sleep(self.idle_interval_s)
                    continue
                turn_started = time.perf_counter()
                turn = self._start_round(reason="listening")
                realtime_session = self._start_realtime_session(turn)
                self._publish_state(
                    phase="listening",
                    last_status="listening",
                    turn=turn,
                    realtime_voice_session=realtime_session,
                )
                listen_started = time.perf_counter()
                chunk_count = max(
                    self.min_chunk_count,
                    min(self.max_chunk_count, self._rolling_chunk_count),
                )
                observation = self._read_audio_observation(chunk_count=chunk_count)
                listen_asr_s = time.perf_counter() - listen_started
                transcript = observation.text.strip()
                try:
                    turn = self._finalize_asr(turn, transcript)
                except RuntimeError as exc:
                    self._call_realtime(realtime_session, "fail", "finalize_asr_rejected", turn=turn)
                    self._publish_stale_round(
                        turn,
                        reason="finalize_asr_rejected",
                        last_transcript=transcript,
                        last_error=str(exc),
                        realtime_voice_session=realtime_session,
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                self._call_realtime(realtime_session, "finalize_transcript", transcript, turn=turn)
                if self._last_microfeedback:
                    self._call_realtime(
                        realtime_session,
                        "update_microfeedback",
                        str(self._last_microfeedback.get("text") or ""),
                        turn=turn,
                    )

                if not self.conversation_active:
                    wake_transcript, woke = self._strip_trigger(transcript, self.wake_word)
                    if not woke:
                        if not transcript:
                            status = "no_transcript"
                            self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                        else:
                            status = "waiting_for_wake_word"
                            self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                        stage_latency_ms = self._stage_latency_ms(
                            listen_asr_s=listen_asr_s,
                            total_s=time.perf_counter() - turn_started,
                        )
                        self._call_realtime(realtime_session, "complete", status=status, turn=turn)
                        self._publish_state(
                            phase="idle",
                            last_status=status,
                            turn=turn,
                            realtime_voice_session=realtime_session,
                            conversation_active=False,
                            last_transcript=transcript,
                            last_latency_s={
                                "listen_asr": round(listen_asr_s, 2),
                                "total": round(time.perf_counter() - turn_started, 2),
                            },
                            last_stage_latency_ms=stage_latency_ms,
                            last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                            last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                        )
                        self._publish_engagement_state(
                            phase="sleeping",
                            conversation_active=False,
                            reason="waiting_for_wake_word",
                        )
                        self._sleep(self.empty_interval_s)
                        continue

                    self.conversation_active = True
                    self._pause_realtime_wake_source()
                    transcript = wake_transcript
                    observation = self._replace_transcript(observation, transcript)
                    self._publish_engagement_state(
                        phase="running",
                        conversation_active=True,
                        reason="wake_word_detected",
                    )
                    if not transcript:
                        self._publish_state(
                            phase="idle",
                            last_status="wake_acknowledged",
                            turn=turn,
                            conversation_active=True,
                            last_transcript=self.wake_word,
                            last_reply=self.waking_phrase,
                        )
                        think_started = time.perf_counter()
                        ack_dispatched = self._dispatch_ack_reply(self.waking_phrase, turn)
                        if not ack_dispatched:
                            self._publish_stale_ack(
                                turn,
                                reason="wake_ack_round_not_current",
                                last_transcript=self.wake_word,
                            )
                            self._sleep(self.empty_interval_s)
                            continue
                        self._call_realtime(realtime_session, "append_reply_delta", self.waking_phrase, turn=turn)
                        self._call_realtime(realtime_session, "start_speaking", turn=turn)
                        think_s = time.perf_counter() - think_started
                        total_s = time.perf_counter() - turn_started
                        stage_latency_ms = self._stage_latency_ms(
                            listen_asr_s=listen_asr_s,
                            think_s=think_s,
                            total_s=total_s,
                        )
                        self._call_realtime(realtime_session, "complete", status="wake_acknowledged", turn=turn)
                        self._publish_state(
                            phase="idle",
                            last_status="wake_acknowledged",
                            turn=turn,
                            realtime_voice_session=realtime_session,
                            conversation_active=True,
                            last_transcript=self.wake_word,
                            last_reply=self.waking_phrase,
                            last_reply_delta=self.waking_phrase,
                            last_latency_s={
                                "listen_asr": round(listen_asr_s, 2),
                                "think": round(think_s, 2),
                                "speak": 0.0,
                                "total": round(total_s, 2),
                            },
                            last_stage_latency_ms=stage_latency_ms,
                            last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                            last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                            last_error="",
                        )
                        self._sleep(self.empty_interval_s)
                        continue

                if not transcript:
                    stage_latency_ms = self._stage_latency_ms(
                        listen_asr_s=listen_asr_s,
                        total_s=time.perf_counter() - turn_started,
                    )
                    self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                    self._call_realtime(realtime_session, "complete", status="no_transcript", turn=turn)
                    self._publish_state(
                        phase="idle",
                        last_status="no_transcript",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        last_transcript="",
                        last_error="",
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                        last_stage_latency_ms=stage_latency_ms,
                        last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                        last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                transcript_for_cognitive, requested_sleep = self._strip_trigger(
                    transcript,
                    self.sleep_word,
                )
                if requested_sleep:
                    transcript = transcript_for_cognitive
                if requested_sleep and not transcript:
                    think_started = time.perf_counter()
                    self.conversation_active = False
                    self._resume_realtime_wake_source()
                    self._publish_engagement_state(
                        phase="stopped",
                        conversation_active=False,
                        reason="sleep_word_detected",
                    )
                    self._publish_state(
                        phase="idle",
                        last_status="sleep_acknowledged",
                        turn=turn,
                        conversation_active=False,
                        last_transcript=self.sleep_word,
                        last_reply=self.sleeping_phrase,
                    )
                    ack_dispatched = self._dispatch_ack_reply(self.sleeping_phrase, turn)
                    if not ack_dispatched:
                        self._publish_stale_ack(
                            turn,
                            reason="sleep_ack_round_not_current",
                            last_transcript=self.sleep_word,
                        )
                        self._sleep(self.empty_interval_s)
                        continue
                    self._call_realtime(realtime_session, "append_reply_delta", self.sleeping_phrase, turn=turn)
                    self._call_realtime(realtime_session, "start_speaking", turn=turn)
                    think_s = time.perf_counter() - think_started
                    self._call_realtime(realtime_session, "complete", status="sleep_acknowledged", turn=turn)
                    self._publish_state(
                        phase="idle",
                        last_status="sleep_acknowledged",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        conversation_active=False,
                        last_transcript=self.sleep_word,
                        last_reply=self.sleeping_phrase,
                        last_reply_delta=self.sleeping_phrase,
                        last_error="",
                        last_latency_s={
                            "listen_asr": round(listen_asr_s, 2),
                            "think": round(think_s, 2),
                            "speak": 0.0,
                            "total": round(time.perf_counter() - turn_started, 2),
                        },
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                if len(transcript_for_cognitive) <= 1:
                    self._rolling_chunk_count = max(self.min_chunk_count, chunk_count - 1)
                    self._call_realtime(realtime_session, "complete", status="short_transcript_ignored", turn=turn)
                    self._publish_state(
                        phase="idle",
                        last_status="short_transcript_ignored",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        last_transcript=transcript_for_cognitive,
                    )
                    self._sleep(self.empty_interval_s)
                    continue

                observation = self._replace_transcript(observation, transcript_for_cognitive)
                self._publish_state(
                    phase="thinking",
                    last_status="transcribed",
                    turn=turn,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_error="",
                )
                self._publish_state(
                    phase="thinking",
                    last_status="thinking",
                    turn=turn,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_error="",
                )
                actions, reply, think_s, stale_reason, emitted_reply_delta = self._run_cognition_turn(
                    observation,
                    turn,
                    realtime_session,
                )
                if stale_reason:
                    self._publish_stale_round(
                        turn,
                        reason=stale_reason,
                        last_transcript=observation.text,
                        realtime_voice_session=realtime_session,
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if not self._actions_allowed_for_turn(turn, actions, reply):
                    self._publish_stale_round(
                        turn,
                        reason="round_not_current_or_unstable",
                        last_transcript=observation.text,
                        realtime_voice_session=realtime_session,
                    )
                    self._sleep(self.empty_interval_s)
                    continue
                if reply and not emitted_reply_delta:
                    self._call_realtime(realtime_session, "append_reply_delta", reply, turn=turn)
                    self._publish_state(
                        phase="thinking",
                        last_status="reply_delta",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        conversation_active=True,
                        last_transcript=observation.text,
                        last_reply=reply,
                        last_reply_delta=reply,
                        last_error="",
                    )
                speak_started = time.perf_counter()
                if actions:
                    self._call_realtime(realtime_session, "start_speaking", turn=turn)
                    self._publish_state(
                        phase="speaking",
                        last_status="speaking_dispatch",
                        turn=turn,
                        realtime_voice_session=realtime_session,
                        conversation_active=True,
                        last_transcript=observation.text,
                        last_reply=reply,
                        last_error="",
                    )
                    if not self._is_current_turn(turn):
                        self._publish_stale_round(
                            turn,
                            reason="speaking_round_not_current",
                            last_transcript=observation.text,
                            realtime_voice_session=realtime_session,
                        )
                        self._sleep(self.empty_interval_s)
                        continue
                    outcomes = self.body_runtime.dispatch_actions(actions)
                    all_ok = bool(outcomes) and all(
                        getattr(outcome, "status", "") == "ok" for outcome in outcomes
                    )
                    status = "ok" if all_ok else "degraded"
                    self._rolling_chunk_count = min(self.max_chunk_count, chunk_count + 1)
                else:
                    outcomes = []
                    status = "no_reply"
                speak_s = time.perf_counter() - speak_started
                reply_status = "reply_ready" if reply and status == "ok" else "reply_degraded" if reply else status
                self._call_realtime(
                    realtime_session,
                    "complete",
                    status=reply_status,
                    turn=turn,
                )
                turn_count = int(self.body_runtime.voice_dialogue_state.get("turn_count", 0) or 0) + 1
                total_s = time.perf_counter() - turn_started
                stage_latency_ms = self._stage_latency_ms(
                    listen_asr_s=listen_asr_s,
                    think_s=think_s,
                    speak_s=speak_s,
                    total_s=total_s,
                )
                self._publish_state(
                    phase="idle",
                    last_status=reply_status,
                    turn=turn,
                    realtime_voice_session=realtime_session,
                    conversation_active=True,
                    last_transcript=observation.text,
                    last_reply=reply,
                    turn_count=turn_count,
                    last_error="" if status == "ok" else "speech_dispatch_degraded",
                    last_latency_s={
                        "listen_asr": round(listen_asr_s, 2),
                        "think": round(think_s, 2),
                        "speak": round(speak_s, 2),
                        "total": round(total_s, 2),
                    },
                    last_stage_latency_ms=stage_latency_ms,
                    last_bottleneck_stage=self._bottleneck_stage(stage_latency_ms),
                    last_bottleneck_ms=self._bottleneck_ms(stage_latency_ms),
                    streaming={
                        "asrPartial": True,
                        "llmDelta": emitted_reply_delta,
                        "ttsChunk": bool(actions and status == "ok"),
                    },
                    last_completed_turn={
                        "round_id": turn.round_id,
                        "cancellation_token": turn.cancellation_token,
                        "turn_count": turn_count,
                        "transcript": observation.text,
                        "reply": reply,
                        "status": status,
                        "latency_s": {
                            "listen_asr": round(listen_asr_s, 2),
                            "think": round(think_s, 2),
                            "speak": round(speak_s, 2),
                            "total": round(total_s, 2),
                        },
                        "stage_latency_ms": stage_latency_ms,
                        "bottleneck_stage": self._bottleneck_stage(stage_latency_ms),
                        "bottleneck_ms": self._bottleneck_ms(stage_latency_ms),
                        "completed_at_ts": time.time(),
                    },
                )
                self._sleep(self.idle_interval_s)
            except Exception as exc:  # pragma: no cover - runtime boundary
                session = self._realtime_session_for_turn()
                self._call_realtime(session, "fail", str(exc))
                payload = {
                    "phase": "error",
                    "last_status": "error",
                    "last_error": str(exc),
                }
                payload.update(self._realtime_updates(session))
                self.body_runtime.update_voice_dialogue_state(**payload)
                self._sleep(max(1.5, self.empty_interval_s))

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(max(0.0, seconds))

    @staticmethod
    def _stage_latency_ms(
        *,
        listen_asr_s: float,
        total_s: float,
        think_s: float = 0.0,
        speak_s: float = 0.0,
    ) -> dict[str, float]:
        stages = {
            "listen_asr": round(max(0.0, listen_asr_s) * 1000, 2),
            "think": round(max(0.0, think_s) * 1000, 2),
            "speak": round(max(0.0, speak_s) * 1000, 2),
            "total": round(max(0.0, total_s) * 1000, 2),
        }
        stages["overhead"] = round(
            max(0.0, stages["total"] - stages["listen_asr"] - stages["think"] - stages["speak"]),
            2,
        )
        return stages

    @classmethod
    def _bottleneck_stage(cls, stage_latency_ms: dict[str, float]) -> str:
        candidates = {
            key: value
            for key, value in stage_latency_ms.items()
            if key not in {"total", "overhead"}
        }
        if not candidates:
            return ""
        return max(candidates, key=candidates.get)

    @classmethod
    def _bottleneck_ms(cls, stage_latency_ms: dict[str, float]) -> float | None:
        stage = cls._bottleneck_stage(stage_latency_ms)
        if not stage:
            return None
        return float(stage_latency_ms.get(stage, 0.0))
