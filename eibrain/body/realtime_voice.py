"""Realtime voice session state for streaming dialogue."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


VOICE_PHASES = {
    "idle",
    "listening",
    "partial_asr",
    "thinking_stream",
    "speaking_stream",
    "barge_in",
    "completed",
    "error",
}

VOICE_LANES = {
    "listening",
    "fast_think",
    "slow_thinking",
    "speaking",
    "interrupt",
    "complete",
}

TERMINAL_PHASES = {"barge_in", "completed", "error"}


@dataclass(slots=True)
class RealtimeVoiceEvent:
    phase: str
    status: str
    transcript: str = ""
    reply_delta: str = ""
    detail: str = ""
    at_s: float = 0.0
    lane: str = "listening"
    event_type: str = ""
    round_id: str = ""
    cancellation_token: str = ""
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly event payload with scheduler aliases."""

        return {
            "phase": self.phase,
            "status": self.status,
            "lane": self.lane,
            "event_type": self.event_type,
            "transcript": self.transcript,
            "reply_delta": self.reply_delta,
            "detail": self.detail,
            "at_s": self.at_s,
            "round_id": self.round_id,
            "roundId": self.round_id,
            "cancellation_token": self.cancellation_token,
            "cancellationToken": self.cancellation_token,
            "payload": dict(self.payload),
        }


@dataclass(slots=True)
class RealtimeVoiceSession:
    """Tracks one realtime voice turn without owning audio or network I/O."""

    session_id: str
    actor_id: str
    clock: object = time.perf_counter
    round_id: str | None = None
    cancellation_token: str | None = None
    phase: str = "idle"
    status: str = "idle"
    transcript_partial: str = ""
    transcript_final: str = ""
    reply_text: str = ""
    microfeedback: str = ""
    interrupted: bool = False
    interrupt_reason: str = ""
    started_at_s: float | None = None
    phase_started_at_s: float | None = None
    first_audio_at_s: float | None = None
    first_partial_at_s: float | None = None
    first_microfeedback_at_s: float | None = None
    final_asr_at_s: float | None = None
    first_reply_at_s: float | None = None
    first_speech_at_s: float | None = None
    completed_at_s: float | None = None
    generation_cancelled: bool = False
    tts_stopped: bool = False
    action_plan_cancelled: bool = False
    cancellation_chain: list[dict[str, object]] = field(default_factory=list)
    events: list[RealtimeVoiceEvent] = field(default_factory=list)
    _round_sequence: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        self.round_id = self._stable_identifier(self.round_id, suffix="round")
        self.cancellation_token = self._stable_identifier(
            self.cancellation_token,
            suffix="token",
            seed=self.round_id,
        )

    def start_listening(
        self,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        fresh_round: bool = False,
    ) -> None:
        if fresh_round:
            self._rotate_round(round_id=round_id, cancellation_token=cancellation_token)
        else:
            self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        self.started_at_s = now_s
        self.first_audio_at_s = None
        self.first_partial_at_s = None
        self.first_microfeedback_at_s = None
        self.final_asr_at_s = None
        self.first_reply_at_s = None
        self.first_speech_at_s = None
        self.completed_at_s = None
        self.transcript_partial = ""
        self.transcript_final = ""
        self.reply_text = ""
        self.microfeedback = ""
        self.interrupted = False
        self.interrupt_reason = ""
        self.generation_cancelled = False
        self.tts_stopped = False
        self.action_plan_cancelled = False
        self.cancellation_chain.clear()
        if fresh_round:
            self.events.clear()
        self._transition(
            "listening",
            "waiting_for_audio",
            at_s=now_s,
            lane="listening",
            event_type="listening_started",
        )

    def note_audio(
        self,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        if self.first_audio_at_s is None:
            self.first_audio_at_s = now_s
        self._record("listening", "audio_detected", at_s=now_s, lane="listening", event_type="audio_detected")

    def update_partial_transcript(
        self,
        text: str,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        if self.first_partial_at_s is None:
            self.first_partial_at_s = now_s
        self.transcript_partial = text.strip()
        self._transition(
            "partial_asr",
            "partial_transcript",
            transcript=self.transcript_partial,
            at_s=now_s,
            lane="listening",
            event_type="asr_partial",
        )

    def finalize_transcript(
        self,
        text: str,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        self.final_asr_at_s = now_s
        self.transcript_final = text.strip()
        self._transition(
            "thinking_stream",
            "final_transcript",
            transcript=self.transcript_final,
            at_s=now_s,
            lane="listening",
            event_type="asr_final",
        )

    def update_microfeedback(
        self,
        text: str,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        if self.first_microfeedback_at_s is None:
            self.first_microfeedback_at_s = now_s
        self.microfeedback = text.strip()
        self._record(
            self.phase,
            "microfeedback",
            detail=self.microfeedback,
            at_s=now_s,
            lane="fast_think",
            event_type="microfeedback",
            payload={"text": self.microfeedback},
        )

    def append_reply_delta(
        self,
        delta: str,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        if self.first_reply_at_s is None:
            self.first_reply_at_s = now_s
        self.reply_text += delta
        self._transition(
            "thinking_stream",
            "reply_delta",
            reply_delta=delta,
            at_s=now_s,
            lane="slow_thinking",
            event_type="agent_think",
        )

    def start_speaking(
        self,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        if self.first_speech_at_s is None:
            self.first_speech_at_s = now_s
        self._transition(
            "speaking_stream",
            "speech_started",
            at_s=now_s,
            lane="speaking",
            event_type="tts_started",
        )

    def interrupt(
        self,
        *,
        reason: str = "user_barge_in",
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        self.interrupted = True
        self.interrupt_reason = reason
        self._transition(
            "barge_in",
            "interrupted",
            detail=reason,
            at_s=now_s,
            lane="interrupt",
            event_type="interrupt",
            payload={"reason": reason},
        )
        self._mark_cancellation_step("generation", "generation_cancelled", reason, at_s=now_s)
        self._mark_cancellation_step("tts", "tts_stopped", reason, at_s=now_s)
        self._mark_cancellation_step("action_plan", "action_plan_cancelled", reason, at_s=now_s)

    def mark_generation_cancelled(
        self,
        *,
        reason: str = "interrupt",
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_mismatch(round_id=round_id, cancellation_token=cancellation_token)
        self._mark_cancellation_step("generation", "generation_cancelled", reason, at_s=self._now())

    def mark_tts_stopped(
        self,
        *,
        reason: str = "interrupt",
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_mismatch(round_id=round_id, cancellation_token=cancellation_token)
        self._mark_cancellation_step("tts", "tts_stopped", reason, at_s=self._now())

    def mark_action_plan_cancelled(
        self,
        *,
        reason: str = "interrupt",
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_mismatch(round_id=round_id, cancellation_token=cancellation_token)
        self._mark_cancellation_step("action_plan", "action_plan_cancelled", reason, at_s=self._now())

    def record_stream_event(
        self,
        *,
        event_type: str,
        status: str,
        lane: str,
        payload: dict[str, object] | None = None,
        detail: str = "",
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        self._record(
            self.phase,
            status,
            detail=detail,
            at_s=self._now(),
            lane=lane,
            event_type=event_type,
            payload=payload,
        )

    def complete(
        self,
        *,
        status: str = "ok",
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        self.completed_at_s = now_s
        self._transition("completed", status, at_s=now_s, lane="complete", event_type="complete")

    def fail(
        self,
        error: str,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
    ) -> None:
        self._reject_if_supplied_stale(round_id=round_id, cancellation_token=cancellation_token)
        now_s = self._now()
        self.completed_at_s = now_s
        self._transition("error", "error", detail=error, at_s=now_s, lane="complete", event_type="error")

    def is_current(self, *, round_id: str, cancellation_token: str) -> bool:
        return (
            str(self.round_id) == round_id
            and str(self.cancellation_token) == cancellation_token
            and not self.interrupted
            and self.phase not in TERMINAL_PHASES
        )

    def reject_if_stale(self, *, round_id: str, cancellation_token: str) -> None:
        if not self.is_current(round_id=round_id, cancellation_token=cancellation_token):
            raise RuntimeError("round/token is not current or already cancelled")

    reject_if_cancelled = reject_if_stale

    def snapshot(self) -> dict[str, object]:
        complete = self._is_complete()
        closed_loop_state = self._closed_loop_state()
        return {
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "round_id": self.round_id,
            "roundId": self.round_id,
            "cancellation_token": self.cancellation_token,
            "cancellationToken": self.cancellation_token,
            "phase": self.phase,
            "status": self.status,
            "complete": complete,
            "closed_loop": self._is_closed_loop_complete(),
            "closed_loop_state": closed_loop_state,
            "transcript_partial": self.transcript_partial,
            "transcript_final": self.transcript_final,
            "reply_text": self.reply_text,
            "microfeedback": self.microfeedback,
            "interrupted": self.interrupted,
            "interrupt_reason": self.interrupt_reason,
            "first_partial_at_s": self.first_partial_at_s,
            "first_microfeedback_at_s": self.first_microfeedback_at_s,
            "first_speech_at_s": self.first_speech_at_s,
            "generation_cancelled": self.generation_cancelled,
            "tts_stopped": self.tts_stopped,
            "action_plan_cancelled": self.action_plan_cancelled,
            "cancellation_chain": [dict(item) for item in self.cancellation_chain],
            "latency_ms": self.latency_ms(),
            "event_count": len(self.events),
            "events": [event.to_dict() for event in self.events],
        }

    def latency_ms(self) -> dict[str, float]:
        started = self.started_at_s
        if started is None:
            return {}
        result: dict[str, float] = {}
        if self.first_audio_at_s is not None:
            result["audio_detect"] = self._elapsed_ms(started, self.first_audio_at_s)
        if self.first_partial_at_s is not None:
            result["first_partial_asr"] = self._elapsed_ms(started, self.first_partial_at_s)
        if self.first_microfeedback_at_s is not None:
            result["first_microfeedback"] = self._elapsed_ms(started, self.first_microfeedback_at_s)
        if self.final_asr_at_s is not None:
            result["final_asr"] = self._elapsed_ms(started, self.final_asr_at_s)
        if self.first_reply_at_s is not None:
            result["first_reply_token"] = self._elapsed_ms(started, self.first_reply_at_s)
        if self.first_speech_at_s is not None:
            result["first_speech"] = self._elapsed_ms(started, self.first_speech_at_s)
        if self.completed_at_s is not None:
            result["total"] = self._elapsed_ms(started, self.completed_at_s)
        if self.final_asr_at_s is not None and self.first_reply_at_s is not None:
            result["final_asr_to_first_reply_token"] = self._elapsed_ms(
                self.final_asr_at_s,
                self.first_reply_at_s,
            )
        if self.first_reply_at_s is not None and self.first_speech_at_s is not None:
            result["first_reply_token_to_first_speech"] = self._elapsed_ms(
                self.first_reply_at_s,
                self.first_speech_at_s,
            )
        if self.first_speech_at_s is not None and self.completed_at_s is not None:
            result["first_speech_to_complete"] = self._elapsed_ms(
                self.first_speech_at_s,
                self.completed_at_s,
            )
        return result

    def _rotate_round(
        self,
        *,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> None:
        self._round_sequence += 1
        self.round_id = self._stable_identifier(
            round_id,
            suffix=f"round-{self._round_sequence}",
        )
        self.cancellation_token = self._stable_identifier(
            cancellation_token,
            suffix="token",
            seed=self.round_id,
        )

    def _is_complete(self) -> bool:
        return self.phase == "completed"

    def _closed_loop_state(self) -> dict[str, bool]:
        return {
            "final_asr": self.final_asr_at_s is not None,
            "first_reply_delta": self.first_reply_at_s is not None,
            "first_speech": self.first_speech_at_s is not None,
            "complete": self._is_complete(),
        }

    def _is_closed_loop_complete(self) -> bool:
        return all(self._closed_loop_state().values())

    def _transition(
        self,
        phase: str,
        status: str,
        *,
        transcript: str = "",
        reply_delta: str = "",
        detail: str = "",
        at_s: float,
        lane: str | None = None,
        event_type: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        if phase not in VOICE_PHASES:
            raise ValueError(f"unknown voice phase: {phase}")
        self.phase = phase
        self.status = status
        self.phase_started_at_s = at_s
        self._record(
            phase,
            status,
            transcript=transcript,
            reply_delta=reply_delta,
            detail=detail,
            at_s=at_s,
            lane=lane,
            event_type=event_type,
            payload=payload,
        )

    def _record(
        self,
        phase: str,
        status: str,
        *,
        transcript: str = "",
        reply_delta: str = "",
        detail: str = "",
        at_s: float,
        lane: str | None = None,
        event_type: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        event_lane = lane or self._lane_for_phase(phase)
        if event_lane not in VOICE_LANES:
            raise ValueError(f"unknown voice lane: {event_lane}")
        self.events.append(
            RealtimeVoiceEvent(
                phase=phase,
                status=status,
                transcript=transcript,
                reply_delta=reply_delta,
                detail=detail,
                at_s=at_s,
                lane=event_lane,
                event_type=event_type or status,
                round_id=str(self.round_id),
                cancellation_token=str(self.cancellation_token),
                payload=dict(payload or {}),
            )
        )

    def _mark_cancellation_step(self, target: str, event_type: str, reason: str, *, at_s: float) -> None:
        flag_by_target = {
            "generation": "generation_cancelled",
            "tts": "tts_stopped",
            "action_plan": "action_plan_cancelled",
        }
        flag_name = flag_by_target[target]
        if getattr(self, flag_name):
            return
        setattr(self, flag_name, True)
        step = {
            "target": target,
            "event_type": event_type,
            "reason": reason,
            "at_s": at_s,
            "round_id": str(self.round_id),
            "roundId": str(self.round_id),
            "cancellation_token": str(self.cancellation_token),
            "cancellationToken": str(self.cancellation_token),
        }
        self.cancellation_chain.append(step)
        self._record(
            self.phase,
            event_type,
            detail=reason,
            at_s=at_s,
            lane="interrupt",
            event_type=event_type,
            payload={"target": target, "reason": reason},
        )

    def _reject_if_supplied_stale(
        self,
        *,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> None:
        if round_id is None and cancellation_token is None:
            return
        self.reject_if_stale(
            round_id=round_id if round_id is not None else str(self.round_id),
            cancellation_token=cancellation_token if cancellation_token is not None else str(self.cancellation_token),
        )

    def _reject_if_supplied_mismatch(
        self,
        *,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> None:
        if round_id is not None and round_id != str(self.round_id):
            raise RuntimeError("round/token does not belong to this session")
        if cancellation_token is not None and cancellation_token != str(self.cancellation_token):
            raise RuntimeError("round/token does not belong to this session")

    def _now(self) -> float:
        return float(self.clock())

    @staticmethod
    def _elapsed_ms(start_s: float, end_s: float) -> float:
        return round(max(0.0, end_s - start_s) * 1000, 2)

    def _stable_identifier(self, value: str | None, *, suffix: str, seed: str | None = None) -> str:
        text = (value or "").strip()
        if text:
            return text
        base = (seed or self.session_id or self.actor_id or "voice").strip() or "voice"
        return f"{base}-{suffix}"

    @staticmethod
    def _lane_for_phase(phase: str) -> str:
        if phase in {"listening", "partial_asr", "idle"}:
            return "listening"
        if phase == "thinking_stream":
            return "slow_thinking"
        if phase == "speaking_stream":
            return "speaking"
        if phase == "barge_in":
            return "interrupt"
        return "complete"
