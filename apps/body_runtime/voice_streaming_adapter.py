"""Apply voice-streaming eiprotocol events to realtime voice session state."""

from __future__ import annotations

from collections.abc import Mapping

from eibrain.body.realtime_voice import RealtimeVoiceSession
from eiprotocol.models import EventEnvelope


ASR_PARTIAL_EVENTS = {
    "ei.voice.asr.partial",
    "ei.dialogue.asr.partial",
}
ASR_FINAL_EVENTS = {
    "ei.voice.asr.final",
    "ei.dialogue.asr.final",
}
AUDIO_FRAME_EVENTS = {
    "ei.voice.audio.frame",
}
AGENT_DELTA_EVENTS = {
    "ei.dialogue.agent.delta",
}
TTS_CHUNK_EVENTS = {
    "ei.voice.tts.chunk",
}
SPEAKING_START_EVENTS = {
    "ei.voice.tts.sentence_start",
    "ei.voice.playback.started",
}
PLAYBACK_STOP_EVENTS = {
    "ei.voice.playback.stopped",
}
INTERRUPT_EVENTS = {
    "ei.voice.barge_in.detected",
    "ei.dialogue.interrupt.requested",
}
HEARTBEAT_EVENTS = {
    "ei.voice.session.heartbeat",
}

COMPLETED_PLAYBACK_REASONS = {"", "complete", "completed", "done", "ended", "eof"}


class VoiceStreamingAdapter:
    """Bridge eiprotocol voice-streaming events into a RealtimeVoiceSession."""

    def __init__(self, session: RealtimeVoiceSession) -> None:
        self.session = session
        self._audio_frames = 0
        self._tts_chunks = 0
        self._turn_summary: dict[str, dict[str, object]] = {
            "asrPartial": {"count": 0, "seen": False},
            "asrFinal": {"count": 0, "seen": False},
            "llmDelta": {"count": 0, "seen": False},
            "ttsChunk": {"count": 0, "seen": False},
            "playback": {"count": 0, "seen": False},
        }
        self._last_heartbeat: dict[str, object] = {}
        self._asr_trace: dict[str, object] = {
            "partial_count": 0,
            "final_count": 0,
            "last_event": {},
        }

    def apply(self, event: Mapping[str, object] | EventEnvelope) -> dict[str, object]:
        payload = self._event_payload(event)
        name = str(payload.get("name", "") or "")
        content = self._mapping(payload.get("content"))
        round_id = self._first_text(
            payload.get("roundId"),
            payload.get("round_id"),
            content.get("roundId"),
            content.get("round_id"),
        )
        session_id = self._first_text(
            payload.get("sessionId"),
            payload.get("session_id"),
            content.get("sessionId"),
            content.get("session_id"),
        )
        cancellation_token = self._first_text(
            payload.get("cancellationToken"),
            payload.get("cancellation_token"),
            content.get("cancellationToken"),
            content.get("cancellation_token"),
        )
        operation = self._apply_named_event(
            name,
            content,
            session_id=session_id,
            round_id=round_id,
            cancellation_token=cancellation_token,
        )
        duplicate = operation.startswith("duplicate_")
        return {
            "applied": operation not in {"ignored"} and not duplicate,
            "eventName": name,
            "name": name,
            "operation": operation,
            "duplicate": duplicate,
            "terminal": duplicate or self.session.phase in {"barge_in", "completed", "error"},
            "round_id": round_id,
            "roundId": round_id,
            "session_id": session_id,
            "sessionId": session_id,
            "cancellation_token": cancellation_token,
            "cancellationToken": cancellation_token,
            "content": dict(content),
            "live_trace": self._live_trace(),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "streaming": {
                "audio_frames": self._audio_frames,
                "tts_chunks": self._tts_chunks,
                "last_heartbeat_state": str(self._last_heartbeat.get("state", "") or ""),
            },
            "live_trace": self._live_trace(),
        }

    def _apply_named_event(
        self,
        name: str,
        content: Mapping[str, object],
        *,
        session_id: str | None,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> str:
        round_scope = {
            "round_id": round_id,
            "cancellation_token": cancellation_token,
        }
        if name in AUDIO_FRAME_EVENTS:
            self.session.note_audio(**round_scope)
            self._audio_frames += 1
            return "note_audio"
        if name in ASR_PARTIAL_EVENTS:
            self._note_turn_signal("asrPartial")
            self.session.update_partial_transcript(
                self._text_content(content, "text"),
                **round_scope,
            )
            self._record_asr_trace(name, content, session_id=session_id, round_id=round_id, final=False)
            return "update_partial_transcript"
        if name in ASR_FINAL_EVENTS:
            self._note_turn_signal("asrFinal")
            self.session.finalize_transcript(
                self._text_content(content, "text"),
                **round_scope,
            )
            self._record_asr_trace(name, content, session_id=session_id, round_id=round_id, final=True)
            return "finalize_transcript"
        if name in AGENT_DELTA_EVENTS:
            self._note_turn_signal("llmDelta")
            self.session.append_reply_delta(
                self._text_content(content, "delta", "text"),
                **round_scope,
            )
            return "append_reply_delta"
        if name in SPEAKING_START_EVENTS:
            if name == "ei.voice.playback.started":
                self._note_turn_signal("playback")
            self.session.start_speaking(**round_scope)
            return "start_speaking"
        if name in TTS_CHUNK_EVENTS:
            self._note_turn_signal("ttsChunk")
            self.session.record_stream_event(
                event_type="tts_chunk",
                status="tts_chunk_observed",
                lane="speaking",
                payload=dict(content),
                **round_scope,
            )
            self._tts_chunks += 1
            return "observe_tts_chunk"
        if name in PLAYBACK_STOP_EVENTS:
            reason = self._text_content(content, "reason", default="completed")
            if reason.strip().lower() in COMPLETED_PLAYBACK_REASONS and not self.session.interrupted:
                if self._is_duplicate_completed_stop(round_id, cancellation_token):
                    return "duplicate_playback_stop"
                self.session.complete(status="playback_completed", **round_scope)
                return "complete_playback"
            self.session.mark_tts_stopped(reason=reason or "playback_stopped", **round_scope)
            return "mark_tts_stopped"
        if name in INTERRUPT_EVENTS:
            self.session.interrupt(
                reason=self._text_content(content, "reason", default="user_barge_in"),
                **round_scope,
            )
            return "interrupt"
        if name in HEARTBEAT_EVENTS:
            self.session.record_stream_event(
                event_type="voice_heartbeat",
                status="voice_heartbeat_observed",
                lane="listening",
                payload=dict(content),
            )
            self._last_heartbeat = dict(content)
            return "observe_heartbeat"
        return "ignored"

    def _live_trace(self) -> dict[str, object]:
        return {
            "audio_frames": self._audio_frames,
            "tts_chunks": self._tts_chunks,
            "last_heartbeat": dict(self._last_heartbeat),
            "asr": dict(self._asr_trace),
            "turn_summary": {key: dict(value) for key, value in self._turn_summary.items()},
        }

    def _note_turn_signal(self, key: str) -> None:
        signal = self._turn_summary.get(key)
        if signal is None:
            return
        signal["count"] = int(signal.get("count") or 0) + 1
        signal["seen"] = True

    def _record_asr_trace(
        self,
        name: str,
        content: Mapping[str, object],
        *,
        session_id: str | None,
        round_id: str | None,
        final: bool,
    ) -> None:
        count_key = "final_count" if final else "partial_count"
        self._asr_trace[count_key] = int(self._asr_trace.get(count_key, 0) or 0) + 1
        self._asr_trace["last_event"] = {
            "name": name,
            "session_id": session_id,
            "round_id": round_id,
            "latency_ms": self._first_int(content.get("latencyMs"), content.get("latency_ms")),
            "frame_index": self._first_int(content.get("frameIndex"), content.get("frame_index")),
            "frames_received": self._first_int(content.get("framesReceived"), content.get("frames_received")),
            "frames_processed": self._first_int(content.get("framesProcessed"), content.get("frames_processed")),
            "frames_dropped": self._first_int(content.get("framesDropped"), content.get("frames_dropped")),
            "provider_state": self._first_text(content.get("providerState"), content.get("provider_state")),
        }

    def _is_duplicate_completed_stop(
        self,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> bool:
        if self.session.phase != "completed":
            return False
        if round_id is not None and round_id != str(self.session.round_id):
            return False
        if cancellation_token is not None and cancellation_token != str(self.session.cancellation_token):
            return False
        return True

    @staticmethod
    def _event_payload(event: Mapping[str, object] | EventEnvelope) -> dict[str, object]:
        if isinstance(event, Mapping):
            return dict(event)
        to_dict = getattr(event, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
        raise TypeError("event must be a mapping or EventEnvelope-like object")

    @staticmethod
    def _mapping(value: object) -> dict[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _first_text(*values: object) -> str | None:
        for value in values:
            if value is None:
                continue
            text = str(value)
            if text:
                return text
        return None

    @staticmethod
    def _first_int(*values: object) -> int | None:
        for value in values:
            if value in (None, ""):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _text_content(cls, content: Mapping[str, object], *keys: str, default: str = "") -> str:
        return cls._first_text(*(content.get(key) for key in keys)) or default
