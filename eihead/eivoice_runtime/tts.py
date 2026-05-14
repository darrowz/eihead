from __future__ import annotations

import base64
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Protocol


Clock = Callable[[], float]


@dataclass(frozen=True)
class StreamingTtsRequest:
    text: str
    round_id: str
    cancellation_token: str = ""
    voice_code: str = "gentle_companion_zh_cn"
    emotion: str = "warm"
    speed: float = 1.0
    volume: float = 0.8
    sentence_id: str | None = None
    sample_rate_hz: int = 24000
    channels: int = 1
    audio_format: str = "pcm16"

    @property
    def resolved_sentence_id(self) -> str:
        return self.sentence_id or f"{self.round_id}-sentence-1"


@dataclass(frozen=True)
class StreamingTtsAudioChunk:
    payload: bytes
    index: int
    duration_ms: int = 60
    sample_rate_hz: int = 24000
    channels: int = 1
    audio_format: str = "pcm16"


class StreamingTtsProvider(Protocol):
    def stream(self, request: StreamingTtsRequest) -> Iterable[StreamingTtsAudioChunk]:
        ...

    def status(self) -> dict[str, Any]:
        ...


class SimulatedStreamingTtsProvider:
    """Deterministic offline provider for testing the streaming TTS contract."""

    def __init__(
        self,
        *,
        chunk_chars: int = 6,
        chunk_duration_ms: int = 60,
        sample_rate_hz: int = 24000,
        channels: int = 1,
    ) -> None:
        if chunk_chars <= 0:
            raise ValueError("chunk_chars must be positive")
        if chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        self.chunk_chars = chunk_chars
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self._state: dict[str, Any] = {
            "provider": "simulated",
            "state": "idle",
            "chunks_produced": 0,
        }

    def stream(self, request: StreamingTtsRequest) -> Iterator[StreamingTtsAudioChunk]:
        pieces = _split_text(request.text, self.chunk_chars)
        self._state = {
            "provider": "simulated",
            "state": "streaming" if pieces else "complete",
            "round_id": request.round_id,
            "last_voice_code": request.voice_code,
            "last_emotion": request.emotion,
            "chunk_chars": self.chunk_chars,
            "chunk_duration_ms": self.chunk_duration_ms,
            "chunks_expected": len(pieces),
            "chunks_produced": 0,
        }
        for index, piece in enumerate(pieces):
            payload = (
                f"simtts|{request.round_id}|{request.voice_code}|"
                f"{request.emotion}|{index}|{piece}"
            ).encode("utf-8")
            self._state["chunks_produced"] = index + 1
            yield StreamingTtsAudioChunk(
                payload=payload,
                index=index,
                duration_ms=self.chunk_duration_ms,
                sample_rate_hz=request.sample_rate_hz or self.sample_rate_hz,
                channels=request.channels or self.channels,
                audio_format=request.audio_format,
            )
        self._state["state"] = "complete"

    def status(self) -> dict[str, Any]:
        return dict(self._state)


@dataclass
class _StreamingTtsRoundState:
    request: StreamingTtsRequest
    started_at_s: float
    first_chunk_latency: float | None = None
    chunk_count: int = 0
    cancelled: bool = False
    cancel_reason: str = ""
    superseded_by_round_id: str | None = None
    completed: bool = False
    provider_state: dict[str, Any] = field(default_factory=dict)
    last_error: dict[str, Any] | None = None


class StreamingTtsSession:
    """Owns one interruptible streaming TTS queue contract without cloud I/O."""

    def __init__(
        self,
        *,
        provider: StreamingTtsProvider | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.provider = provider or SimulatedStreamingTtsProvider()
        self._clock = clock or perf_counter
        self._rounds: dict[str, _StreamingTtsRoundState] = {}
        self._active_round_id: str | None = None
        self._last_round_id: str | None = None
        self._cancelled_round_count = 0
        self._last_interrupt: dict[str, Any] | None = None

    def synthesize(
        self,
        *,
        text: str,
        round_id: str,
        cancellation_token: str | None = None,
        voice_code: str = "gentle_companion_zh_cn",
        emotion: str = "warm",
        speed: float = 1.0,
        volume: float = 0.8,
        sentence_id: str | None = None,
        sample_rate_hz: int = 24000,
        channels: int = 1,
        audio_format: str = "pcm16",
    ) -> Iterator[dict[str, Any]]:
        request = StreamingTtsRequest(
            text=str(text or ""),
            round_id=str(round_id),
            cancellation_token=str(cancellation_token or ""),
            voice_code=str(voice_code or "gentle_companion_zh_cn"),
            emotion=str(emotion or "warm"),
            speed=float(speed),
            volume=float(volume),
            sentence_id=sentence_id,
            sample_rate_hz=int(sample_rate_hz),
            channels=int(channels),
            audio_format=str(audio_format or "pcm16"),
        )
        state = self._begin_round(request)
        return self._iter_events(state)

    def cancel(self, *, round_id: str | None = None, reason: str = "cancelled") -> dict[str, Any]:
        target_round_id = round_id or self._active_round_id
        if target_round_id is None:
            return {"cancelled": False, "round_id": None, "reason": reason}
        state = self._rounds.get(str(target_round_id))
        if state is None:
            return {"cancelled": False, "round_id": str(target_round_id), "reason": reason}
        self._mark_cancelled(state, reason=reason)
        return {
            "cancelled": True,
            "round_id": state.request.round_id,
            "roundId": state.request.round_id,
            "cancellation_token": state.request.cancellation_token,
            "cancellationToken": state.request.cancellation_token,
            "reason": state.cancel_reason,
        }

    def interrupt(self, *, round_id: str | None = None, reason: str = "interrupt") -> dict[str, Any]:
        return self.cancel(round_id=round_id, reason=reason)

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()

    def status(self, *, round_id: str | None = None) -> dict[str, Any]:
        target_round_id = round_id or self._active_round_id or self._last_round_id
        if target_round_id is None or target_round_id not in self._rounds:
            return {
                "schema": "eihead.eivoice_runtime.streaming_tts.status.v1",
                "round_id": None,
                "first_chunk_latency": None,
                "chunk_count": 0,
                "voice_code": None,
                "emotion": None,
                "cancelled": False,
                "interrupt_stop_ready": True,
                "cancelled_round_count": self._cancelled_round_count,
                "cancelledRoundCount": self._cancelled_round_count,
                "last_interrupt": dict(self._last_interrupt) if self._last_interrupt is not None else None,
                "lastInterrupt": dict(self._last_interrupt) if self._last_interrupt is not None else None,
                "provider_state": self._provider_status(),
            }
        state = self._rounds[str(target_round_id)]
        return self._status_for_state(state)

    def _begin_round(self, request: StreamingTtsRequest) -> _StreamingTtsRoundState:
        if self._active_round_id and self._active_round_id != request.round_id:
            previous = self._rounds.get(self._active_round_id)
            if previous is not None and not previous.completed:
                previous.superseded_by_round_id = request.round_id
                self._mark_cancelled(previous, reason="superseded")
        state = _StreamingTtsRoundState(
            request=request,
            started_at_s=self._now(),
            provider_state=self._provider_status(),
        )
        self._rounds[request.round_id] = state
        self._active_round_id = request.round_id
        self._last_round_id = request.round_id
        return state

    def _iter_events(self, state: _StreamingTtsRoundState) -> Iterator[dict[str, Any]]:
        request = state.request
        yield self._sentence_start_event(state)
        audio_stream = iter(self.provider.stream(request))
        while True:
            if self._is_cancelled_or_stale(state):
                yield self._complete_event(state, cancelled=True)
                return
            try:
                chunk = next(audio_stream)
            except StopIteration:
                break
            if state.first_chunk_latency is None:
                state.first_chunk_latency = round((self._now() - state.started_at_s) * 1000.0, 2)
            state.chunk_count += 1
            state.provider_state = self._provider_status()
            yield self._audio_chunk_event(state, chunk)
        yield self._complete_event(state, cancelled=state.cancelled)

    def _mark_cancelled(self, state: _StreamingTtsRoundState, *, reason: str) -> None:
        if state.cancelled:
            return
        state.cancelled = True
        state.cancel_reason = str(reason or "cancelled")
        self._cancelled_round_count += 1
        self._last_interrupt = {
            "reason": state.cancel_reason,
            "round_id": state.request.round_id,
            "roundId": state.request.round_id,
            "cancellation_token": state.request.cancellation_token,
            "cancellationToken": state.request.cancellation_token,
            "superseded_by_round_id": state.superseded_by_round_id,
            "supersededByRoundId": state.superseded_by_round_id,
            "ts": self._now(),
        }
        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            try:
                cancel(state.cancel_reason)
            except TypeError:
                try:
                    cancel()
                except BaseException as exc:  # pragma: no cover - defensive provider boundary
                    state.last_error = _error_record(exc, context="cancel")
            except BaseException as exc:  # pragma: no cover - defensive provider boundary
                state.last_error = _error_record(exc, context="cancel")
        state.provider_state = self._provider_status()

    def _is_cancelled_or_stale(self, state: _StreamingTtsRoundState) -> bool:
        if state.cancelled:
            return True
        if self._active_round_id != state.request.round_id:
            self._mark_cancelled(state, reason="superseded")
            return True
        return False

    def _sentence_start_event(self, state: _StreamingTtsRoundState) -> dict[str, Any]:
        request = state.request
        return self._event(
            name="ei.voice.tts.sentence_start",
            event="sentence_start",
            request=request,
            content={
                "eventType": "TTS_SENTENCE_START",
                "sentenceId": request.resolved_sentence_id,
                "sentence_id": request.resolved_sentence_id,
                "text": request.text,
                "voiceCode": request.voice_code,
                "voice_code": request.voice_code,
                "emotion": request.emotion,
                "speed": request.speed,
                "volume": request.volume,
            },
        )

    def _audio_chunk_event(
        self,
        state: _StreamingTtsRoundState,
        chunk: StreamingTtsAudioChunk,
    ) -> dict[str, Any]:
        request = state.request
        audio_base64 = base64.b64encode(chunk.payload).decode("ascii")
        return self._event(
            name="ei.voice.tts.chunk",
            event="audio_chunk",
            request=request,
            content={
                "eventType": "TTS",
                "sentenceId": request.resolved_sentence_id,
                "sentence_id": request.resolved_sentence_id,
                "index": chunk.index,
                "chunkIndex": chunk.index,
                "chunk_index": chunk.index,
                "audioBase64": audio_base64,
                "audio_base64": audio_base64,
                "durationMs": chunk.duration_ms,
                "duration_ms": chunk.duration_ms,
                "sampleRateHz": chunk.sample_rate_hz,
                "sample_rate_hz": chunk.sample_rate_hz,
                "channels": chunk.channels,
                "format": chunk.audio_format,
                "voiceCode": request.voice_code,
                "voice_code": request.voice_code,
                "emotion": request.emotion,
                "speed": request.speed,
                "volume": request.volume,
            },
        )

    def _complete_event(
        self,
        state: _StreamingTtsRoundState,
        *,
        cancelled: bool,
    ) -> dict[str, Any]:
        state.completed = True
        state.cancelled = bool(cancelled)
        state.provider_state = self._provider_status()
        request = state.request
        return self._event(
            name="ei.voice.tts.complete",
            event="complete",
            request=request,
            content={
                "eventType": "TTS_COMPLETE",
                "sentenceId": request.resolved_sentence_id,
                "sentence_id": request.resolved_sentence_id,
                "cancelled": state.cancelled,
                "reason": state.cancel_reason,
                "chunkCount": state.chunk_count,
                "chunk_count": state.chunk_count,
                "firstChunkLatency": state.first_chunk_latency,
                "first_chunk_latency": state.first_chunk_latency,
                "voiceCode": request.voice_code,
                "voice_code": request.voice_code,
                "emotion": request.emotion,
                "speed": request.speed,
                "volume": request.volume,
                "providerState": dict(state.provider_state),
                "provider_state": dict(state.provider_state),
            },
        )

    def _event(
        self,
        *,
        name: str,
        event: str,
        request: StreamingTtsRequest,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        scoped_content = {
            "roundId": request.round_id,
            "round_id": request.round_id,
            "cancellationToken": request.cancellation_token,
            "cancellation_token": request.cancellation_token,
        }
        scoped_content.update(content)
        return {
            "name": name,
            "event": event,
            "event_type": "tts",
            "round_id": request.round_id,
            "roundId": request.round_id,
            "cancellation_token": request.cancellation_token,
            "cancellationToken": request.cancellation_token,
            "content": scoped_content,
        }

    def _status_for_state(self, state: _StreamingTtsRoundState) -> dict[str, Any]:
        provider_state = dict(state.provider_state)
        if state.request.round_id == self._active_round_id:
            provider_state = self._provider_status() or provider_state
        return {
            "schema": "eihead.eivoice_runtime.streaming_tts.status.v1",
            "round_id": state.request.round_id,
            "roundId": state.request.round_id,
            "cancellation_token": state.request.cancellation_token,
            "cancellationToken": state.request.cancellation_token,
            "first_chunk_latency": state.first_chunk_latency,
            "first_chunk_latency_ms": state.first_chunk_latency,
            "chunk_count": state.chunk_count,
            "chunkCount": state.chunk_count,
            "voice_code": state.request.voice_code,
            "voiceCode": state.request.voice_code,
            "emotion": state.request.emotion,
            "speed": state.request.speed,
            "volume": state.request.volume,
            "cancelled": state.cancelled,
            "reason": state.cancel_reason,
            "superseded_by_round_id": state.superseded_by_round_id,
            "supersededByRoundId": state.superseded_by_round_id,
            "completed": state.completed,
            "interrupt_stop_ready": True,
            "cancelled_round_count": self._cancelled_round_count,
            "cancelledRoundCount": self._cancelled_round_count,
            "last_interrupt": dict(self._last_interrupt) if self._last_interrupt is not None else None,
            "lastInterrupt": dict(self._last_interrupt) if self._last_interrupt is not None else None,
            "provider_state": provider_state,
            "last_error": dict(state.last_error) if state.last_error is not None else None,
        }

    def _provider_status(self) -> dict[str, Any]:
        status = self.provider.status()
        return dict(status) if isinstance(status, dict) else {}

    def _now(self) -> float:
        return float(self._clock())


def _split_text(text: str, chunk_chars: int) -> list[str]:
    cleaned = str(text or "")
    if not cleaned:
        return []
    return [cleaned[index : index + chunk_chars] for index in range(0, len(cleaned), chunk_chars)]


def _error_record(error: BaseException | str, *, context: str) -> dict[str, Any]:
    if isinstance(error, BaseException):
        return {"kind": type(error).__name__, "message": str(error), "context": context}
    return {"kind": "Error", "message": str(error), "context": context}


__all__ = [
    "SimulatedStreamingTtsProvider",
    "StreamingTtsAudioChunk",
    "StreamingTtsProvider",
    "StreamingTtsRequest",
    "StreamingTtsSession",
]
