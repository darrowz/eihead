"""ASR provider/session stream wrappers for runtime orchestration.

The ASR adapter layer normalizes provider results and transport artifacts. Runtime
session lifecycle decisions and policy orchestration stay in `eivoice_runtime`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol

from .core import AudioFrame


Clock = Callable[[], float]


@dataclass(frozen=True)
class AsrProviderResult:
    text: str
    final: bool = False
    provider_state: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StreamingAsrProvider(Protocol):
    provider_name: str

    @property
    def state(self) -> str:
        ...

    def can_accept_frame(self) -> bool:
        ...

    def accept_frame(self, frame: AudioFrame) -> Iterable[AsrProviderResult]:
        ...

    def diagnostics(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class StreamingAsrEvent:
    name: str
    text: str
    final: bool
    session_id: str
    round_id: str
    latency_ms: int
    frame_index: int
    frames_received: int
    frames_processed: int
    frames_dropped: int
    audio_ms_received: int
    provider: str
    provider_state: str
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        content: dict[str, Any] = {
            "text": self.text,
            "final": self.final,
            "latencyMs": self.latency_ms,
            "latency_ms": self.latency_ms,
            "frameIndex": self.frame_index,
            "frame_index": self.frame_index,
            "framesReceived": self.frames_received,
            "frames_received": self.frames_received,
            "framesProcessed": self.frames_processed,
            "frames_processed": self.frames_processed,
            "framesDropped": self.frames_dropped,
            "frames_dropped": self.frames_dropped,
            "audioMs": self.audio_ms_received,
            "audio_ms": self.audio_ms_received,
            "provider": self.provider,
            "providerState": self.provider_state,
            "provider_state": self.provider_state,
        }
        if self.confidence is not None:
            content["confidence"] = self.confidence
        if self.metadata:
            content["metadata"] = dict(self.metadata)
        return {
            "name": self.name,
            "type": "voice",
            "sessionId": self.session_id,
            "session_id": self.session_id,
            "roundId": self.round_id,
            "round_id": self.round_id,
            "content": content,
        }


class SimulatedStreamingAsrProvider:
    provider_name = "simulated"

    def __init__(
        self,
        *,
        partial_text: str = "simulated partial",
        final_text: str = "simulated final",
        final_after_frames: int = 3,
        partial_every_frames: int = 1,
        blocked: bool = False,
    ) -> None:
        if final_after_frames <= 0:
            raise ValueError("final_after_frames must be positive")
        if partial_every_frames <= 0:
            raise ValueError("partial_every_frames must be positive")
        self.partial_text = partial_text
        self.final_text = final_text
        self.final_after_frames = final_after_frames
        self.partial_every_frames = partial_every_frames
        self._blocked = blocked
        self._frames_seen = 0
        self._final_emitted = False
        self._state = "backpressure" if blocked else "idle"
        self._errors: list[dict[str, Any]] = []

    @property
    def state(self) -> str:
        return self._state

    def can_accept_frame(self) -> bool:
        return not self._blocked

    def set_blocked(self, blocked: bool) -> None:
        self._blocked = blocked
        if blocked:
            self._state = "backpressure"
        elif self._final_emitted:
            self._state = "finalized"
        elif self._frames_seen:
            self._state = "streaming"
        else:
            self._state = "idle"

    def accept_frame(self, frame: AudioFrame) -> Iterable[AsrProviderResult]:
        _ = frame
        if self._blocked:
            self._state = "backpressure"
            return []
        self._frames_seen += 1
        self._state = "streaming"
        results: list[AsrProviderResult] = []
        if self.partial_text and self._frames_seen % self.partial_every_frames == 0:
            results.append(
                AsrProviderResult(
                    text=self.partial_text,
                    final=False,
                    provider_state=self._state,
                )
            )
        if not self._final_emitted and self._frames_seen >= self.final_after_frames:
            self._final_emitted = True
            self._state = "finalized"
            results.append(
                AsrProviderResult(
                    text=self.final_text,
                    final=True,
                    provider_state=self._state,
                )
            )
        return results

    def inject_error(self, message: str) -> None:
        self._errors.append({"kind": "ProviderWarning", "message": str(message), "context": "provider"})

    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "provider": self.provider_name,
            "state": self._state,
            "frames_seen": self._frames_seen,
            "final_emitted": self._final_emitted,
            "errors": [dict(error) for error in self._errors],
        }


class StreamingAsrSession:
    def __init__(
        self,
        *,
        session_id: str,
        round_id: str,
        provider: StreamingAsrProvider,
        max_inflight_frames: int = 4,
        clock: Clock | None = None,
    ) -> None:
        if max_inflight_frames <= 0:
            raise ValueError("max_inflight_frames must be positive")
        self.session_id = session_id
        self.round_id = round_id
        self.provider = provider
        self.max_inflight_frames = max_inflight_frames
        self._clock = clock or monotonic
        self._started_at_s = self._clock()
        self._pending: deque[AudioFrame] = deque()
        self._events: list[StreamingAsrEvent] = []
        self._errors: list[dict[str, Any]] = []
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._dropped_oldest = 0
        self._audio_ms_received = 0
        self._partial_count = 0
        self._final_count = 0
        self._first_partial_ms: int | None = None
        self._final_ms: int | None = None
        self._latest_voice: dict[str, Any] = {}
        self._cancelled = False
        self._cancel_count = 0
        self._last_cancel: dict[str, Any] | None = None

    def accept_frame(self, frame: AudioFrame) -> list[StreamingAsrEvent]:
        self._frames_received += 1
        self._audio_ms_received += frame.duration_ms
        self._latest_voice = _frame_diagnostics(frame, audio_ms_received=self._audio_ms_received)
        self._enqueue_frame(frame)
        return self.flush()

    def flush(self) -> list[StreamingAsrEvent]:
        emitted: list[StreamingAsrEvent] = []
        while self._pending and self._provider_ready():
            frame = self._pending.popleft()
            self._frames_processed += 1
            try:
                provider_results = list(self.provider.accept_frame(frame))
            except BaseException as exc:  # pragma: no cover - defensive provider boundary
                self._errors.append(_error_record(exc, context="accept_frame"))
                break
            for result in provider_results:
                event = self._event_from_provider_result(result, frame)
                emitted.append(event)
                self._events.append(event)
        return emitted

    def drain_events(self) -> list[StreamingAsrEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def cancel(self, reason: str = "cancelled") -> dict[str, Any]:
        self._cancelled = True
        self._cancel_count += 1
        self._last_cancel = {
            "reason": str(reason or "cancelled"),
            "round_id": self.round_id,
            "roundId": self.round_id,
            "frames_pending": len(self._pending),
        }
        cancel = getattr(self.provider, "cancel", None)
        if callable(cancel):
            try:
                result = cancel(reason)
            except TypeError:
                result = cancel()
            except BaseException as exc:  # pragma: no cover - defensive provider boundary
                self._errors.append(_error_record(exc, context="cancel"))
                self._pending.clear()
                return {"cancelled": False, "reason": reason, "error": str(exc)}
            self._pending.clear()
            response = dict(result) if isinstance(result, Mapping) else {"cancelled": True, "reason": reason}
            response.setdefault("round_id", self.round_id)
            response.setdefault("roundId", self.round_id)
            return response
        self._pending.clear()
        return {"cancelled": True, "reason": reason, "round_id": self.round_id, "roundId": self.round_id}

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as exc:  # pragma: no cover - defensive provider boundary
                self._errors.append(_error_record(exc, context="close"))

    def status(self) -> dict[str, Any]:
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        provider_diagnostics = _provider_diagnostics(self.provider)
        provider_errors = provider_diagnostics.get("errors")
        errors = [dict(error) for error in self._errors]
        if isinstance(provider_errors, list):
            errors.extend(dict(error) for error in provider_errors if isinstance(error, Mapping))
        return {
            "schema": "eihead.eivoice_runtime.asr.v1",
            "enabled": True,
            "session_id": self.session_id,
            "round_id": self.round_id,
            "provider": _provider_name(self.provider, provider_diagnostics),
            "provider_state": self._provider_state(provider_diagnostics),
            "partial_count": self._partial_count,
            "final_count": self._final_count,
            "first_partial_ms": self._first_partial_ms,
            "final_ms": self._final_ms,
            "frames_received": self._frames_received,
            "frames_processed": self._frames_processed,
            "frames_pending": len(self._pending),
            "frames_dropped": self._frames_dropped,
            "dropped_oldest": self._dropped_oldest,
            "max_inflight_frames": self.max_inflight_frames,
            "audio_ms_received": self._audio_ms_received,
            "latest_voice": dict(self._latest_voice),
            "interrupt_stop_ready": True,
            "cancelled": self._cancelled,
            "cancel_count": self._cancel_count,
            "last_cancel": dict(self._last_cancel) if self._last_cancel is not None else None,
            "errors": errors,
            "provider_diagnostics": dict(provider_diagnostics),
        }

    def _enqueue_frame(self, frame: AudioFrame) -> None:
        if len(self._pending) >= self.max_inflight_frames:
            self._pending.popleft()
            self._frames_dropped += 1
            self._dropped_oldest += 1
        self._pending.append(frame)

    def _event_from_provider_result(self, result: AsrProviderResult, frame: AudioFrame) -> StreamingAsrEvent:
        latency_ms = self._latency_ms()
        if result.final:
            self._final_count += 1
            self._final_ms = latency_ms
            name = "ei.voice.asr.final"
        else:
            self._partial_count += 1
            if self._first_partial_ms is None:
                self._first_partial_ms = latency_ms
            name = "ei.voice.asr.partial"
        return StreamingAsrEvent(
            name=name,
            text=result.text,
            final=result.final,
            session_id=self.session_id,
            round_id=self.round_id,
            latency_ms=latency_ms,
            frame_index=frame.sequence,
            frames_received=self._frames_received,
            frames_processed=self._frames_processed,
            frames_dropped=self._frames_dropped,
            audio_ms_received=self._audio_ms_received,
            provider=_provider_name(self.provider, {}),
            provider_state=result.provider_state or self._provider_state({}),
            confidence=result.confidence,
            metadata=result.metadata,
        )

    def _provider_ready(self) -> bool:
        can_accept = getattr(self.provider, "can_accept_frame", None)
        if not callable(can_accept):
            return True
        try:
            return bool(can_accept())
        except BaseException as exc:  # pragma: no cover - defensive provider boundary
            self._errors.append(_error_record(exc, context="can_accept_frame"))
            return False

    def _provider_state(self, provider_diagnostics: Mapping[str, Any]) -> str:
        if self._pending and not self._provider_ready():
            return "backpressure"
        state = provider_diagnostics.get("state") if isinstance(provider_diagnostics, Mapping) else None
        return str(state or getattr(self.provider, "state", "unknown") or "unknown")

    def _latency_ms(self) -> int:
        return max(0, int(round((self._clock() - self._started_at_s) * 1000)))


def _frame_diagnostics(frame: AudioFrame, *, audio_ms_received: int) -> dict[str, Any]:
    return {
        "sequence": frame.sequence,
        "duration_ms": frame.duration_ms,
        "sample_rate_hz": frame.sample_rate_hz,
        "channels": frame.channels,
        "pcm_length": len(frame.pcm),
        "payload_length": len(frame.payload),
        "audio_ms_received": audio_ms_received,
    }


def _provider_diagnostics(provider: StreamingAsrProvider) -> dict[str, Any]:
    diagnostics = getattr(provider, "diagnostics", None)
    if not callable(diagnostics):
        return {}
    try:
        value = diagnostics()
    except BaseException as exc:  # pragma: no cover - defensive provider boundary
        return {"state": "error", "errors": [_error_record(exc, context="diagnostics")]}
    return dict(value) if isinstance(value, Mapping) else {}


def _provider_name(provider: StreamingAsrProvider, diagnostics: Mapping[str, Any]) -> str:
    name = diagnostics.get("provider") if isinstance(diagnostics, Mapping) else None
    return str(name or getattr(provider, "provider_name", type(provider).__name__))


def _error_record(error: BaseException | str, *, context: str) -> dict[str, Any]:
    if isinstance(error, BaseException):
        return {"kind": type(error).__name__, "message": _redact_common_secret_text(str(error)), "context": context}
    return {"kind": "Error", "message": _redact_common_secret_text(str(error)), "context": context}


def _redact_common_secret_text(text: str) -> str:
    parts = str(text).split()
    redacted: list[str] = []
    redact_next = False
    for part in parts:
        if redact_next:
            redacted.append(_redact_secret(part))
            redact_next = False
            continue
        redacted.append(part)
        if part.lower().rstrip(":") == "bearer":
            redact_next = True
    return " ".join(redacted)


def _redact_secret(value: str) -> str:
    text = str(value)
    if len(text) <= 2:
        return "*" * len(text)
    return f"{text[0]}***{text[-1]}"
