"""Core primitives for voice runtime orchestration.

This module owns conversation-state tracking, ASR/TTS session sequencing and
runtime diagnostics data. It must not own playback policy or session-policy
decisions; those belong to the runtime layer orchestration, while raw playback
execution belongs to `eihead.mouth`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Condition
from time import monotonic, time
from typing import Any, Protocol


@dataclass(frozen=True)
class AudioFrame:
    pcm: bytes = b""
    payload: bytes = b""
    duration_ms: int = 60
    sample_rate_hz: int = 16000
    channels: int = 1
    sequence: int = 0
    created_at_ts: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pcm_length": len(self.pcm),
            "payload_length": len(self.payload),
            "duration_ms": self.duration_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "sequence": self.sequence,
            "created_at_ts": self.created_at_ts,
        }


class VoiceRuntimeStateMachine:
    VALID_STATES = {"idle", "conversation"}

    def __init__(self) -> None:
        self.state = "idle"
        self.history: list[dict[str, Any]] = []

    def wake_detected(self) -> str:
        return self._transition("wake_detected", "conversation" if self.state == "idle" else self.state)

    def conversation_completed(self) -> str:
        return self._transition("conversation_completed", "idle")

    def interrupt_requested(self) -> str:
        return self._transition("interrupt_requested", self.state)

    def sleep_requested(self) -> str:
        return self._transition("sleep_requested", "idle")

    def _transition(self, event: str, next_state: str) -> str:
        if next_state not in self.VALID_STATES:
            raise ValueError(f"invalid voice runtime state: {next_state}")
        previous_state = self.state
        self.state = next_state
        self.history.append(
            {
                "event": event,
                "from": previous_state,
                "to": next_state,
                "ts": time(),
            }
        )
        return self.state


class WakewordRingBuffer:
    def __init__(self, capacity_ms: int = 1500) -> None:
        if capacity_ms <= 0:
            raise ValueError("capacity_ms must be positive")
        self.capacity_ms = capacity_ms
        self._frames: deque[AudioFrame] = deque()
        self._duration_ms = 0

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    @property
    def depth(self) -> int:
        return len(self._frames)

    def append(self, frame: AudioFrame) -> None:
        self._frames.append(frame)
        self._duration_ms += frame.duration_ms
        while self._duration_ms > self.capacity_ms and self._frames:
            removed = self._frames.popleft()
            self._duration_ms -= removed.duration_ms

    def drain(self) -> list[AudioFrame]:
        frames = list(self._frames)
        self._frames.clear()
        self._duration_ms = 0
        return frames


class BoundedAudioQueue:
    def __init__(self, capacity: int, full_policy: str, name: str = "audio_queue") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if full_policy not in {"block", "drop_oldest", "drop_newest"}:
            raise ValueError("full_policy must be block, drop_oldest, or drop_newest")
        self.capacity = capacity
        self.full_policy = full_policy
        self.name = name
        self._items: deque[Any] = deque()
        self._condition = Condition()
        self._pushed = 0
        self._popped = 0
        self._dropped_oldest = 0
        self._dropped_newest = 0

    def push(self, item: Any, *, block: bool | None = None, timeout: float | None = None) -> bool:
        with self._condition:
            if len(self._items) >= self.capacity:
                if self.full_policy == "drop_oldest":
                    self._items.popleft()
                    self._dropped_oldest += 1
                elif self.full_policy == "drop_newest":
                    self._dropped_newest += 1
                    return False
                else:
                    should_block = True if block is None else block
                    if not should_block or not self._wait_for_space(timeout):
                        return False

            self._items.append(item)
            self._pushed += 1
            self._condition.notify()
            return True

    def pop(self, *, block: bool = False, timeout: float | None = None) -> Any | None:
        with self._condition:
            if not self._items and (not block or not self._wait_for_item(timeout)):
                return None
            item = self._items.popleft()
            self._popped += 1
            self._condition.notify()
            return item

    def stats(self) -> dict[str, int | str]:
        with self._condition:
            return {
                "name": self.name,
                "capacity": self.capacity,
                "full_policy": self.full_policy,
                "pushed": self._pushed,
                "popped": self._popped,
                "dropped_oldest": self._dropped_oldest,
                "dropped_newest": self._dropped_newest,
                "depth": len(self._items),
            }

    def _wait_for_space(self, timeout: float | None) -> bool:
        if timeout is None:
            while len(self._items) >= self.capacity:
                self._condition.wait()
            return True

        deadline = monotonic() + timeout
        while len(self._items) >= self.capacity:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            self._condition.wait(remaining)
        return True

    def _wait_for_item(self, timeout: float | None) -> bool:
        if timeout is None:
            while not self._items:
                self._condition.wait()
            return True

        deadline = monotonic() + timeout
        while not self._items:
            remaining = deadline - monotonic()
            if remaining <= 0:
                return False
            self._condition.wait(remaining)
        return True


class OpusCodec(Protocol):
    def encode(self, frame: AudioFrame) -> AudioFrame:
        ...

    def decode(self, frame: AudioFrame) -> AudioFrame:
        ...


class PassthroughOpusCodec:
    def encode(self, frame: AudioFrame) -> AudioFrame:
        source = frame.pcm or frame.payload
        return AudioFrame(
            payload=source,
            duration_ms=frame.duration_ms,
            sample_rate_hz=frame.sample_rate_hz,
            channels=frame.channels,
            sequence=frame.sequence,
            created_at_ts=frame.created_at_ts,
        )

    def decode(self, frame: AudioFrame) -> AudioFrame:
        source = frame.payload or frame.pcm
        return AudioFrame(
            pcm=source,
            duration_ms=frame.duration_ms,
            sample_rate_hz=frame.sample_rate_hz,
            channels=frame.channels,
            sequence=frame.sequence,
            created_at_ts=frame.created_at_ts,
        )


class EiVoiceRuntimeCore:
    def __init__(self, codec: OpusCodec | None = None) -> None:
        self.state_machine = VoiceRuntimeStateMachine()
        self.wakeword_buffer = WakewordRingBuffer()
        self.codec = codec or PassthroughOpusCodec()
        self._interrupt_stop_ready = False
        self._last_interrupt: dict[str, Any] | None = None
        self._cancelled_round_count = 0
        self.opus_encode_queue = BoundedAudioQueue(
            capacity=3,
            full_policy="block",
            name="opus_encode_queue",
        )
        self.ws_send_queue = BoundedAudioQueue(
            capacity=25,
            full_policy="drop_oldest",
            name="ws_send_queue",
        )
        self.opus_decode_queue = BoundedAudioQueue(
            capacity=25,
            full_policy="drop_newest",
            name="opus_decode_queue",
        )
        self.audio_playback_queue = BoundedAudioQueue(
            capacity=3,
            full_policy="block",
            name="audio_playback_queue",
        )

    @property
    def state(self) -> str:
        return self.state_machine.state

    def set_interrupt_stop_ready(self, ready: bool) -> None:
        self._interrupt_stop_ready = bool(ready)

    def record_interrupt(
        self,
        *,
        reason: str,
        cleared: int,
        source: str = "runtime",
        round_id: str | None = None,
        playback_frames_cleared: int = 0,
        decode_frames_cleared: int = 0,
        transport_inbound_events_cleared: int = 0,
    ) -> dict[str, Any]:
        self._cancelled_round_count += 1
        self._last_interrupt = {
            "reason": str(reason or "interrupt"),
            "source": str(source or "runtime"),
            "round_id": round_id,
            "roundId": round_id,
            "cleared": int(cleared),
            "playback_frames_cleared": int(playback_frames_cleared),
            "playbackFramesCleared": int(playback_frames_cleared),
            "decode_frames_cleared": int(decode_frames_cleared),
            "decodeFramesCleared": int(decode_frames_cleared),
            "transport_inbound_events_cleared": int(transport_inbound_events_cleared),
            "transportInboundEventsCleared": int(transport_inbound_events_cleared),
            "ts": time(),
        }
        return dict(self._last_interrupt)

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state_machine.state,
            "transition_count": len(self.state_machine.history),
            "interruptStopReady": self._interrupt_stop_ready,
            "interrupt_stop_ready": self._interrupt_stop_ready,
            "lastInterrupt": dict(self._last_interrupt) if self._last_interrupt is not None else None,
            "last_interrupt": dict(self._last_interrupt) if self._last_interrupt is not None else None,
            "cancelledRoundCount": self._cancelled_round_count,
            "cancelled_round_count": self._cancelled_round_count,
            "wakeword_buffer": {
                "capacity_ms": self.wakeword_buffer.capacity_ms,
                "duration_ms": self.wakeword_buffer.duration_ms,
                "depth": self.wakeword_buffer.depth,
            },
            "queues": {
                "opus_encode_queue": self.opus_encode_queue.stats(),
                "ws_send_queue": self.ws_send_queue.stats(),
                "opus_decode_queue": self.opus_decode_queue.stats(),
                "audio_playback_queue": self.audio_playback_queue.stats(),
            },
        }
