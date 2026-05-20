"""Voice stream transport adapters for runtime transport plumbing.

This module owns transport state (queues, reconnect timing, heartbeat and errors),
while session-level orchestration and playback policy remain the runtime boundary.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from threading import RLock
from time import monotonic
from typing import Any, Protocol

from eihead.eivoice_runtime.joyinside_voice import JoyInsideVoiceEvent, ping

from .core import BoundedAudioQueue


Clock = Callable[[], float]


class VoiceStreamTransport(Protocol):
    def mark_connected(self) -> None:
        ...

    def begin_reconnect(self) -> None:
        ...

    def schedule_reconnect(self, reason: str) -> float:
        ...

    def ready_to_reconnect(self) -> bool:
        ...

    def send_event(self, event: Mapping[str, Any] | JoyInsideVoiceEvent) -> bool:
        ...

    def receive_event(self, *, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        ...

    def drain_inbound_events(self) -> list[dict[str, Any]]:
        ...

    def push_inbound_event(self, event: Mapping[str, Any] | JoyInsideVoiceEvent) -> bool:
        ...

    def pop_outbound_event(self, *, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        ...

    def send_ping(self, *, uid: str | None = None, mid: str | None = None) -> dict[str, Any]:
        ...

    def record_pong(self, event: Mapping[str, Any] | JoyInsideVoiceEvent | None = None) -> None:
        ...

    def check_heartbeat(self) -> str | None:
        ...

    def record_error(self, error: BaseException | str, *, context: str) -> dict[str, Any]:
        ...

    def status(self) -> dict[str, Any]:
        ...


class InMemoryVoiceStreamTransport:
    def __init__(
        self,
        *,
        clock: Clock | None = None,
        outbound_capacity: int = 25,
        inbound_capacity: int = 25,
        heartbeat_interval_s: float = 10.0,
        pong_timeout_s: float = 5.0,
        reconnect_base_delay_s: float = 1.0,
        reconnect_max_delay_s: float = 30.0,
        recent_error_limit: int = 10,
        transport_name: str = "in_memory",
    ) -> None:
        self._clock = clock or monotonic
        self._lock = RLock()
        self._transport_name = transport_name
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._pong_timeout_s = float(pong_timeout_s)
        self._reconnect_base_delay_s = float(reconnect_base_delay_s)
        self._reconnect_max_delay_s = float(reconnect_max_delay_s)
        self._outbound_queue = BoundedAudioQueue(
            capacity=outbound_capacity,
            full_policy="drop_oldest",
            name="outbound_queue",
        )
        self._inbound_queue = BoundedAudioQueue(
            capacity=inbound_capacity,
            full_policy="drop_newest",
            name="inbound_queue",
        )
        self._connection_state = "idle"
        now = self._clock()
        self._last_transition_at = now
        self._last_activity_at = now
        self._last_ping_at: float | None = None
        self._last_pong_at: float | None = None
        self._awaiting_pong = False
        self._latency_ms: int | None = None
        self._reconnect_attempt = 0
        self._reconnect_backoff_s: float | None = None
        self._next_retry_at: float | None = None
        self._reconnect_reason: str | None = None
        self._last_error: dict[str, Any] | None = None
        self._recent_errors: deque[dict[str, Any]] = deque(maxlen=recent_error_limit)

    def mark_connected(self) -> None:
        with self._lock:
            now = self._clock()
            self._connection_state = "connected"
            self._last_transition_at = now
            self._last_activity_at = now
            self._awaiting_pong = False
            self._reconnect_attempt = 0
            self._reconnect_backoff_s = None
            self._next_retry_at = None
            self._reconnect_reason = None
            self._last_error = None

    def begin_reconnect(self) -> None:
        with self._lock:
            now = self._clock()
            if not self._ready_to_reconnect_locked(now):
                raise RuntimeError("reconnect is not ready")
            self._connection_state = "connecting"
            self._last_transition_at = now
            self._last_activity_at = now

    def schedule_reconnect(self, reason: str) -> float:
        with self._lock:
            now = self._clock()
            self._reconnect_attempt += 1
            self._reconnect_backoff_s = min(
                self._reconnect_base_delay_s * (2 ** (self._reconnect_attempt - 1)),
                self._reconnect_max_delay_s,
            )
            self._next_retry_at = now + self._reconnect_backoff_s
            self._reconnect_reason = str(reason)
            self._connection_state = "reconnect_wait"
            self._last_transition_at = now
            self._last_activity_at = now
            self._awaiting_pong = False
            return self._reconnect_backoff_s

    def ready_to_reconnect(self) -> bool:
        with self._lock:
            return self._ready_to_reconnect_locked(self._clock())

    def send_event(self, event: Mapping[str, Any] | JoyInsideVoiceEvent) -> bool:
        payload = _event_dict(event)
        accepted = self._outbound_queue.push(payload, block=False)
        with self._lock:
            if accepted:
                self._last_activity_at = self._clock()
        return accepted

    def receive_event(self, *, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        payload = self._inbound_queue.pop(block=block, timeout=timeout)
        if payload is not None:
            with self._lock:
                self._last_activity_at = self._clock()
        return payload

    def drain_inbound_events(self) -> list[dict[str, Any]]:
        drained: list[dict[str, Any]] = []
        while True:
            payload = self.receive_event()
            if payload is None:
                return drained
            drained.append(payload)

    def push_inbound_event(self, event: Mapping[str, Any] | JoyInsideVoiceEvent) -> bool:
        payload = _event_dict(event)
        accepted = self._inbound_queue.push(payload, block=False)
        with self._lock:
            if accepted:
                self._last_activity_at = self._clock()
        return accepted

    def pop_outbound_event(self, *, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        payload = self._outbound_queue.pop(block=block, timeout=timeout)
        if payload is not None:
            with self._lock:
                self._last_activity_at = self._clock()
        return payload

    def send_ping(self, *, uid: str | None = None, mid: str | None = None) -> dict[str, Any]:
        payload = ping(uid=uid, mid=mid, timestamp=self._clock()).to_dict()
        self.send_event(payload)
        with self._lock:
            now = self._clock()
            self._last_ping_at = now
            self._last_activity_at = now
            self._awaiting_pong = True
        return payload

    def record_pong(self, event: Mapping[str, Any] | JoyInsideVoiceEvent | None = None) -> None:
        _ = event
        with self._lock:
            now = self._clock()
            self._last_pong_at = now
            self._last_activity_at = now
            self._awaiting_pong = False
            if self._last_ping_at is not None:
                self._latency_ms = max(0, int(round((now - self._last_ping_at) * 1000)))

    def check_heartbeat(self) -> str | None:
        with self._lock:
            now = self._clock()
            if self._connection_state != "connected":
                return None
            if not self._heartbeat_timed_out_locked(now):
                return None
        self.record_error(TimeoutError("pong not received before timeout"), context="heartbeat")
        self.schedule_reconnect("heartbeat_timeout")
        return "heartbeat_timeout"

    def heartbeat_due(self) -> bool:
        with self._lock:
            now = self._clock()
            if self._connection_state != "connected" or self._awaiting_pong:
                return False
            if self._last_ping_at is None:
                return (now - self._last_activity_at) >= self._heartbeat_interval_s
            return (now - self._last_ping_at) >= self._heartbeat_interval_s

    def record_error(self, error: BaseException | str, *, context: str) -> dict[str, Any]:
        with self._lock:
            entry = _error_record(error, context=context, now=self._clock())
            self._last_error = entry
            self._recent_errors.append(entry)
            return deepcopy(entry)

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            outbound_stats = self._queue_stats_with_alias(self._outbound_queue.stats(), "ws_send_queue")
            inbound_stats = self._queue_stats_with_alias(self._inbound_queue.stats(), "opus_decode_queue")
            return {
                "transport": self._transport_name,
                "state": self._connection_state,
                "conversation_state": self._connection_state,
                "connection": {
                    "state": self._connection_state,
                    "last_transition_at": self._last_transition_at,
                    "last_activity_at": self._last_activity_at,
                },
                "heartbeat": {
                    "interval_s": self._heartbeat_interval_s,
                    "pong_timeout_s": self._pong_timeout_s,
                    "awaiting_pong": self._awaiting_pong,
                    "last_ping_at": self._last_ping_at,
                    "last_pong_at": self._last_pong_at,
                    "latency_ms": self._latency_ms,
                    "due": self.heartbeat_due(),
                    "timed_out": self._heartbeat_timed_out_locked(now),
                },
                "reconnect": {
                    "attempt": self._reconnect_attempt,
                    "backoff_s": self._reconnect_backoff_s,
                    "next_retry_at": self._next_retry_at,
                    "ready": self._ready_to_reconnect_locked(now),
                    "reason": self._reconnect_reason,
                },
                "queues": {
                    "outbound_queue": deepcopy(outbound_stats),
                    "inbound_queue": deepcopy(inbound_stats),
                    "ws_send_queue": deepcopy(outbound_stats),
                    "opus_decode_queue": deepcopy(inbound_stats),
                },
                "last_error": deepcopy(self._last_error),
                "recent_errors": [deepcopy(item) for item in self._recent_errors],
            }

    def _queue_stats_with_alias(self, stats: dict[str, Any], alias: str) -> dict[str, Any]:
        result = dict(stats)
        result["name"] = alias
        return result

    def _ready_to_reconnect_locked(self, now: float) -> bool:
        return self._next_retry_at is not None and now >= self._next_retry_at

    def _heartbeat_timed_out_locked(self, now: float) -> bool:
        return self._awaiting_pong and self._last_ping_at is not None and (now - self._last_ping_at) > self._pong_timeout_s


class FakeWebSocketTransport(InMemoryVoiceStreamTransport):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(transport_name="fake_websocket", **kwargs)

    def open(self) -> None:
        self.mark_connected()

    def close(self, reason: str | None = None) -> None:
        with self._lock:
            now = self._clock()
            self._connection_state = "closed"
            self._reconnect_reason = reason
            self._last_transition_at = now
            self._last_activity_at = now
            self._awaiting_pong = False

    def deliver_from_server(self, event: Mapping[str, Any] | JoyInsideVoiceEvent) -> bool:
        return self.push_inbound_event(event)

    def recv_from_client(self, *, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        return self.pop_outbound_event(block=block, timeout=timeout)


def _event_dict(event: Mapping[str, Any] | JoyInsideVoiceEvent) -> dict[str, Any]:
    if isinstance(event, JoyInsideVoiceEvent):
        return event.to_dict()
    if isinstance(event, Mapping):
        return deepcopy(dict(event))
    raise TypeError("voice transport events must be mappings or JoyInsideVoiceEvent instances")


def _error_record(error: BaseException | str, *, context: str, now: float) -> dict[str, Any]:
    if isinstance(error, BaseException):
        kind = type(error).__name__
        message = str(error)
    else:
        kind = "Error"
        message = str(error)
    return {
        "kind": kind,
        "message": message,
        "context": context,
        "ts": now,
    }
