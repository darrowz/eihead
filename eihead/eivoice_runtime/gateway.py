"""Fake-device-friendly eihead voice gateway.

This module keeps microphone and speaker access behind injected objects so the
streaming voice state machine can be tested before honjia hardware is wired in.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from eiprotocol import builders
from eiprotocol.builders import EventIdFactory, build_event
from eiprotocol.models import EventEnvelope, SourceRef, TargetRef


def _source_ref(value: SourceRef | Mapping[str, Any]) -> SourceRef:
    if isinstance(value, SourceRef):
        return value
    return SourceRef.from_dict(value)


def _target_ref(value: TargetRef | Mapping[str, Any] | None) -> TargetRef | None:
    if value is None or isinstance(value, TargetRef):
        return value
    return TargetRef.from_dict(value)


def _camel_audio_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    audio_base64 = frame.get("audioBase64", frame.get("audio_base64", ""))
    payload: dict[str, Any] = {
        "audioBase64": str(audio_base64 or ""),
    }
    optional_fields = {
        "sampleRateHz": ("sampleRateHz", "sample_rate_hz"),
        "channels": ("channels",),
        "format": ("format",),
        "durationMs": ("durationMs", "duration_ms"),
        "rmsDbfs": ("rmsDbfs", "rms_dbfs"),
    }
    for output_key, input_keys in optional_fields.items():
        for input_key in input_keys:
            if input_key in frame and frame[input_key] is not None:
                payload[output_key] = frame[input_key]
                break
    return payload


class EiVoiceGateway:
    """Small voice I/O state machine with injectable fake devices."""

    def __init__(
        self,
        *,
        session_id: str,
        actor_id: str,
        capture: Any,
        playback: Any,
        stream_id: str = "mic",
        tts_stream_id: str = "tts",
        round_id: str = "",
        trace_id: str = "",
        head_instance_id: str = "head-runtime",
        brain_instance_id: str = "brain-runtime",
        ids: EventIdFactory | None = None,
    ) -> None:
        self.session_id = session_id
        self.actor_id = actor_id
        self.capture = capture
        self.playback = playback
        self.stream_id = stream_id
        self.tts_stream_id = tts_stream_id
        self.round_id = round_id
        self.trace_id = trace_id
        self.state = "idle"
        self.transcript_partial = ""
        self.transcript_final = ""
        self.connected = True
        self.reconnect_reason = ""
        self._ids = ids or EventIdFactory()
        self._sequence = 0
        self._capture_chunk_index = 0
        self._tts_chunk_index = 0
        self._captured_frames = 0
        self._tts_queue: list[dict[str, Any]] = []
        self._head_source = {
            "domain": "eihead",
            "instanceId": head_instance_id,
            "uid": actor_id,
        }
        self._brain_source = {
            "domain": "eibrain",
            "instanceId": brain_instance_id,
            "uid": actor_id,
        }
        self._head_target = {"domain": "eihead", "instanceId": head_instance_id}
        self._brain_target = {"domain": "eibrain", "instanceId": brain_instance_id}

    def capture_audio_frame(self) -> list[EventEnvelope]:
        frame = self._read_capture_frame()
        if frame is None:
            return []

        content = {
            "streamId": self.stream_id,
            "chunkIndex": self._capture_chunk_index,
            **_camel_audio_frame(frame),
        }
        self._capture_chunk_index += 1
        self._captured_frames += 1
        self.state = "capturing"
        return [
            self._build_voice_event(
                name="ei.voice.audio.frame",
                event_type="observation",
                source=self._head_source,
                target=self._brain_target,
                content=content,
                builder_name="build_voice_audio_frame_event",
                builder_kwargs={
                    "stream_id": content["streamId"],
                    "chunk_index": content["chunkIndex"],
                    "audio_base64": content["audioBase64"],
                },
                round_scoped=False,
            )
        ]

    def accept_asr_partial(self, text: str) -> EventEnvelope:
        self.transcript_partial = text
        self.state = "listening"
        return self._build_asr_event(text=text, final=False)

    def accept_asr_final(self, text: str) -> EventEnvelope:
        self.transcript_final = text
        self.transcript_partial = ""
        self.state = "heard"
        return self._build_asr_event(text=text, final=True)

    def enqueue_tts_chunk(self, audio_base64: str, *, sentence_id: str = "") -> EventEnvelope:
        content = {
            "streamId": self.tts_stream_id,
            "chunkIndex": self._tts_chunk_index,
            "audioBase64": audio_base64,
        }
        if sentence_id:
            content["sentenceId"] = sentence_id

        playback_chunk = {
            "streamId": content["streamId"],
            "chunkIndex": content["chunkIndex"],
            "audioBase64": audio_base64,
            "sentenceId": sentence_id,
        }
        self._tts_chunk_index += 1
        self._tts_queue.append(dict(playback_chunk))
        enqueue = getattr(self.playback, "enqueue_chunk", None)
        if callable(enqueue):
            enqueue(playback_chunk)

        return self._build_voice_event(
            name="ei.voice.tts.chunk",
            event_type="dialogue",
            source=self._brain_source,
            target=self._head_target,
            content=content,
            builder_name="build_voice_tts_chunk_event",
            builder_kwargs={
                "stream_id": content["streamId"],
                "chunk_index": content["chunkIndex"],
                "audio_base64": audio_base64,
            },
            round_scoped=True,
        )

    def start_playback(self) -> EventEnvelope:
        start = getattr(self.playback, "start", None)
        if callable(start):
            start()
        self.state = "playing"
        return self._playback_state_event(started=True, state="playing")

    def stop_playback(self, reason: str = "completed") -> EventEnvelope:
        stop = getattr(self.playback, "stop", None)
        if callable(stop):
            stop(reason=reason)
        self._tts_queue.clear()
        self.state = "stopped"
        return self._playback_state_event(started=False, state="stopped", reason=reason)

    def probe_barge_in(self) -> EventEnvelope | None:
        if self.state != "playing":
            return None

        result = self._probe_capture_barge_in()
        detected = bool(result.get("detected", result.get("active", result.get("voice", False))))
        if not detected:
            return None

        reason = str(result.get("reason") or "near_field_speech")
        self.state = "barge_in"
        content = {"reason": reason, "state": self.state}
        for key in ("rms_dbfs", "rmsDbfs", "latency_ms", "latencyMs"):
            if key in result:
                output_key = {"rms_dbfs": "rmsDbfs", "latency_ms": "latencyMs"}.get(key, key)
                content[output_key] = result[key]
        return self._build_voice_event(
            name="ei.voice.barge_in.detected",
            event_type="dialogue",
            source=self._head_source,
            target=self._brain_target,
            content=content,
            builder_name="build_voice_barge_in_detected_event",
            builder_kwargs={"reason": reason},
            round_scoped=True,
        )

    def heartbeat(self) -> EventEnvelope:
        content = {
            "state": self.state,
            "health": {
                "capture": self._device_health(self.capture),
                "playback": self._device_health(self.playback),
            },
            "queueLengths": {
                "capture": self._capture_queue_length(),
                "tts": len(self._tts_queue),
            },
            "reconnect": {
                "connected": self.connected,
                "reason": self.reconnect_reason,
            },
        }
        return self._build_voice_event(
            name="ei.voice.session.heartbeat",
            event_type="control",
            source=self._head_source,
            target=self._brain_target,
            content=content,
            builder_name="build_voice_session_heartbeat_event",
            builder_kwargs={
                "state": self.state,
                "health": content["health"],
            },
            round_scoped=False,
        )

    def ack_capture_frame(self, count: int = 1) -> int:
        remaining = self._call_device_count_method(
            self.capture,
            ("ack_frame", "ack_frames", "acknowledge_frame", "acknowledge_frames"),
            count=count,
        )
        if remaining is not None:
            return remaining
        return self._capture_queue_length()

    def flush_capture(self) -> int:
        remaining = self._call_device_count_method(
            self.capture,
            ("flush_capture", "flush", "clear_capture", "clear"),
        )
        if remaining is not None:
            return remaining
        return self._capture_queue_length()

    def mark_disconnected(self, reason: str = "transport_lost") -> None:
        self.connected = False
        self.reconnect_reason = reason
        self.state = "disconnected"

    def mark_connected(self) -> None:
        self.connected = True
        self.reconnect_reason = ""
        if self.state == "disconnected":
            self.state = "idle"

    def _build_asr_event(self, *, text: str, final: bool) -> EventEnvelope:
        return self._build_voice_event(
            name="ei.voice.asr.final" if final else "ei.voice.asr.partial",
            event_type="dialogue",
            source=self._head_source,
            target=self._brain_target,
            content={"text": text, "final": final},
            builder_name="build_voice_asr_event",
            builder_kwargs={"text": text, "final": final},
            round_scoped=True,
        )

    def _playback_state_event(self, *, started: bool, state: str, reason: str = "") -> EventEnvelope:
        content = {"state": state}
        if reason:
            content["reason"] = reason
        return self._build_voice_event(
            name="ei.voice.playback.started" if started else "ei.voice.playback.stopped",
            event_type="dialogue",
            source=self._head_source,
            target=self._brain_target,
            content=content,
            builder_name="build_voice_playback_state_event",
            builder_kwargs={"started": started, "reason": reason},
            round_scoped=True,
        )

    def _build_voice_event(
        self,
        *,
        name: str,
        event_type: str,
        source: Mapping[str, Any],
        target: Mapping[str, Any],
        content: Mapping[str, Any],
        builder_name: str,
        builder_kwargs: Mapping[str, Any],
        round_scoped: bool,
    ) -> EventEnvelope:
        self._sequence += 1
        common_kwargs = {
            "source": source,
            "session_id": self.session_id,
            "round_id": self.round_id if round_scoped else None,
            "trace_id": self.trace_id,
            "sequence": self._sequence,
            "target": target,
        }
        builder = getattr(builders, builder_name, None)
        if callable(builder):
            event = builder(**dict(builder_kwargs), **common_kwargs)
            event.content.update(dict(content))
            return event

        try:
            return build_event(
                name=name,
                event_type=event_type,
                source=source,
                target=target,
                content=content,
                session_id=self.session_id,
                round_id=self.round_id,
                trace_id=self.trace_id,
                sequence=self._sequence,
                priority="realtime" if event_type != "control" else "normal",
                round_scoped=round_scoped,
            )
        except ValueError:
            return EventEnvelope(
                event_id=self._ids.event_id(),
                event_type=event_type,
                name=name,
                time=self._ids.time(),
                sequence=self._sequence,
                request_id=self._ids.request_id(),
                session_id=self.session_id,
                round_id=self.round_id if round_scoped else "",
                trace_id=self.trace_id,
                source=_source_ref(source),
                target=_target_ref(target),
                priority="realtime" if event_type != "control" else "normal",
                content=dict(content),
            )

    def _read_capture_frame(self) -> Mapping[str, Any] | None:
        for method_name in ("read_frame", "capture_audio_frame", "read"):
            reader = getattr(self.capture, method_name, None)
            if callable(reader):
                frame = reader()
                break
        else:
            return None

        if frame is None:
            return None
        if isinstance(frame, bytes):
            return {"audioBase64": base64.b64encode(frame).decode("ascii")}
        if isinstance(frame, Mapping):
            return frame
        raise TypeError("capture frame must be bytes, mapping, or None")

    def _probe_capture_barge_in(self) -> dict[str, Any]:
        for method_name in ("probe_barge_in", "detect_barge_in", "is_voice_active"):
            probe = getattr(self.capture, method_name, None)
            if callable(probe):
                result = probe()
                if isinstance(result, Mapping):
                    return dict(result)
                return {"detected": bool(result)}
        return {"detected": False, "reason": "not_wired"}

    def _capture_queue_length(self) -> int:
        for method_name in ("queue_length", "pending_frames", "frames_left"):
            reader = getattr(self.capture, method_name, None)
            if callable(reader):
                try:
                    return max(0, int(reader()))
                except (TypeError, ValueError):
                    return 0

        frames = getattr(self.capture, "frames", None)
        if isinstance(frames, (list, tuple)):
            return len(frames)
        return 0

    @staticmethod
    def _call_device_count_method(
        device: Any,
        method_names: tuple[str, ...],
        *,
        count: int | None = None,
    ) -> int | None:
        for method_name in method_names:
            method = getattr(device, method_name, None)
            if not callable(method):
                continue
            try:
                result = method() if count is None else method(max(0, int(count)))
            except TypeError:
                result = method()
            if result is None:
                return None
            try:
                return max(0, int(result))
            except (TypeError, ValueError):
                return None
        return None

    def _device_health(self, device: Any) -> dict[str, Any]:
        health = getattr(device, "health", None)
        if callable(health):
            result = health()
            if isinstance(result, Mapping):
                return dict(result)
            return {"status": str(result)}
        return {"status": "unknown"}


__all__ = ["EiVoiceGateway"]
