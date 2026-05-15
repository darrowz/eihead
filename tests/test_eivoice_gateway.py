from __future__ import annotations

from eihead.eivoice_runtime.gateway import EiVoiceGateway
from eiprotocol.models import EventEnvelope


class FakeCapture:
    def __init__(self) -> None:
        self.frames = [
            {
                "audio_base64": "UklGRg==",
                "sample_rate_hz": 16000,
                "channels": 1,
                "format": "pcm16",
                "duration_ms": 20,
                "rms_dbfs": -31.0,
            }
        ]
        self.barge_in = False

    def read_frame(self) -> dict[str, object] | None:
        if not self.frames:
            return None
        return self.frames.pop(0)

    def probe_barge_in(self) -> dict[str, object]:
        return {
            "detected": self.barge_in,
            "reason": "near_field_speech",
            "rms_dbfs": -22.5,
        }

    def health(self) -> dict[str, object]:
        return {"status": "ok", "frames_left": len(self.frames)}


class FakePlayback:
    def __init__(self) -> None:
        self.chunks: list[dict[str, object]] = []
        self.started = 0
        self.stops: list[str] = []

    def enqueue_chunk(self, chunk: dict[str, object]) -> None:
        self.chunks.append(dict(chunk))

    def start(self) -> None:
        self.started += 1

    def stop(self, reason: str = "completed") -> None:
        self.stops.append(reason)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "buffered": len(self.chunks)}


class AckedCapture:
    def __init__(self) -> None:
        self._pending = [
            {
                "audio_base64": "UklGRg==",
                "sample_rate_hz": 16000,
                "channels": 1,
                "format": "pcm16",
                "duration_ms": 20,
            }
        ]
        self._in_flight = 0

    def read_frame(self) -> dict[str, object] | None:
        if not self._pending:
            return None
        self._in_flight += 1
        return dict(self._pending[0])

    def pending_frames(self) -> int:
        return len(self._pending) + self._in_flight

    def ack_frame(self, count: int = 1) -> int:
        acked = min(max(0, int(count)), self._in_flight)
        self._in_flight -= acked
        for _ in range(min(acked, len(self._pending))):
            self._pending.pop(0)
        return self.pending_frames()

    def flush(self) -> int:
        self._pending.clear()
        self._in_flight = 0
        return 0

    def health(self) -> dict[str, object]:
        return {"status": "ok", "frames_left": self.pending_frames()}


def test_capture_audio_frame_emits_voice_frame_without_real_devices() -> None:
    gateway = EiVoiceGateway(
        session_id="s1",
        actor_id="u1",
        capture=FakeCapture(),
        playback=FakePlayback(),
        stream_id="mic-a",
        round_id="round-1",
    )

    events = gateway.capture_audio_frame()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, EventEnvelope)
    assert event.name == "ei.voice.audio.frame"
    assert event.event_type == "observation"
    assert event.source.domain == "eihead"
    assert event.session_id == "s1"
    assert event.round_id == ""
    assert event.content["streamId"] == "mic-a"
    assert event.content["chunkIndex"] == 0
    assert event.content["audioBase64"] == "UklGRg=="
    assert event.content["sampleRateHz"] == 16000
    assert event.content["rmsDbfs"] == -31.0


def test_asr_partial_and_final_update_transcript_and_emit_voice_events() -> None:
    gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=FakeCapture(), playback=FakePlayback())

    partial = gateway.accept_asr_partial("你好")
    final = gateway.accept_asr_final("你好鸿途")

    assert partial.name == "ei.voice.asr.partial"
    assert partial.content["text"] == "你好"
    assert partial.content["final"] is False
    assert final.name == "ei.voice.asr.final"
    assert final.content["text"] == "你好鸿途"
    assert final.content["final"] is True
    assert gateway.transcript_partial == ""
    assert gateway.transcript_final == "你好鸿途"


def test_tts_queue_and_playback_state_machine_use_fake_playback() -> None:
    playback = FakePlayback()
    gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=FakeCapture(), playback=playback)

    chunk = gateway.enqueue_tts_chunk("AAEC", sentence_id="sent-1")
    started = gateway.start_playback()
    stopped = gateway.stop_playback(reason="completed")

    assert chunk.name == "ei.voice.tts.chunk"
    assert chunk.content["audioBase64"] == "AAEC"
    assert chunk.content["sentenceId"] == "sent-1"
    assert playback.chunks == [
        {
            "streamId": chunk.content["streamId"],
            "chunkIndex": 0,
            "audioBase64": "AAEC",
            "sentenceId": "sent-1",
        }
    ]
    assert playback.started == 1
    assert playback.stops == ["completed"]
    assert started.name == "ei.voice.playback.started"
    assert started.content["state"] == "playing"
    assert stopped.name == "ei.voice.playback.stopped"
    assert stopped.content["state"] == "stopped"
    assert stopped.content["reason"] == "completed"


def test_probe_barge_in_only_emits_event_while_playing_and_detected() -> None:
    capture = FakeCapture()
    gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=capture, playback=FakePlayback())

    capture.barge_in = True
    assert gateway.probe_barge_in() is None

    gateway.enqueue_tts_chunk("AAEC")
    gateway.start_playback()
    barge = gateway.probe_barge_in()

    assert barge is not None
    assert barge.name == "ei.voice.barge_in.detected"
    assert barge.content["reason"] == "near_field_speech"
    assert barge.content["state"] == "barge_in"
    assert gateway.state == "barge_in"


def test_heartbeat_reports_queue_lengths_state_health_and_reconnect_state() -> None:
    capture = FakeCapture()
    playback = FakePlayback()
    gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=capture, playback=playback, round_id="round-1")

    gateway.capture_audio_frame()
    gateway.enqueue_tts_chunk("AAEC")
    gateway.mark_disconnected("transport_lost")
    heartbeat = gateway.heartbeat()

    assert heartbeat.name == "ei.voice.session.heartbeat"
    assert heartbeat.event_type == "control"
    assert heartbeat.round_id == ""
    assert heartbeat.content["state"] == "disconnected"
    assert heartbeat.content["health"]["capture"]["status"] == "ok"
    assert heartbeat.content["health"]["playback"]["status"] == "ok"
    assert heartbeat.content["queueLengths"] == {"capture": 0, "tts": 1}
    assert heartbeat.content["reconnect"]["connected"] is False
    assert heartbeat.content["reconnect"]["reason"] == "transport_lost"


def test_heartbeat_queue_lengths_reflect_current_buffer_not_cumulative_counts() -> None:
    playback = FakePlayback()
    gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=FakeCapture(), playback=playback)

    gateway.capture_audio_frame()
    gateway.enqueue_tts_chunk("AAEC")
    before_stop = gateway.heartbeat()

    gateway.start_playback()
    gateway.stop_playback(reason="completed")
    after_stop = gateway.heartbeat()

    assert before_stop.content["queueLengths"] == {"capture": 0, "tts": 1}
    assert after_stop.content["queueLengths"] == {"capture": 0, "tts": 0}


def test_heartbeat_capture_queue_length_can_drop_after_explicit_ack_or_flush() -> None:
    capture = AckedCapture()
    gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=capture, playback=FakePlayback())

    gateway.capture_audio_frame()
    before_ack = gateway.heartbeat()

    gateway.ack_capture_frame()
    after_ack = gateway.heartbeat()

    flush_capture = AckedCapture()
    flush_gateway = EiVoiceGateway(session_id="s1", actor_id="u1", capture=flush_capture, playback=FakePlayback())
    flush_gateway.capture_audio_frame()
    before_flush = flush_gateway.heartbeat()

    flush_gateway.flush_capture()
    after_flush = flush_gateway.heartbeat()

    assert before_ack.content["queueLengths"]["capture"] == 2
    assert after_ack.content["queueLengths"]["capture"] == 0
    assert before_flush.content["queueLengths"]["capture"] == 2
    assert after_flush.content["queueLengths"]["capture"] == 0
