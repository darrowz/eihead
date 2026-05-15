from __future__ import annotations

from dataclasses import replace

from eihead.eivoice_runtime import AudioFrame


class ManualClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance_ms(self, milliseconds: int) -> None:
        self.value += milliseconds / 1000


def _frame(sequence: int, *, duration_ms: int = 60) -> AudioFrame:
    return AudioFrame(
        pcm=f"voice-{sequence}".encode("ascii"),
        duration_ms=duration_ms,
        sample_rate_hz=16000,
        channels=1,
        sequence=sequence,
    )


class FakeCaptureSource:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = list(frames)

    def read_frame(self) -> AudioFrame | None:
        if not self.frames:
            return None
        return self.frames.pop(0)


class PassthroughFrontend:
    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        return replace(frame)

    def readiness(self) -> dict[str, object]:
        return {"mode": "test"}


class FakePlaybackSink:
    def play(self, frame: AudioFrame) -> None:
        self.last_frame = frame

    def stop(self) -> None:
        self.stopped = True


def test_streaming_session_emits_partial_and_final_events_for_60ms_frame() -> None:
    from eihead.eivoice_runtime import SimulatedStreamingAsrProvider, StreamingAsrSession

    clock = ManualClock()
    session = StreamingAsrSession(
        session_id="session-1",
        round_id="round-1",
        provider=SimulatedStreamingAsrProvider(
            partial_text="ni hao",
            final_text="ni hao final",
            final_after_frames=1,
        ),
        clock=clock,
    )

    clock.advance_ms(42)
    events = session.accept_frame(_frame(1))
    event_payloads = [event.to_dict() for event in events]

    assert [event.name for event in events] == ["ei.voice.asr.partial", "ei.voice.asr.final"]
    assert event_payloads[0]["sessionId"] == "session-1"
    assert event_payloads[0]["roundId"] == "round-1"
    assert event_payloads[0]["content"]["text"] == "ni hao"
    assert event_payloads[0]["content"]["final"] is False
    assert event_payloads[0]["content"]["latencyMs"] == 42
    assert event_payloads[0]["content"]["frameIndex"] == 1
    assert event_payloads[0]["content"]["framesReceived"] == 1
    assert event_payloads[0]["content"]["framesProcessed"] == 1
    assert event_payloads[0]["content"]["framesDropped"] == 0
    assert event_payloads[1]["content"]["text"] == "ni hao final"
    assert event_payloads[1]["content"]["final"] is True
    assert session.drain_events() == events
    assert session.drain_events() == []


def test_streaming_session_drops_oldest_frames_under_provider_backpressure() -> None:
    from eihead.eivoice_runtime import SimulatedStreamingAsrProvider, StreamingAsrSession

    provider = SimulatedStreamingAsrProvider(
        partial_text="latest",
        final_text="latest final",
        final_after_frames=2,
        blocked=True,
    )
    session = StreamingAsrSession(
        session_id="session-2",
        round_id="round-2",
        provider=provider,
        max_inflight_frames=2,
    )

    assert session.accept_frame(_frame(1)) == []
    assert session.accept_frame(_frame(2)) == []
    assert session.accept_frame(_frame(3)) == []
    assert session.accept_frame(_frame(4)) == []

    blocked = session.diagnostics()
    assert blocked["provider_state"] == "backpressure"
    assert blocked["frames_received"] == 4
    assert blocked["frames_pending"] == 2
    assert blocked["frames_dropped"] == 2
    assert blocked["dropped_oldest"] == 2
    assert blocked["latest_voice"]["sequence"] == 4
    assert blocked["latest_voice"]["duration_ms"] == 60

    provider.set_blocked(False)
    events = session.flush()

    assert [event.frame_index for event in events] == [3, 4, 4]
    assert [event.name for event in events] == [
        "ei.voice.asr.partial",
        "ei.voice.asr.partial",
        "ei.voice.asr.final",
    ]
    assert session.diagnostics()["frames_processed"] == 2


def test_streaming_session_diagnostics_track_counts_latency_provider_state_and_errors() -> None:
    from eihead.eivoice_runtime import SimulatedStreamingAsrProvider, StreamingAsrSession

    clock = ManualClock()
    provider = SimulatedStreamingAsrProvider(
        partial_text="diagnostic",
        final_text="diagnostic final",
        final_after_frames=2,
    )
    session = StreamingAsrSession(
        session_id="session-3",
        round_id="round-3",
        provider=provider,
        clock=clock,
    )

    clock.advance_ms(30)
    session.accept_frame(_frame(1))
    clock.advance_ms(90)
    session.accept_frame(_frame(2))
    provider.inject_error("provider warning")

    diagnostics = session.diagnostics()

    assert diagnostics["schema"] == "eihead.eivoice_runtime.asr.v1"
    assert diagnostics["session_id"] == "session-3"
    assert diagnostics["round_id"] == "round-3"
    assert diagnostics["provider"] == "simulated"
    assert diagnostics["provider_state"] == "finalized"
    assert diagnostics["partial_count"] == 2
    assert diagnostics["final_count"] == 1
    assert diagnostics["first_partial_ms"] == 30
    assert diagnostics["final_ms"] == 120
    assert diagnostics["frames_received"] == 2
    assert diagnostics["frames_processed"] == 2
    assert diagnostics["audio_ms_received"] == 120
    assert diagnostics["errors"][-1]["message"] == "provider warning"


def test_streaming_session_cancel_updates_interrupt_diagnostics() -> None:
    from eihead.eivoice_runtime import SimulatedStreamingAsrProvider, StreamingAsrSession

    provider = SimulatedStreamingAsrProvider(
        partial_text="partial",
        final_text="final",
        final_after_frames=3,
        blocked=True,
    )
    session = StreamingAsrSession(
        session_id="session-cancel",
        round_id="round-cancel",
        provider=provider,
        max_inflight_frames=2,
    )

    assert session.accept_frame(_frame(1)) == []
    assert session.accept_frame(_frame(2)) == []
    cancellation = session.cancel(reason="barge_in")
    diagnostics = session.diagnostics()

    assert cancellation["cancelled"] is True
    assert diagnostics["interrupt_stop_ready"] is True
    assert diagnostics["cancelled"] is True
    assert diagnostics["cancel_count"] == 1
    assert diagnostics["last_cancel"]["reason"] == "barge_in"
    assert diagnostics["frames_pending"] == 0


def test_runtime_can_hold_streaming_asr_diagnostics_without_real_provider() -> None:
    from eihead.eivoice_runtime import (
        EiVoiceRuntimeRunner,
        SimulatedStreamingAsrProvider,
        StreamingAsrSession,
    )

    session = StreamingAsrSession(
        session_id="runtime-session",
        round_id="runtime-round",
        provider=SimulatedStreamingAsrProvider(
            partial_text="runtime partial",
            final_text="runtime final",
            final_after_frames=1,
        ),
    )
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_frame(10)),
        audio_frontend=PassthroughFrontend(),
        playback_sink=FakePlaybackSink(),
        asr_session=session,
    )

    assert runner.step_capture() is True
    status = runner.status()
    asr_events = [event.to_dict() for event in runner.drain_asr_events()]

    assert status["diagnostics"]["asr"]["enabled"] is True
    assert status["diagnostics"]["asr"]["partial_count"] == 1
    assert status["diagnostics"]["asr"]["final_count"] == 1
    assert status["diagnostics"]["asr"]["provider"] == "simulated"
    assert asr_events[0]["content"]["text"] == "runtime partial"
    assert asr_events[-1]["content"]["final"] is True


def test_runtime_status_reports_asr_disabled_when_no_provider_is_configured() -> None:
    from eihead.eivoice_runtime import EiVoiceRuntimeRunner

    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=PassthroughFrontend(),
        playback_sink=FakePlaybackSink(),
    )

    assert runner.status()["diagnostics"]["asr"] == {
        "enabled": False,
        "provider": None,
        "provider_state": "not_configured",
    }
