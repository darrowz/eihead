from __future__ import annotations

from eihead.eivoice_runtime import (
    AudioFrame,
    BoundedAudioQueue,
    EiVoiceRuntimeCore,
    VoiceRuntimeStateMachine,
    WakewordRingBuffer,
)


def _frame(sequence: int, duration_ms: int = 60) -> AudioFrame:
    return AudioFrame(
        pcm=f"pcm-{sequence}".encode("ascii"),
        duration_ms=duration_ms,
        sample_rate_hz=16000,
        channels=1,
        sequence=sequence,
    )


def test_ring_buffer_retains_only_recent_1500ms_and_drains() -> None:
    ring = WakewordRingBuffer(capacity_ms=1500)

    for sequence in range(30):
        ring.append(_frame(sequence))

    drained = ring.drain()

    assert [frame.sequence for frame in drained] == list(range(5, 30))
    assert sum(frame.duration_ms for frame in drained) == 1500
    assert ring.drain() == []


def test_ws_send_queue_drops_oldest_when_full() -> None:
    queue = BoundedAudioQueue(capacity=3, full_policy="drop_oldest", name="ws_send_queue")

    for sequence in range(4):
        assert queue.push(_frame(sequence))

    assert [queue.pop().sequence, queue.pop().sequence, queue.pop().sequence] == [1, 2, 3]
    assert queue.pop() is None
    assert queue.stats()["pushed"] == 4
    assert queue.stats()["dropped_oldest"] == 1
    assert queue.stats()["dropped_newest"] == 0


def test_opus_decode_queue_drops_newest_when_full() -> None:
    queue = BoundedAudioQueue(capacity=3, full_policy="drop_newest", name="opus_decode_queue")

    results = [queue.push(_frame(sequence)) for sequence in range(4)]

    assert results == [True, True, True, False]
    assert [queue.pop().sequence, queue.pop().sequence, queue.pop().sequence] == [0, 1, 2]
    assert queue.pop() is None
    assert queue.stats()["pushed"] == 3
    assert queue.stats()["dropped_oldest"] == 0
    assert queue.stats()["dropped_newest"] == 1


def test_state_machine_idle_conversation_idle_and_interrupt_history() -> None:
    state = VoiceRuntimeStateMachine()

    state.wake_detected()
    state.interrupt_requested()
    state.conversation_completed()

    assert state.state == "idle"
    assert [(entry["event"], entry["from"], entry["to"]) for entry in state.history] == [
        ("wake_detected", "idle", "conversation"),
        ("interrupt_requested", "conversation", "conversation"),
        ("conversation_completed", "conversation", "idle"),
    ]


def test_audio_frame_to_dict_exposes_metadata_not_raw_audio() -> None:
    frame = AudioFrame(
        pcm=b"raw-pcm",
        payload=b"opus-payload",
        duration_ms=60,
        sample_rate_hz=16000,
        channels=1,
        sequence=42,
    )

    encoded = frame.to_dict()

    assert encoded["pcm_length"] == 7
    assert encoded["payload_length"] == 12
    assert encoded["duration_ms"] == 60
    assert encoded["sequence"] == 42
    assert "pcm" not in encoded
    assert "payload" not in encoded


def test_runtime_status_includes_joyinside_four_queue_metrics() -> None:
    runtime = EiVoiceRuntimeCore()

    runtime.ws_send_queue.push(_frame(1))
    runtime.opus_decode_queue.push(_frame(2))
    for sequence in range(30):
        runtime.ws_send_queue.push(_frame(sequence))
        runtime.opus_decode_queue.push(_frame(sequence))

    status = runtime.status()

    assert status["state"] == "idle"
    assert set(status["queues"]) == {
        "opus_encode_queue",
        "ws_send_queue",
        "opus_decode_queue",
        "audio_playback_queue",
    }
    assert status["queues"]["opus_encode_queue"]["capacity"] == 3
    assert status["queues"]["ws_send_queue"]["capacity"] == 25
    assert status["queues"]["opus_decode_queue"]["capacity"] == 25
    assert status["queues"]["audio_playback_queue"]["capacity"] == 3
    assert status["queues"]["ws_send_queue"]["dropped_oldest"] > 0
    assert status["queues"]["opus_decode_queue"]["dropped_newest"] > 0
    assert status["interruptStopReady"] is False
    assert status["interrupt_stop_ready"] is False
    assert status["lastInterrupt"] is None
    assert status["last_interrupt"] is None
    assert status["cancelledRoundCount"] == 0
    assert status["cancelled_round_count"] == 0
