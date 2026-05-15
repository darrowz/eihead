from __future__ import annotations

import base64

from eihead.eivoice_runtime import SimulatedStreamingTtsProvider, StreamingTtsSession


class ManualClock:
    def __init__(self) -> None:
        self.now_s = 1000.0

    def __call__(self) -> float:
        return self.now_s

    def advance_ms(self, value: float) -> None:
        self.now_s += value / 1000.0


def test_streaming_tts_session_emits_sentence_start_audio_chunks_and_complete() -> None:
    clock = ManualClock()
    session = StreamingTtsSession(
        provider=SimulatedStreamingTtsProvider(chunk_chars=3, chunk_duration_ms=40),
        clock=clock,
    )

    stream = session.synthesize(
        text="你好，真流式。",
        round_id="round-tts-1",
        cancellation_token="token-tts-1",
        voice_code="joyinside_warm_zh_cn",
        emotion="warm",
        speed=1.08,
        volume=0.72,
    )

    sentence_start = next(stream)
    clock.advance_ms(37)
    events = [sentence_start, *list(stream)]
    audio_chunks = [event for event in events if event["event"] == "audio_chunk"]
    complete = events[-1]

    assert [events[0]["event"], complete["event"]] == ["sentence_start", "complete"]
    assert events[0]["name"] == "ei.voice.tts.sentence_start"
    assert events[0]["round_id"] == "round-tts-1"
    assert events[0]["content"]["sentenceId"] == "round-tts-1-sentence-1"
    assert events[0]["content"]["voiceCode"] == "joyinside_warm_zh_cn"
    assert events[0]["content"]["emotion"] == "warm"
    assert len(audio_chunks) >= 2
    assert all(chunk["name"] == "ei.voice.tts.chunk" for chunk in audio_chunks)
    assert all(chunk["round_id"] == "round-tts-1" for chunk in audio_chunks)
    assert base64.b64decode(audio_chunks[0]["content"]["audioBase64"]).startswith(b"simtts|")
    assert audio_chunks[0]["content"]["voiceCode"] == "joyinside_warm_zh_cn"
    assert audio_chunks[0]["content"]["speed"] == 1.08
    assert complete["name"] == "ei.voice.tts.complete"
    assert complete["content"]["cancelled"] is False
    assert complete["content"]["chunkCount"] == len(audio_chunks)

    status = session.status()

    assert status["round_id"] == "round-tts-1"
    assert status["first_chunk_latency"] == 37.0
    assert status["chunk_count"] == len(audio_chunks)
    assert status["voice_code"] == "joyinside_warm_zh_cn"
    assert status["emotion"] == "warm"
    assert status["cancelled"] is False
    assert status["provider_state"]["provider"] == "simulated"
    assert status["provider_state"]["last_voice_code"] == "joyinside_warm_zh_cn"


def test_streaming_tts_interrupt_cancels_round_and_stops_old_audio_chunks() -> None:
    clock = ManualClock()
    session = StreamingTtsSession(
        provider=SimulatedStreamingTtsProvider(chunk_chars=2, chunk_duration_ms=30),
        clock=clock,
    )
    stream = session.synthesize(
        text="旧轮次不应继续播放",
        round_id="round-old",
        cancellation_token="token-old",
        voice_code="joyinside_warm_zh_cn",
        emotion="warm",
    )

    assert next(stream)["event"] == "sentence_start"
    cancellation = session.interrupt(reason="user_barge_in")
    remaining = list(stream)

    assert cancellation["cancelled"] is True
    assert cancellation["round_id"] == "round-old"
    assert [event["event"] for event in remaining] == ["complete"]
    assert remaining[0]["content"]["cancelled"] is True
    assert remaining[0]["content"]["reason"] == "user_barge_in"
    assert not any(event["event"] == "audio_chunk" for event in remaining)

    old_status = session.status(round_id="round-old")
    assert old_status["cancelled"] is True
    assert old_status["round_id"] == "round-old"
    assert old_status["chunk_count"] == 0
    assert old_status["provider_state"]["provider"] == "simulated"

    new_events = list(
        session.synthesize(
            text="新轮次可以继续播放",
            round_id="round-new",
            cancellation_token="token-new",
            voice_code="joyinside_clear_zh_cn",
            emotion="focused",
        )
    )

    assert any(event["event"] == "audio_chunk" for event in new_events)
    assert all(event["round_id"] == "round-new" for event in new_events)
    assert session.status()["round_id"] == "round-new"


def test_streaming_tts_superseded_round_updates_session_interrupt_diagnostics() -> None:
    session = StreamingTtsSession(
        provider=SimulatedStreamingTtsProvider(chunk_chars=2, chunk_duration_ms=30),
        clock=ManualClock(),
    )
    old_stream = session.synthesize(
        text="旧轮次",
        round_id="round-old-2",
        cancellation_token="token-old-2",
    )

    assert next(old_stream)["event"] == "sentence_start"

    new_stream = session.synthesize(
        text="新轮次",
        round_id="round-new-2",
        cancellation_token="token-new-2",
    )
    old_remaining = list(old_stream)

    assert next(new_stream)["event"] == "sentence_start"
    assert [event["event"] for event in old_remaining] == ["complete"]
    assert old_remaining[0]["content"]["cancelled"] is True
    assert old_remaining[0]["content"]["reason"] == "superseded"

    old_status = session.status(round_id="round-old-2")
    current_status = session.status(round_id="round-new-2")

    assert old_status["cancelled"] is True
    assert old_status["reason"] == "superseded"
    assert old_status["superseded_by_round_id"] == "round-new-2"
    assert current_status["interrupt_stop_ready"] is True
    assert current_status["cancelled_round_count"] == 1
    assert current_status["last_interrupt"]["reason"] == "superseded"
    assert current_status["last_interrupt"]["roundId"] == "round-old-2"
    assert current_status["last_interrupt"]["supersededByRoundId"] == "round-new-2"
