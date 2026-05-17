from __future__ import annotations

import base64
from dataclasses import replace
import subprocess

from eihead.eivoice_runtime import AudioFrame, EiVoiceRuntimeRunner, FakeWebSocketTransport


def _pcm_frame(sequence: int, payload: bytes | None = None) -> AudioFrame:
    return AudioFrame(
        pcm=f"pcm-{sequence}".encode("ascii"),
        payload=payload or b"",
        duration_ms=20,
        sample_rate_hz=16000,
        channels=1,
        sequence=sequence,
    )


class FakeCaptureSource:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = list(frames)
        self.read_calls = 0

    def read_frame(self) -> AudioFrame | None:
        self.read_calls += 1
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakeAudioFrontend:
    def __init__(self) -> None:
        self.processed_sequences: list[int] = []
        self._readiness = {
            "aec": {"enabled": True, "available": True},
            "ns": {"enabled": True, "available": True},
            "vad": {"enabled": True, "available": True},
            "loopback": {"enabled": True, "available": True},
            "warnings": [],
        }

    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        self.processed_sequences.append(frame.sequence)
        return replace(frame, pcm=frame.pcm + b"-frontend")

    def readiness(self) -> dict[str, object]:
        return dict(self._readiness)


class FakeCodec:
    def __init__(self) -> None:
        self.encoded_sequences: list[int] = []
        self.decoded_sequences: list[int] = []

    def encode(self, frame: AudioFrame) -> AudioFrame:
        self.encoded_sequences.append(frame.sequence)
        return replace(frame, pcm=b"", payload=frame.pcm + b"-opus")

    def decode(self, frame: AudioFrame) -> AudioFrame:
        self.decoded_sequences.append(frame.sequence)
        return replace(frame, payload=b"", pcm=frame.payload + b"-decoded")


class FakeWsReceiveSource:
    def __init__(self, *frames: AudioFrame) -> None:
        self.frames = list(frames)
        self.read_calls = 0

    def read_frame(self) -> AudioFrame | None:
        self.read_calls += 1
        if not self.frames:
            return None
        return self.frames.pop(0)


class FakePlaybackSink:
    def __init__(self) -> None:
        self.played: list[AudioFrame] = []
        self.stop_calls = 0

    def play(self, frame: AudioFrame) -> None:
        self.played.append(frame)

    def stop(self) -> None:
        self.stop_calls += 1


class StoppableCaptureSource:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1

    def status(self) -> dict[str, object]:
        return {"running": False}


class StubTranscriber:
    def status(self) -> dict[str, object]:
        return {"state": "not_loaded"}


class FakeMiniMaxSynthesizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text: str) -> dict[str, object]:
        self.texts.append(text)
        return {
            "status": "ok",
            "audio_bytes": b"RIFFfake-wav",
            "details": {
                "backend": "minimax",
                "model": "speech-2.8-hd",
                "voice_id": "female-shaonv",
                "audio_size": 12,
            },
        }


class FakeDialogueClient:
    def __init__(self, reply: str = "我是 eibrain。") -> None:
        self.reply = reply
        self.requests: list[dict[str, object]] = []

    def reply_to_transcript(self, text: str, **kwargs: object) -> dict[str, object]:
        self.requests.append({"text": text, **kwargs})
        return {
            "status": "ok",
            "reply_text": self.reply,
            "details": {"provider": "fake_eibrain", "round_id": kwargs.get("round_id", "")},
        }


class OneShotTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, frames: list[AudioFrame]) -> str:
        self.calls += 1
        return self.text

    def status(self) -> dict[str, object]:
        return {"state": "ready"}


def test_native_voice_turn_uses_dialogue_client_reply_before_tts() -> None:
    from eihead.eivoice_runtime.native_loop import NativeVoiceInteractionLoop, NativeVoiceLoopConfig

    capture_source = StoppableCaptureSource()
    dialogue = FakeDialogueClient(reply="我是 eibrain。")
    spoken: list[str] = []

    loop = NativeVoiceInteractionLoop(
        NativeVoiceLoopConfig(playback_echo_cooldown_ms=0),
        capture_source=capture_source,  # type: ignore[arg-type]
        transcriber=OneShotTranscriber("你好鸿佳"),  # type: ignore[arg-type]
        dialogue_client=dialogue,
        tts_synthesizer=None,
    )
    loop._play_text = lambda text: spoken.append(text) or {  # type: ignore[method-assign]
        "status": "ok",
        "success": True,
        "details": {"playback_elapsed_ms": 7},
    }
    loop._frames.append(_pcm_frame(1, payload=b"\x01\x02"))

    loop._finalize_utterance()

    assert dialogue.requests[0]["text"] == "你好鸿佳"
    assert spoken == ["我是 eibrain。"]
    status = loop.voice_status()
    assert status["voice_dialogue"]["last_transcript"] == "你好鸿佳"
    assert status["voice_dialogue"]["last_reply"] == "我是 eibrain。"
    assert status["voice_dialogue"]["last_stage_latency_ms"]["dialogue"] >= 0
    assert status["last_turn"] == {"transcript": "你好鸿佳", "reply": "我是 eibrain。", "status": "turn_complete"}


def test_eibrain_subprocess_dialogue_client_parses_play_speech_action() -> None:
    from eihead.eivoice_runtime.native_loop import EIBrainSubprocessDialogueClient, NativeVoiceLoopConfig

    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            '[{"kind":"play_speech_action","text":"我是 eibrain。"}]',
            "",
        )

    client = EIBrainSubprocessDialogueClient(
        NativeVoiceLoopConfig(
            dialogue_backend="eibrain_subprocess",
            dialogue_command="/opt/eihead/current/.venv/bin/python",
            dialogue_module="apps.cognitive_runtime",
            dialogue_cwd="/dev-project/eibrain",
            dialogue_config_path="/dev-project/eibrain/config/eibrain.honjia.yaml",
            dialogue_pythonpath="/dev-project/eibrain:/dev-project/eiprotocol",
            dialogue_timeout_s=12,
            dialogue_session_id="honjia-voice",
            dialogue_actor_id="darrow",
        ),
        runner=runner,
    )

    result = client.reply_to_transcript("介绍一下你自己", round_id="voice-1", asr_latency_ms=12.5)

    assert result["status"] == "ok"
    assert result["reply_text"] == "我是 eibrain。"
    assert commands[0][:3] == ["/opt/eihead/current/.venv/bin/python", "-m", "apps.cognitive_runtime"]
    assert "--text" in commands[0]
    assert "介绍一下你自己" in commands[0]
    assert result["details"]["event_name"] == "ei.voice.asr.final"


def test_native_voice_speak_prefers_minimax_tts_when_configured() -> None:
    from eihead.eivoice_runtime.native_loop import NativeVoiceInteractionLoop, NativeVoiceLoopConfig

    commands: list[list[str]] = []
    capture_source = StoppableCaptureSource()
    synthesizer = FakeMiniMaxSynthesizer()

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    loop = NativeVoiceInteractionLoop(
        NativeVoiceLoopConfig(
            speaker_device="plughw:CARD=SPA3700,DEV=0",
            tts_backend="minimax",
            minimax_api_key="secret-tts",
            minimax_model="speech-2.8-hd",
            minimax_voice_id="female-shaonv",
            playback_echo_cooldown_ms=0,
        ),
        capture_source=capture_source,  # type: ignore[arg-type]
        transcriber=StubTranscriber(),  # type: ignore[arg-type]
        runner=runner,
        tts_synthesizer=synthesizer,
    )

    outcome = loop.speak("我听到了：你好鸿佳")

    assert outcome["status"] == "ok"
    assert capture_source.stop_calls == 1
    assert synthesizer.texts == ["我听到了：你好鸿佳"]
    assert commands[0][:4] == ["aplay", "-q", "-D", "plughw:CARD=SPA3700,DEV=0"]
    assert outcome["details"]["backend"] == "minimax"
    assert outcome["details"]["model"] == "speech-2.8-hd"
    assert outcome["details"]["voice_id"] == "female-shaonv"


def test_native_voice_speak_uses_chinese_espeak_voice_and_stops_capture() -> None:
    from eihead.eivoice_runtime.native_loop import NativeVoiceInteractionLoop, NativeVoiceLoopConfig

    commands: list[list[str]] = []
    capture_source = StoppableCaptureSource()

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    loop = NativeVoiceInteractionLoop(
        NativeVoiceLoopConfig(
            speaker_device="plughw:CARD=SPA3700,DEV=0",
            tts_command="espeak-ng",
            tts_voice="cmn",
            tts_rate_wpm=150,
            playback_echo_cooldown_ms=0,
        ),
        capture_source=capture_source,  # type: ignore[arg-type]
        transcriber=StubTranscriber(),  # type: ignore[arg-type]
        runner=runner,
    )

    outcome = loop.speak("我听到了：你好鸿佳")

    assert outcome["status"] == "ok"
    assert capture_source.stop_calls == 1
    assert commands[0][:5] == ["espeak-ng", "-v", "cmn", "-s", "150"]
    assert commands[1][:4] == ["aplay", "-q", "-D", "plughw:CARD=SPA3700,DEV=0"]
    assert outcome["details"]["command"] == "espeak-ng"
    assert outcome["details"]["voice"] == "cmn"


def test_runner_step_methods_move_audio_through_worker_queues() -> None:
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(1)),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        ws_receive_source=FakeWsReceiveSource(_pcm_frame(7, payload=b"remote-7")),
        playback_sink=FakePlaybackSink(),
    )

    capture_result = runner.step_capture()
    encode_result = runner.step_encode()
    receive_result = runner.step_receive()
    decode_result = runner.step_decode()
    playback_result = runner.step_playback()

    sent_frame = runner.core.ws_send_queue.pop()

    assert capture_result is True
    assert encode_result is True
    assert receive_result is True
    assert decode_result is True
    assert playback_result is True
    assert sent_frame is not None
    assert sent_frame.sequence == 1
    assert sent_frame.payload == b"pcm-1-frontend-opus"
    assert runner.playback_sink.played[0].sequence == 7
    assert runner.playback_sink.played[0].pcm == b"remote-7-decoded"


def test_runner_step_once_executes_all_workers_and_status_merges_metrics() -> None:
    frontend = FakeAudioFrontend()
    codec = FakeCodec()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(2)),
        audio_frontend=frontend,
        codec=codec,
        ws_receive_source=FakeWsReceiveSource(_pcm_frame(8, payload=b"remote-8")),
        playback_sink=FakePlaybackSink(),
    )

    step_result = runner.step_once()
    status = runner.status()

    assert step_result == {
        "capture": True,
        "encode": True,
        "receive": True,
        "decode": True,
        "playback": True,
    }
    assert status["state"] == "idle"
    assert status["worker_metrics"]["capture_frames"] == 1
    assert status["worker_metrics"]["encode_frames"] == 1
    assert status["worker_metrics"]["receive_frames"] == 1
    assert status["worker_metrics"]["decode_frames"] == 1
    assert status["worker_metrics"]["playback_frames"] == 1
    assert status["worker_metrics"]["step_once_calls"] == 1
    assert status["audio_frontend"] == frontend.readiness()
    assert status["queues"]["opus_encode_queue"]["depth"] == 0
    assert status["queues"]["opus_decode_queue"]["depth"] == 0


def test_runner_can_bridge_encoded_audio_to_transport_and_decode_inbound_tts() -> None:
    transport = FakeWebSocketTransport()
    transport.open()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(11)),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        transport=transport,
        playback_sink=FakePlaybackSink(),
        uid="darrow",
        mid="mid-runner",
    )

    assert runner.step_capture() is True
    assert runner.step_encode() is True
    assert runner.step_send() is True

    outbound = transport.recv_from_client()
    assert outbound is not None
    assert outbound["contentType"] == "AUDIO"
    assert outbound["uid"] == "darrow"
    assert outbound["mid"] == "mid-runner"
    assert base64.b64decode(outbound["content"]["audioBase64"]) == b"pcm-11-frontend-opus"

    transport.deliver_from_server(
        {
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "index": 12,
                "audioBase64": base64.b64encode(b"remote-opus").decode("ascii"),
            },
        }
    )

    assert runner.step_receive() is True
    assert runner.step_decode() is True
    assert runner.step_playback() is True
    assert runner.playback_sink.played[0].sequence == 12
    assert runner.playback_sink.played[0].pcm == b"remote-opus-decoded"
    assert runner.status()["transport"]["transport"] == "fake_websocket"


def test_interrupt_playback_clears_pending_frames_stops_sink_and_tracks_metrics() -> None:
    sink = FakePlaybackSink()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        ws_receive_source=FakeWsReceiveSource(),
        playback_sink=sink,
    )

    assert runner.core.audio_playback_queue.push(_pcm_frame(3))
    assert runner.core.audio_playback_queue.push(_pcm_frame(4))

    cleared = runner.interrupt_playback(reason="user_barge_in", round_id="round-worker-1")
    status = runner.status()

    assert cleared == 2
    assert runner.core.audio_playback_queue.pop() is None
    assert sink.stop_calls == 1
    assert status["interruptStopReady"] is True
    assert status["cancelledRoundCount"] == 1
    assert status["lastInterrupt"]["reason"] == "user_barge_in"
    assert status["lastInterrupt"]["roundId"] == "round-worker-1"
    assert status["lastInterrupt"]["cleared"] == 2
    assert status["worker_metrics"]["playback_interrupts"] == 1
    assert status["worker_metrics"]["playback_frames_cleared"] == 2


def test_interrupt_playback_clears_all_downstream_tts_buffers() -> None:
    transport = FakeWebSocketTransport()
    transport.open()
    sink = FakePlaybackSink()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        transport=transport,
        playback_sink=sink,
    )
    transport.deliver_from_server(
        {
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "index": 5,
                "audioBase64": base64.b64encode(b"transport-old-opus").decode("ascii"),
            },
        }
    )
    assert runner.core.opus_decode_queue.push(_pcm_frame(6, payload=b"decode-old-opus"))
    assert runner.core.audio_playback_queue.push(_pcm_frame(7, payload=b"playback-old-pcm"))

    cleared = runner.interrupt_playback(reason="superseded", round_id="round-worker-old")
    status = runner.status()

    assert cleared == 3
    assert runner.step_receive() is False
    assert runner.step_decode() is False
    assert runner.step_playback() is False
    assert sink.played == []
    assert sink.stop_calls == 1
    assert status["worker_metrics"]["playback_frames_cleared"] == 1
    assert status["worker_metrics"]["decode_frames_cleared"] == 1
    assert status["worker_metrics"]["transport_inbound_events_cleared"] == 1
    assert status["diagnostics"]["interrupt"]["ready"] is True
    assert status["diagnostics"]["interrupt"]["cancelled_round_count"] == 1
    assert status["diagnostics"]["interrupt"]["last_interrupt"]["reason"] == "superseded"
    assert status["diagnostics"]["interrupt"]["last_interrupt"]["transportInboundEventsCleared"] == 1
    assert status["transport"]["queues"]["inbound_queue"]["depth"] == 0


def test_runner_reports_no_work_when_sources_and_queues_are_empty() -> None:
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(),
        audio_frontend=FakeAudioFrontend(),
        codec=FakeCodec(),
        ws_receive_source=FakeWsReceiveSource(),
        playback_sink=FakePlaybackSink(),
    )

    step_result = runner.step_once()
    status = runner.status()

    assert step_result == {
        "capture": False,
        "encode": False,
        "receive": False,
        "decode": False,
        "playback": False,
    }
    assert status["worker_metrics"]["capture_empty_polls"] == 1
    assert status["worker_metrics"]["receive_empty_polls"] == 1
    assert status["worker_metrics"]["idle_steps"] == 1


def test_noop_acoustic_frontend_config_reports_aec_ns_vad_loopback_diagnostics() -> None:
    from eihead.eivoice_runtime import AcousticFrontendConfig, NoOpAcousticFrontend

    default_readiness = NoOpAcousticFrontend().readiness()
    assert default_readiness["healthy"] is True

    frontend = NoOpAcousticFrontend(
        AcousticFrontendConfig(
            aec_enabled=True,
            aec_available=False,
            ns_enabled=True,
            ns_available=True,
            vad_enabled=False,
            vad_available=False,
            loopback_enabled=True,
            loopback_available=True,
        )
    )

    processed = frontend.process_capture(_pcm_frame(31))
    readiness = frontend.readiness()

    assert processed.sequence == 31
    assert readiness["mode"] == "noop"
    assert readiness["aec"] == {"enabled": True, "available": False, "state": "unavailable"}
    assert readiness["ns"] == {"enabled": True, "available": True, "state": "ready"}
    assert readiness["vad"] == {"enabled": False, "available": False, "state": "disabled"}
    assert readiness["loopback"] == {"enabled": True, "available": True, "state": "ready"}
    assert readiness["processed_frames"] == 1
    assert readiness["last_frame_duration_ms"] == 20
    assert "AEC configured but unavailable" in readiness["warnings"]


def test_runner_ws_send_queue_keeps_latest_frames_when_transport_is_slow() -> None:
    from eihead.eivoice_runtime import AcousticFrontendConfig, NoOpAcousticFrontend

    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(*[_pcm_frame(sequence) for sequence in range(30)]),
        audio_frontend=NoOpAcousticFrontend(
            AcousticFrontendConfig(
                aec_enabled=True,
                aec_available=True,
                ns_enabled=True,
                ns_available=True,
                vad_enabled=True,
                vad_available=True,
                loopback_enabled=True,
                loopback_available=True,
            )
        ),
        codec=FakeCodec(),
        playback_sink=FakePlaybackSink(),
    )

    for _ in range(30):
        assert runner.step_capture() is True
        assert runner.step_encode() is True

    status = runner.status()
    queued_sequences: list[int] = []
    while True:
        frame = runner.core.ws_send_queue.pop()
        if frame is None:
            break
        queued_sequences.append(frame.sequence)

    assert queued_sequences == list(range(5, 30))
    assert status["queues"]["ws_send_queue"]["depth"] == 25
    assert status["queues"]["ws_send_queue"]["dropped_oldest"] == 5
    assert status["diagnostics"]["queues"]["ws_send_queue"]["dropped_oldest"] == 5
    assert status["diagnostics"]["audio_frame"]["last_capture_duration_ms"] == 20
    assert status["diagnostics"]["upstream"]["queue_depth"] == 25


def test_runner_status_exposes_joyinside_like_audio_chain_diagnostics() -> None:
    from eihead.eivoice_runtime import AcousticFrontendConfig, NoOpAcousticFrontend

    transport = FakeWebSocketTransport()
    transport.open()
    runner = EiVoiceRuntimeRunner(
        capture_source=FakeCaptureSource(_pcm_frame(41)),
        audio_frontend=NoOpAcousticFrontend(
            AcousticFrontendConfig(
                aec_enabled=True,
                aec_available=True,
                ns_enabled=True,
                ns_available=True,
                vad_enabled=True,
                vad_available=True,
                loopback_enabled=True,
                loopback_available=True,
            )
        ),
        codec=FakeCodec(),
        transport=transport,
        playback_sink=FakePlaybackSink(),
        uid="darrow",
        mid="mid-diagnostics",
    )

    assert runner.step_capture() is True
    assert runner.step_encode() is True
    assert runner.step_send() is True

    outbound = transport.recv_from_client()
    assert outbound is not None
    assert outbound["content"]["durationMs"] == 20
    assert outbound["content"]["sampleRateHz"] == 16000
    assert outbound["content"]["channels"] == 1

    transport.deliver_from_server(
        {
            "contentType": "TTS",
            "content": {
                "eventType": "TTS",
                "index": 42,
                "durationMs": 120,
                "audioBase64": base64.b64encode(b"remote-opus").decode("ascii"),
            },
        }
    )

    assert runner.step_receive() is True
    assert runner.step_decode() is True
    assert runner.step_playback() is True

    diagnostics = runner.status()["diagnostics"]

    assert diagnostics["schema"] == "eihead.eivoice_runtime.diagnostics.v1"
    assert diagnostics["audio_frame"]["last_capture_duration_ms"] == 20
    assert diagnostics["audio_frame"]["last_receive_duration_ms"] == 120
    assert diagnostics["audio_frontend"]["aec"]["enabled"] is True
    assert diagnostics["audio_frontend"]["ns"]["enabled"] is True
    assert diagnostics["audio_frontend"]["vad"]["enabled"] is True
    assert diagnostics["interrupt"]["ready"] is True
    assert diagnostics["interrupt"]["cancelled_round_count"] == 0
    assert diagnostics["interrupt"]["last_interrupt"] is None
    assert diagnostics["upstream"]["state"] == "connected"
    assert diagnostics["downstream"]["state"] == "connected"
    assert diagnostics["heartbeat"]["due"] is False
    assert diagnostics["reconnect"]["ready"] is False
