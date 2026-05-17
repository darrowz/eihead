from __future__ import annotations

from contextlib import contextmanager
import json
import math
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterator
from urllib import request

import pytest

from eihead.monitoring.voice_test import run_voice_manual_test
from eihead.monitoring.web import create_server


class VoiceTestApp:
    def __init__(self, config_path: Path) -> None:
        self.config_path = str(config_path)

    def status(self) -> dict[str, Any]:
        return {"ok": True, "status": "ok", "runtime": "eihead", "node_id": "honjia-test"}

    def capabilities(self) -> dict[str, Any]:
        return {"capabilities": {}}


def test_speaker_manual_test_plays_configured_alsa_output(tmp_path: Path) -> None:
    app = VoiceTestApp(_write_voice_config(tmp_path))
    calls: list[list[str]] = []

    def runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    payload = run_voice_manual_test(
        app,
        {"kind": "speaker", "frequency_hz": 440, "duration_s": 0.2},
        output_dir=tmp_path,
        runner=runner,
        timestamp=10.0,
    )

    assert payload["schema"] == "eihead.monitor.voice_manual_test.v1"
    assert payload["kind"] == "speaker"
    assert payload["status"] == "ok"
    assert payload["device"] == "plughw:CARD=SPA3700,DEV=0"
    assert payload["frequency_hz"] == 440
    assert payload["duration_s"] == pytest.approx(0.2)
    assert payload["readiness_message"] == "扬声器已播放 440Hz 测试音 0.20s"
    assert Path(payload["file"]).exists()
    assert calls == [["aplay", "-q", "-D", "plughw:CARD=SPA3700,DEV=0", payload["file"]]]


def test_microphone_manual_test_records_and_reports_audio_levels(tmp_path: Path) -> None:
    app = VoiceTestApp(_write_voice_config(tmp_path))
    calls: list[list[str]] = []

    def runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        _write_test_wav(Path(args[-1]), sample_rate=16000, duration_s=0.25, amplitude=12000)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    payload = run_voice_manual_test(
        app,
        {"kind": "microphone", "duration_s": 0.25},
        output_dir=tmp_path,
        runner=runner,
        timestamp=11.0,
    )

    assert payload["kind"] == "microphone"
    assert payload["status"] == "ok"
    assert payload["device"] == "plughw:CARD=U4K,DEV=0"
    assert payload["sample_rate"] == 16000
    assert payload["channels"] == 1
    assert payload["duration_s"] == pytest.approx(0.25)
    assert payload["rms_dbfs"] > -12.0
    assert payload["peak_dbfs"] > -9.0
    assert payload["readiness_message"].startswith("麦克风已录制 0.25s")
    assert calls == [
        [
            "arecord",
            "-q",
            "-D",
            "plughw:CARD=U4K,DEV=0",
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-d",
            "1",
            payload["file"],
        ]
    ]


def test_monitor_exposes_voice_manual_test_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = VoiceTestApp(_write_voice_config(tmp_path))

    def fake_voice_test(
        runtime_app: Any,
        request_payload: dict[str, Any],
        *,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        assert runtime_app is app
        assert request_payload == {"kind": "speaker"}
        return {
            "schema": "eihead.monitor.voice_manual_test.v1",
            "kind": "speaker",
            "status": "ok",
            "readiness_message": "扬声器已播放 660Hz 测试音 0.70s",
            "captured_at_ts": timestamp,
        }

    monkeypatch.setattr("eihead.monitoring.web.run_voice_manual_test", fake_voice_test)

    with running_server(app) as base_url:
        payload = post_json(f"{base_url}/api/voice/test", {"kind": "speaker"})

    assert payload["kind"] == "speaker"
    assert payload["status"] == "ok"
    assert payload["captured_at_ts"] == 123.0


def test_lightweight_monitor_renders_voice_manual_test_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIHEAD_MONITOR_LIGHTWEIGHT_ROOT", "1")
    app = VoiceTestApp(_write_voice_config(tmp_path))

    with running_server(app) as base_url:
        body = read_text(f"{base_url}/")

    assert "语音自测" in body
    assert "测麦克风" in body
    assert "播放测试音" in body
    assert "/api/voice/test" in body


@contextmanager
def running_server(app: Any) -> Iterator[str]:
    server = create_server(app, host="127.0.0.1", port=0, clock=lambda: 123.0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def read_text(url: str) -> str:
    with request.urlopen(url, timeout=2.0) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=2.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _write_voice_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "devices:",
                "  microphone:",
                "    device: plughw:CARD=U4K,DEV=0",
                "    sample_rate: 16000",
                "    channels: 1",
                "  speaker:",
                "    device: plughw:CARD=SPA3700,DEV=0",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_test_wav(
    path: Path,
    *,
    sample_rate: int,
    duration_s: float,
    amplitude: int,
) -> None:
    import wave

    total_samples = int(sample_rate * duration_s)
    frames = bytearray()
    for index in range(total_samples):
        sample = int(amplitude * math.sin(2 * math.pi * 440 * (index / sample_rate)))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
