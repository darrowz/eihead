from __future__ import annotations

import json

from eihead.ear import (
    EarDeviceConfig,
    build_ear_realtime_status,
    legacy_ear_details_to_status,
    read_ear_config_from_legacy_details,
)


def _legacy_snapshot(*, capture_details: dict[str, object], asr_details: dict[str, object]) -> dict[str, object]:
    return {
        "organs": {
            "ear": {
                "subfunctions": {
                    "capture": {"details": capture_details},
                    "asr": {"details": asr_details},
                }
            }
        }
    }


def test_ear_device_config_defaults_are_truthful() -> None:
    config = EarDeviceConfig()

    assert config.device == "default"
    assert config.sample_rate == 16000
    assert config.channels == 1
    assert config.provider == "sherpa_onnx"
    assert config.model == ""
    assert config.streaming_vad is False
    assert config.vad_frame_ms == 80
    assert config.vad_min_capture_ms == 0
    assert config.to_dict()["transcribe_vad_miss"] is False


def test_streaming_vad_field_roundtrips_from_legacy_capture_details() -> None:
    capture_details = {
        "capture_device": "hw:0,0",
        "sample_rate": 24000,
        "channels": 2,
        "streaming_vad": True,
        "vad_frame_ms": 96,
        "vad_min_voice_ms": 180,
        "vad_miss_rms_threshold": 0.011,
    }
    asr_details = {"provider": "sherpa_onnx", "model_dir": "/models/zipformer", "status": "transcribed", "transcript": "ok"}

    config = read_ear_config_from_legacy_details(capture_details=capture_details, asr_details=asr_details)

    assert config.streaming_vad is True
    assert config.vad_frame_ms == 96
    assert config.vad_min_voice_ms == 180
    assert config.vad_miss_rms_threshold == 0.011


def test_legacy_capture_asr_details_map_to_realtime_status_fields() -> None:
    capture_details = {
        "capture_device": "hw:0,0",
        "sample_rate": 48000,
        "channels": 1,
        "status": "captured",
        "capture_stderr": "",
        "dbfs": -22.0,
        "rms_level": 0.089,
        "vad_triggered": True,
        "elapsed_ms": 42.0,
    }
    asr_details = {
        "provider": "sherpa_onnx",
        "model_dir": "/models/zipformer",
        "status": "transcribed",
        "transcript": "hello ear",
        "elapsed_ms": 23.5,
    }

    status = legacy_ear_details_to_status(_legacy_snapshot(capture_details=capture_details, asr_details=asr_details))

    assert status.not_wired is False
    assert status.status == "ok"
    assert status.capture.status == "captured"
    assert status.capture.vad_triggered is True
    assert status.audio_level == -22.0
    assert status.rms == 0.089
    assert status.capture_elapsed_ms == 42.0
    assert status.decode_elapsed_ms == 23.5
    assert status.total_elapsed_ms == 65.5
    assert status.transcript == "hello ear"
    assert status.final is True
    assert status.config.device == "hw:0,0"
    assert status.config.model == "/models/zipformer"


def test_vad_capture_and_asr_stage_latencies_are_reported_without_live_audio() -> None:
    status = build_ear_realtime_status(
        capture_details={
            "capture_device": "hw:0,0",
            "sample_rate": 48000,
            "channels": 1,
            "status": "captured",
            "vad_triggered": True,
            "vad_elapsed_ms": 8.25,
            "capture_elapsed_ms": 42.0,
        },
        asr_details={
            "provider": "sherpa_onnx",
            "model_dir": "/models/zipformer",
            "status": "transcribed",
            "transcript": "hello honjia",
            "asr_decode_elapsed_ms": 23.5,
        },
        config=EarDeviceConfig(device="hw:0,0", provider="sherpa_onnx", model="/models/zipformer"),
    )

    assert status.not_wired is False
    assert status.capture.vad_elapsed_ms == 8.25
    assert status.asr.decode_elapsed_ms == 23.5
    assert status.stage_latency_ms == {
        "vad": 8.25,
        "capture": 42.0,
        "asr": 23.5,
        "total": 73.75,
    }
    assert status.to_dict()["stage_latency_ms"]["vad"] == 8.25


def test_not_wired_when_missing_device_or_asr_model() -> None:
    status = build_ear_realtime_status(
        capture_details={"capture_device": "", "status": "ok", "sample_rate": 16000, "channels": 1},
        asr_details={"provider": "sherpa_onnx", "status": "transcribed", "transcript": "hello"},
    )

    assert status.not_wired is True
    assert status.status == "degraded"
    assert "no microphone device configured" in status.readiness_message
    assert "no asr model configured" in status.readiness_message


def test_noop_or_live_probe_skipped_is_not_reported_healthy() -> None:
    status = build_ear_realtime_status(
        capture_details={"capture_device": "hw:0,0", "status": "healthy", "driver": "noop"},
        asr_details={"status": "live_probe_skipped", "provider": "sherpa_onnx", "model_dir": "/models/zipformer"},
        config={"device": "hw:0,0", "provider": "sherpa_onnx", "model": "/models/zipformer"},
    )

    assert status.not_wired is True
    assert status.status == "degraded"
    assert "capture driver is noop" in status.readiness_message


def test_empty_transcript_stays_visible_and_is_not_a_error_state() -> None:
    status = build_ear_realtime_status(
        capture_details={"capture_device": "hw:0,0", "status": "ok", "dbfs": -16.0, "rms_level": 0.02},
        asr_details={"provider": "sherpa_onnx", "model_dir": "/models/zipformer", "status": "silence", "transcript": "", "elapsed_ms": 0.2},
        config=EarDeviceConfig(device="hw:0,0", provider="sherpa_onnx", model="/models/zipformer"),
    )

    assert status.not_wired is False
    assert status.transcript == ""
    assert status.last_error is None
    assert status.status == "ok"


def test_realtime_status_to_dict_is_json_friendly() -> None:
    status = build_ear_realtime_status(
        capture_details={"capture_device": "hw:0,0", "status": "ok", "dbfs": -16.0, "rms_level": 0.02},
        asr_details={"provider": "sherpa_onnx", "model_dir": "/models/zipformer", "status": "transcribed", "transcript": "test"},
        config=EarDeviceConfig(device="hw:0,0", provider="sherpa_onnx", model="/models/zipformer"),
    )

    payload = status.to_dict()
    roundtrip = json.loads(json.dumps(payload))

    assert roundtrip["schema"] == "eihead.ear.realtime_status.v1"
    assert isinstance(roundtrip["config"], dict)
    assert isinstance(roundtrip["capture"], dict)
    assert isinstance(roundtrip["asr"], dict)
    assert roundtrip["not_wired"] is False
