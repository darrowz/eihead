from __future__ import annotations

from dataclasses import dataclass
import json

from eihead.mouth import playback
from eihead.mouth import (
    MouthTtsConfig,
    SpeechPlaybackStatus,
    build_mouth_status,
)


def test_build_mouth_status_reports_minimax_as_primary_online_provider() -> None:
    config = playback.MouthTtsConfig(
        provider="minimax",
        model="speech-2.5-hd",
        voice_id="female-tianmei",
        output_device="default",
        api_base_url="https://api.minimax.chat",
    )

    status = playback.build_mouth_status(config=config, status="idle")

    assert status.status == "idle"
    assert status.provider == "minimax"
    assert status.model == "speech-2.5-hd"
    assert status.voice_id == "female-tianmei"
    assert status.busy is False
    assert status.not_wired is False
    assert status.health == "online"
    assert status.data_status == "live"
    assert "minimax" in status.readiness_message.lower()


def test_build_mouth_status_marks_noop_as_compat_fallback_not_live_tts() -> None:
    config = playback.MouthTtsConfig(provider="noop", output_device="default")

    status = playback.build_mouth_status(config=config, status="idle")

    assert status.status == "idle"
    assert status.provider == "noop"
    assert status.health == "degraded"
    assert status.data_status == "compat"
    assert status.busy is False
    assert "fallback" in status.readiness_message.lower()


def test_build_mouth_status_maps_last_playback_details_and_busy_state() -> None:
    config = playback.MouthTtsConfig(provider="minimax", model="speech-2.5-hd", voice_id="voice-a")

    status = playback.build_mouth_status(
        config=config,
        status="playing",
        details={
            "text": "你好，鸿图，现在开始播报今天的安排。",
            "synthesis_elapsed_ms": 120,
            "playback_elapsed_ms": 980,
            "total_elapsed_ms": 1100,
        },
    )

    assert status.status == "playing"
    assert status.busy is True
    assert status.text_preview.startswith("你好，鸿图")
    assert status.text_char_count == len("你好，鸿图，现在开始播报今天的安排。")
    assert status.synthesis_elapsed_ms == 120
    assert status.playback_elapsed_ms == 980
    assert status.total_elapsed_ms == 1100


def test_build_mouth_status_reports_tts_stage_latency_and_stop_state() -> None:
    config = playback.MouthTtsConfig(provider="minimax", model="speech-2.5-hd", voice_id="voice-a")

    status = playback.build_mouth_status(
        config=config,
        status="stopped",
        details={
            "text": "stop after this sentence",
            "synthesis_elapsed_ms": 120,
            "playback_elapsed_ms": 980,
            "total_elapsed_ms": 1100,
            "stop": {
                "status": "accepted",
                "success": True,
                "busy_before": True,
                "busy_cleared": True,
                "busy_retained": False,
                "details": {"reason": "user_interrupt"},
            },
        },
    )

    assert status.busy is False
    assert status.playback_state == "stopped"
    assert status.stage_latency_ms == {
        "tts_synthesis": 120.0,
        "tts_playback": 980.0,
        "tts_total": 1100.0,
    }
    assert status.stop is not None
    assert status.stop.status == "stopped"
    assert status.stop.success is True
    assert status.stop.busy_before is True
    assert status.stop.busy_cleared is True
    assert status.stop.busy_retained is False
    assert status.stop.reason == "user_interrupt"
    assert status.to_dict()["stop"]["busy_cleared"] is True


def test_summarize_speak_action_extracts_text_voice_and_session_from_mapping_or_dataclass() -> None:
    from_mapping = playback.summarize_speak_action(
        {
            "type": "speak",
            "text": "请看前方。",
            "session_id": "session-1",
            "voice_id": "voice-a",
        }
    )
    from_dataclass = playback.summarize_speak_action(
        FakeSpeakAction(
            text="请看前方。",
            session_id="session-2",
            voice_id="voice-b",
        )
    )

    assert from_mapping.text == "请看前方。"
    assert from_mapping.text_char_count == len("请看前方。")
    assert from_mapping.voice_id == "voice-a"
    assert from_mapping.session_id == "session-1"
    assert from_dataclass.voice_id == "voice-b"
    assert from_dataclass.session_id == "session-2"


def test_summarize_stop_speech_result_reports_not_busy_after_stop() -> None:
    summary = playback.summarize_stop_speech_result(
        {
            "status": "accepted",
            "success": True,
            "details": {"reason": "user_interrupt"},
        }
    )

    assert summary.status == "stopped"
    assert summary.busy is False
    assert summary.success is True
    assert summary.reason == "user_interrupt"


def test_status_and_summaries_are_json_friendly_and_exported_from_package() -> None:
    assert MouthTtsConfig is playback.MouthTtsConfig
    assert SpeechPlaybackStatus is playback.SpeechPlaybackStatus
    assert build_mouth_status is playback.build_mouth_status

    status = playback.build_mouth_status(
        config=playback.MouthTtsConfig(provider="piper", model="zh_CN-huayan-medium", voice_id="huayan"),
        status="completed",
        details={"text_preview": "播报结束", "text_char_count": 4},
    )
    payload = status.to_dict()

    assert payload["provider"] == "piper"
    assert payload["health"] == "degraded"
    assert payload["data_status"] == "compat"
    assert json.loads(json.dumps(payload, ensure_ascii=False))["status"] == "completed"


@dataclass(slots=True)
class FakeSpeakAction:
    text: str
    session_id: str
    voice_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": "speak",
            "text": self.text,
            "session_id": self.session_id,
            "voice_id": self.voice_id,
        }
