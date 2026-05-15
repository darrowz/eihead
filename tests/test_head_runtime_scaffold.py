from __future__ import annotations

import io
import json
from pathlib import Path
import tomllib

from apps.head_runtime.app import HeadRuntimeApp as AppsHeadRuntimeApp
from eihead.runtime.app import HeadRuntimeApp
import pytest

from eihead.runtime.legacy_body import LegacyBodyRuntimeAdapter
from eihead.runtime import cli
from eihead.protocol import VisionObservation


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organ_count": 4,
            "organs": {
                "ear": {"status": "mock"},
                "eye": {"status": "mock"},
            },
        }


class FailingSnapshotBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        raise RuntimeError("camera bus unavailable")


class RealtimeVisionBodyRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "stream_id": "front-main",
            "status": "tracking",
        }


class StaticVisionBodyRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> VisionObservation:
        return VisionObservation(ts=1.0, source="eye.compat", frame_id="still-1")


class VoiceHookBodyRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        payload = super().snapshot()
        payload["voice_dialogue"] = {"phase": "snapshot-idle"}
        payload["organs"] = {
            **payload["organs"],
            "mouth": {"health": "unavailable", "status": "snapshot-mute"},
        }
        return payload

    def voice_status(self) -> dict[str, object]:
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": "wired",
            "ear": {"status": "ok", "provider": "faster_whisper"},
            "mouth": {"status": "ok", "backend": "minimax", "model": "speech-2.8-hd"},
            "dialogue": {"phase": "speaking", "last_status": "completed"},
            "readiness_message": "native voice hook wired",
        }


class SnapshotVoiceBodyRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        payload = super().snapshot()
        payload["voice_dialogue"] = {
            "enabled": True,
            "running": True,
            "phase": "speaking",
            "last_status": "completed",
            "last_transcript": "你好 honjia",
            "last_reply": "你好",
            "last_stage_latency_ms": {"capture": 80.0, "llm": 420.0, "tts": 260.0},
            "last_bottleneck_stage": "llm",
            "last_bottleneck_ms": 420.0,
            "last_completed_turn": {"transcript": "你好 honjia", "reply": "你好"},
        }
        payload["organs"] = {
            **payload["organs"],
            "ear": {
                "organ": "ear",
                "health": "healthy",
                "subfunctions": {
                    "capture": {"health": "healthy", "details": {"device": "default"}},
                    "asr": {"health": "healthy", "details": {"provider": "faster_whisper"}},
                },
            },
            "mouth": {
                "organ": "mouth",
                "health": "healthy",
                "subfunctions": {
                    "tts_playback": {
                        "health": "healthy",
                        "details": {
                            "backend": "minimax",
                            "model": "speech-2.8-hd",
                            "voice_id": "female-shaonv",
                        },
                    }
                },
            },
        }
        return payload


def make_fake_head_runtime(config_path: str) -> HeadRuntimeApp:
    return HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path=config_path)


def wired_native_providers() -> dict[str, dict[str, str]]:
    return {
        "eye": {"status": "wired", "provider": "fake-eye"},
        "ear": {"status": "wired", "provider": "fake-ear"},
        "mouth": {"status": "wired", "provider": "fake-mouth"},
        "neck": {"status": "wired", "provider": "fake-neck"},
    }


def test_from_config_path_uses_native_runtime_for_eihead_config(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    body_config_path = tmp_path / "eibrain.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia-test",
                "legacy:",
                f"  body_runtime_config_path: {body_config_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    runtime = HeadRuntimeApp.from_config_path(str(config_path))

    assert runtime.config_path == str(config_path)
    assert runtime.body_runtime is None
    assert runtime.delegate_name == "eihead.native_runtime"


def test_from_config_path_ignores_legacy_body_runtime_factory() -> None:
    def fake_factory(path: str) -> FakeBodyRuntime:
        raise AssertionError(f"legacy factory should not be called: {path}")

    runtime = HeadRuntimeApp.from_config_path("config/eibrain.honjia.yaml", body_runtime_factory=fake_factory)

    assert runtime.body_runtime is None
    assert runtime.delegate_name == "eihead.native_runtime"


def test_legacy_body_adapter_is_removed() -> None:
    with pytest.raises(RuntimeError, match="LegacyBodyRuntimeAdapter has been removed"):
        LegacyBodyRuntimeAdapter()


def test_head_runtime_facade_does_not_embed_legacy_body_imports() -> None:
    app_source = Path("eihead/runtime/app.py").read_text(encoding="utf-8")
    cli_source = Path("eihead/runtime/cli.py").read_text(encoding="utf-8")

    assert "from apps.body_runtime" not in app_source
    assert "from apps.body_runtime" not in cli_source
    assert "from eibrain.protocol.actions" not in app_source
    assert "def _legacy_eibrain_action" not in app_source


def test_head_runtime_imports_and_wraps_body_snapshot() -> None:
    runtime = make_fake_head_runtime("config/test.yaml")

    snapshot = runtime.snapshot()

    assert AppsHeadRuntimeApp is HeadRuntimeApp
    assert snapshot["runtime"] == "eihead"
    assert snapshot["node_role"] == "head"
    assert snapshot["delegate"] == "eihead.native_runtime"
    assert snapshot["body_runtime"]["node_id"] == "honjia-test"
    assert snapshot["body_runtime"]["organ_count"] == 4


def test_snapshot_and_verify_report_degraded_when_native_provider_is_unknown() -> None:
    providers = wired_native_providers()
    providers["eye"] = {"status": "unknown", "reason": "probe_not_configured"}
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        delegate_name="eihead.native",
        native_providers=providers,
        neck_servo_adapter=object(),
    )

    snapshot = runtime.snapshot()
    verify_payload = runtime.verify()

    assert snapshot["ok"] is False
    assert snapshot["status"] == "degraded"
    assert snapshot["checks"]["native_provider_boundaries"] == "degraded"
    assert verify_payload["ok"] is False
    assert verify_payload["status"] == "degraded"
    assert verify_payload["checks"]["native_provider_boundaries"] == "degraded"


def test_verify_reports_ok_when_native_provider_boundaries_are_wired() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        native_providers=wired_native_providers(),
        neck_servo_adapter=object(),
    )

    verify_payload = runtime.verify()

    assert verify_payload["ok"] is True
    assert verify_payload["status"] == "ok"
    assert verify_payload["checks"]["body_runtime_delegate"] == "ok"
    assert verify_payload["check_details"]["body_runtime_delegate"]["delegate"] == "eihead.native_runtime"


def test_snapshot_and_verify_block_when_body_snapshot_raises() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FailingSnapshotBodyRuntime(),
        config_path="config/test.yaml",
        delegate_name="eihead.native",
        native_providers=wired_native_providers(),
        neck_servo_adapter=object(),
    )

    snapshot = runtime.snapshot()
    verify_payload = runtime.verify()

    assert snapshot["ok"] is False
    assert snapshot["status"] == "blocked"
    assert snapshot["checks"]["body_runtime_snapshot"] == "blocked"
    assert snapshot["body_runtime"]["reason"] == "body_runtime_snapshot_failed"
    assert snapshot["body_runtime"]["error"]["type"] == "RuntimeError"
    assert verify_payload["status"] == "blocked"
    assert verify_payload["checks"]["body_runtime_snapshot"] == "blocked"


def test_head_runtime_realtime_vision_hook_is_explicit_and_does_not_fake_static_frames() -> None:
    runtime = make_fake_head_runtime("config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_realtime_vision_hook_delegates_only_when_runtime_exposes_it() -> None:
    runtime = HeadRuntimeApp(body_runtime=RealtimeVisionBodyRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "stream_id": "front-main",
        "status": "tracking",
    }


def test_head_runtime_realtime_vision_hook_rejects_static_compat_observation() -> None:
    runtime = HeadRuntimeApp(body_runtime=StaticVisionBodyRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_voice_status_prefers_native_hook_over_snapshot_fallback() -> None:
    runtime = HeadRuntimeApp(body_runtime=VoiceHookBodyRuntime(), config_path="config/test.yaml")

    payload = runtime.voice_status()

    assert payload == {
        "schema": "eihead.monitor.voice_realtime.v1",
        "status": "wired",
        "ear": {"status": "ok", "provider": "faster_whisper"},
        "mouth": {"status": "ok", "backend": "minimax", "model": "speech-2.8-hd"},
        "dialogue": {"phase": "speaking", "last_status": "completed"},
        "readiness_message": "native voice hook wired",
    }


def test_head_runtime_voice_status_falls_back_to_snapshot_voice_dialogue_and_organs() -> None:
    runtime = HeadRuntimeApp(body_runtime=SnapshotVoiceBodyRuntime(), config_path="config/test.yaml")

    payload = runtime.voice_status()

    assert payload["voice_dialogue"]["phase"] == "speaking"
    assert payload["voice_dialogue"]["last_bottleneck_stage"] == "llm"
    assert payload["ear"]["subfunctions"]["asr"]["details"]["provider"] == "faster_whisper"
    assert payload["mouth"]["subfunctions"]["tts_playback"]["details"]["backend"] == "minimax"
    assert runtime.voice_realtime() == payload


def test_cli_status_uses_injected_runtime_without_hardware() -> None:
    stdout = io.StringIO()

    exit_code = cli.main(
        ["--config", "config/test.yaml", "status"],
        app_factory=make_fake_head_runtime,
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["command"] == "status"
    assert payload["runtime"] == "eihead"
    assert payload["config_path"] == "config/test.yaml"
    assert payload["body_runtime"]["node_id"] == "honjia-test"


def test_cli_serve_and_verify_dispatch_without_hardware() -> None:
    serve_payload = cli.dispatch(
        cli.build_parser().parse_args(["serve"]),
        app_factory=make_fake_head_runtime,
    )
    verify_payload = cli.dispatch(
        cli.build_parser().parse_args(["verify"]),
        app_factory=make_fake_head_runtime,
    )

    assert serve_payload["command"] == "serve"
    assert serve_payload["serve_mode"] == "compatibility_snapshot"
    assert verify_payload["command"] == "verify"
    assert verify_payload["checks"]["head_runtime_import"] == "ok"
    assert verify_payload["organ_count"] == 4


def test_verify_hardware_script_delegates_to_body_verifier(monkeypatch) -> None:
    called = {}

    def fake_verify() -> None:
        called["body_verify"] = True

    monkeypatch.setattr(cli, "_run_body_hardware_verifier", fake_verify)

    cli.verify_hardware_main()

    assert called == {"body_verify": True}


def test_pyproject_exposes_eihead_packages_and_scripts() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["project"]["scripts"]
    include = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]

    assert "eihead*" in include
    assert "apps" in include
    assert "apps.head_runtime*" in include
    assert scripts["eihead-runtime"] == "eihead.runtime.cli:main"
    assert scripts["eihead-verify-hardware"] == "eihead.runtime.cli:verify_hardware_main"
