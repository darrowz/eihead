from __future__ import annotations

from pathlib import Path
from typing import Any

from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime.native_providers import build_native_provider_statuses


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-native-test", "organ_count": 4}


class FakeNativeProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        self.calls.append(provider_name)
        if provider_name == "eye":
            return {
                "status": "wired",
                "provider": environ["EIHEAD_TEST_EYE_PROVIDER"],
                "reason": "probe_reported_wired",
            }
        if provider_name == "ear":
            return {"status": "unknown", "reason": f"config_node={config.node_id}"}
        if provider_name == "mouth":
            return {"status": "unavailable", "reason": "tts_backend_disabled"}
        raise AssertionError(f"unexpected hardware probe for {provider_name}")


class FakeNeckAdapter:
    def apply_plan(self, plan: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "plan_status": plan.get("status")}


class FakeEyeService:
    def status(self) -> dict[str, object]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "status": "tracking",
            "mode": "realtime_stream",
            "provider": "native-eye-service",
            "backend": "gstreamer_hailo",
            "camera_device": "/dev/video42",
            "hailo_device": "/dev/hailo0",
            "stream_ready": True,
            "not_wired": False,
            "readiness_message": "camera and hailo ready",
        }


def test_from_config_path_reports_native_provider_boundaries_without_hardware(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    body_config_path = tmp_path / "eibrain.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia-native-test",
                "legacy:",
                f"  body_runtime_config_path: {body_config_path.as_posix()}",
                "native_providers:",
                "  eye:",
                "    enabled: true",
                "  ear:",
                "    enabled: true",
                "  mouth:",
                "    enabled: false",
                "  neck:",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    def fake_factory(path: str) -> FakeBodyRuntime:
        raise AssertionError(f"legacy factory should not be called: {path}")

    probe = FakeNativeProbe()

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        body_runtime_factory=fake_factory,
        native_provider_probe=probe,
        native_environ={"EIHEAD_TEST_EYE_PROVIDER": "fake-eye-adapter"},
    )

    snapshot = runtime.snapshot()
    native_providers = snapshot["native_providers"]

    assert snapshot["body_runtime"] == {}
    assert native_providers["eye"] == {
        "status": "wired",
        "provider": "fake-eye-adapter",
        "reason": "probe_reported_wired",
    }
    assert native_providers["ear"]["status"] == "unknown"
    assert native_providers["ear"]["reason"] == "config_node=honjia-native-test"
    assert native_providers["mouth"]["status"] == "unavailable"
    assert native_providers["mouth"]["reason"] == "tts_backend_disabled"
    assert native_providers["neck"]["status"] == "unavailable"
    assert native_providers["neck"]["reason"] == "neck_servo_adapter_missing"
    assert "neck" not in probe.calls


def test_injected_neck_adapter_can_be_reported_as_wired_by_probe(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text("node_id: honjia-native-test\n", encoding="utf-8")

    def fake_factory(path: str) -> FakeBodyRuntime:
        raise AssertionError(f"legacy factory should not be called: {path}")

    def probe(provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        if provider_name == "neck":
            return {"status": "wired", "provider": "fake-neck-servo", "reason": "adapter_injected"}
        return {"status": "unknown", "reason": "probe_not_configured"}

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        body_runtime_factory=fake_factory,
        native_provider_probe=probe,
        neck_servo_adapter=FakeNeckAdapter(),
    )

    assert runtime.snapshot()["native_providers"]["neck"] == {
        "status": "wired",
        "provider": "fake-neck-servo",
        "reason": "adapter_injected",
    }


def test_env_can_report_neck_wired_before_adapter_injection() -> None:
    statuses = build_native_provider_statuses(
        config=None,
        environ={
            "EIHEAD_NATIVE_NECK_STATUS": "wired",
            "EIHEAD_NATIVE_NECK_PROVIDER": "raspbot-i2c",
            "EIHEAD_NATIVE_NECK_REASON": "verified_i2c_bus",
        },
    )

    assert statuses["neck"] == {
        "status": "wired",
        "provider": "raspbot-i2c",
        "reason": "verified_i2c_bus",
    }


def test_normalize_keeps_explicit_neck_status_without_adapter() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        native_providers={
            "neck": {
                "status": "wired",
                "provider": "raspbot-i2c",
                "reason": "verified_i2c_bus",
            }
        },
    )

    assert runtime.status()["native_providers"]["neck"] == {
        "status": "wired",
        "provider": "raspbot-i2c",
        "reason": "verified_i2c_bus",
    }


def test_direct_runtime_construction_marks_uninjected_neck_unavailable() -> None:
    runtime = HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path="config/test.yaml")

    native_providers = runtime.status()["native_providers"]

    assert native_providers["eye"]["status"] == "unknown"
    assert native_providers["ear"]["status"] == "unknown"
    assert native_providers["mouth"]["status"] == "unknown"
    assert native_providers["neck"] == {
        "status": "unavailable",
        "reason": "neck_servo_adapter_missing",
    }


def test_native_provider_probe_can_report_degraded_with_truthful_metadata() -> None:
    def probe(provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        if provider_name == "eye":
            return {
                "status": "degraded",
                "provider": "fake-eye-adapter",
                "source": "fake-native-live-probe",
                "checked_at": 5678.5,
                "reason": "low_frame_rate",
                "hardware_verified": True,
                "details": {"fps": 2},
            }
        return {"status": "unknown", "source": "fake-native-live-probe", "reason": "not_checked"}

    statuses = build_native_provider_statuses(
        config=None,
        environ={},
        probe=probe,
        neck_servo_adapter=FakeNeckAdapter(),
    )

    assert statuses["eye"] == {
        "status": "degraded",
        "provider": "fake-eye-adapter",
        "reason": "low_frame_rate",
        "source": "fake-native-live-probe",
        "checked_at": 5678.5,
        "last_checked": 5678.5,
        "hardware_verified": True,
        "details": {"fps": 2},
    }


def test_native_eye_service_status_feeds_status_and_capability_readiness() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        native_providers={
            "eye": FakeEyeService(),
            "ear": {"status": "unknown"},
            "mouth": {"status": "unknown"},
            "neck": {"status": "wired"},
        },
        neck_servo_adapter=FakeNeckAdapter(),
    )

    status_payload = runtime.status()
    eye_status = status_payload["native_providers"]["eye"]

    assert eye_status["status"] == "wired"
    assert eye_status["provider"] == "native-eye-service"
    assert eye_status["details"]["backend"] == "gstreamer_hailo"
    assert eye_status["details"]["camera_device"] == "/dev/video42"
    assert eye_status["details"]["hailo_device"] == "/dev/hailo0"
    assert eye_status["details"]["stream_ready"] is True
    assert eye_status["details"]["not_wired"] is False
    assert eye_status["details"]["readiness_message"] == "camera and hailo ready"

    capabilities = runtime.capabilities()["capabilities"]
    assert capabilities["camera"]["details"]["native_camera_device"] == "/dev/video42"
    assert capabilities["camera"]["details"]["native_stream_ready"] is True
    assert capabilities["hailo"]["details"]["native_hailo_device"] == "/dev/hailo0"
    assert capabilities["hailo"]["details"]["native_stream_ready"] is True
    assert capabilities["vision_backend"]["details"]["native_backend"] == "gstreamer_hailo"
    assert capabilities["vision_backend"]["details"]["native_hailo_device"] == "/dev/hailo0"
