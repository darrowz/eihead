from __future__ import annotations

from pathlib import Path

import pytest

from eihead.runtime import config as eihead_config
from eihead.runtime.config import EiheadConfigError, load_eihead_config, parse_eihead_config


def test_template_loads_runtime_monitor_devices_and_legacy_paths() -> None:
    config_path = Path(__file__).resolve().parents[2] / "config" / "eihead.honjia.yaml"

    config = load_eihead_config(config_path)

    assert config.node_id == "honjia"
    assert config.runtime.host == "0.0.0.0"
    assert config.runtime.port == 18081
    assert config.monitor.host == "0.0.0.0"
    assert config.monitor.port == 18080
    assert config.devices.camera == "/dev/video0"
    assert config.devices.hailo == "/dev/hailo0"
    assert config.devices.i2c == "/dev/i2c-1"
    assert config.devices.neck == "/dev/i2c-1"
    assert config.legacy.eibrain_config_path == "config/eibrain.honjia.yaml"
    assert config.legacy.body_runtime_config_path == "config/eibrain.honjia.yaml"


def test_load_eihead_config_expands_env_and_parses_ports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EIHEAD_TEST_HOST", "0.0.0.0")
    monkeypatch.setenv("EIHEAD_TEST_PORT", "19081")
    config_path = tmp_path / "eihead.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  host: ${EIHEAD_TEST_HOST:-127.0.0.1}",
                "  port: ${EIHEAD_TEST_PORT:-18081}",
                "monitoring:",
                "  port: ${EIHEAD_MONITOR_TEST_PORT:-18080}",
                "devices:",
                "  camera: /dev/video9",
                "  i2c:",
                "    bus: 7",
                "  microphone:",
                "    path: /dev/snd-test",
            ]
        ),
        encoding="utf-8",
    )

    config = load_eihead_config(config_path)

    assert config.runtime.host == "0.0.0.0"
    assert config.runtime.port == 19081
    assert config.monitor.port == 18080
    assert config.devices.camera == "/dev/video9"
    assert config.devices.i2c == "/dev/i2c-7"
    assert config.devices.microphone == "/dev/snd-test"


def test_parse_eihead_config_reads_software_capability_declarations() -> None:
    config = parse_eihead_config(
        {
            "capabilities": {
                "software": {
                    "asr": {
                        "enabled": True,
                        "provider": "sherpa_onnx",
                        "model": "streaming",
                        "limits": {"streaming": True, "languages": ["zh"]},
                    },
                    "tts": {
                        "enabled": True,
                        "provider": "minimax",
                        "status": "degraded",
                    },
                },
                "vision_backend": {
                    "enabled": True,
                    "backend": "hailo",
                    "model": "personface",
                },
            }
        }
    )

    asr = config.capabilities.software["asr"]
    tts = config.capabilities.software["tts"]
    vision = config.capabilities.software["vision_backend"]

    assert asr.provider == "sherpa_onnx"
    assert asr.limits["streaming"] is True
    assert tts.to_dict()["status"] == "degraded"
    assert vision.backend == "hailo"


def test_config_can_emit_capability_registry_config_without_touching_hardware() -> None:
    config = parse_eihead_config(
        {
            "node_id": "honjia-test",
            "devices": {
                "camera": {"path": "/dev/video-test"},
                "hailo": {"path": "/dev/hailo-test"},
                "i2c": {"path": "/dev/i2c-3"},
                "neck": {"path": "/dev/i2c-3"},
            },
            "capabilities": {
                "software": {
                    "asr": {"enabled": True, "provider": "sherpa_onnx"},
                    "vision_backend": {"enabled": True, "backend": "hailo"},
                }
            },
        }
    )

    registry_config = config.as_capability_registry_config()

    assert registry_config["node_id"] == "honjia-test"
    assert registry_config["capabilities"]["camera"]["path"] == "/dev/video-test"
    assert registry_config["capabilities"]["hailo"]["path"] == "/dev/hailo-test"
    assert registry_config["capabilities"]["i2c"]["limits"]["bus"] == 3
    assert registry_config["capabilities"]["neck"]["limits"]["tilt_deg"] is None
    assert registry_config["capabilities"]["asr"]["provider"] == "sherpa_onnx"
    assert registry_config["capabilities"]["vision_backend"]["backend"] == "hailo"


def test_missing_pyyaml_reports_clear_error(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "eihead.yaml"
    config_path.write_text("node_id: honjia\n", encoding="utf-8")
    monkeypatch.setattr(eihead_config, "yaml", None)
    monkeypatch.setattr(eihead_config, "_YAML_IMPORT_ERROR", ImportError("missing yaml"))

    with pytest.raises(EiheadConfigError, match="PyYAML is required"):
        eihead_config.load_eihead_config(config_path)
