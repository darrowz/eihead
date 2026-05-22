from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_voice_systemd_units_include_honjia_native_asr_pythonpath() -> None:
    for unit_name in ("eihead-runtime.service", "eihead-monitor.service"):
        text = _read(f"deploy/systemd/{unit_name}")

        assert "Environment=PYTHONPATH=" in text
        assert "/dev-project/eiprotocol" in text
        assert "/usr/lib/python3/dist-packages" in text
        assert "/home/darrow/.local/lib/python3.13/site-packages" in text


def test_release_installer_installs_runtime_and_monitor_units() -> None:
    text = _read("deploy/install_immutable_release.sh")

    assert "eihead-runtime.service" in text
    assert "eihead-monitor.service" in text
    assert 'sudo systemctl restart "$unit"' in text


def test_monitor_unit_proxies_voice_from_runtime_without_starting_voice_loop() -> None:
    text = _read("deploy/systemd/eihead-monitor.service")

    assert "Environment=EIHEAD_RUNTIME_URL=http://127.0.0.1:18081" in text
    assert "Environment=EIHEAD_NATIVE_VOICE_RUNTIME_DISABLED=1" in text
    assert "Environment=EIHEAD_MONITOR_PROXY_RUNTIME_VOICE=1" in text


def test_runtime_unit_selects_openclaw_realtime_with_explicit_eibrain_fallback() -> None:
    text = _read("deploy/systemd/eihead-runtime.service")

    assert "Environment=EIHEAD_VOICE_TRANSPORT_PROVIDER=openclaw_realtime" in text
    assert "Environment=EIHEAD_VOICE_FALLBACK_PROVIDER=eibrain_subprocess" in text
