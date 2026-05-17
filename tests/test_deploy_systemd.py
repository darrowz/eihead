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
