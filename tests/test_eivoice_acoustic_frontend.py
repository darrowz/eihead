from __future__ import annotations

import pytest

from eihead.eivoice_runtime import AudioFrame, AcousticFrontendConfig, NoOpAcousticFrontend
from eihead.eivoice_runtime.aec import (
    LoopbackReferenceBuffer,
    LoopbackReferenceMatch,
    build_loopback_reference_diagnostics,
)


def _frame(
    sequence: int,
    *,
    pcm: bytes | None = None,
    created_at_ts: float = 1_000.0,
    duration_ms: int = 20,
) -> AudioFrame:
    return AudioFrame(
        pcm=pcm if pcm is not None else f"pcm-{sequence}".encode("ascii"),
        duration_ms=duration_ms,
        sample_rate_hz=16_000,
        channels=1,
        sequence=sequence,
        created_at_ts=created_at_ts,
    )


def test_loopback_reference_buffer_matches_capture_by_sequence_and_age() -> None:
    buffer = LoopbackReferenceBuffer(sample_rate=16_000, frame_ms=20, max_age_ms=120)
    buffer.write_playback(b"speaker-40", sequence=40, created_at_ts=10.000)
    buffer.write_playback(_frame(41, pcm=b"speaker-41", created_at_ts=10.020))

    match = buffer.reference_for_capture(_frame(41, pcm=b"mic-41", created_at_ts=10.050))

    assert match is not None
    assert match.frame.pcm == b"speaker-41"
    assert match.frame.sequence == 41
    assert match.matched_by == "sequence"
    assert match.age_ms == pytest.approx(30.0)


def test_loopback_reference_buffer_can_fall_back_to_capture_time() -> None:
    buffer = LoopbackReferenceBuffer(sample_rate=16_000, frame_ms=20, max_age_ms=120)
    buffer.write_playback(_frame(10, pcm=b"older", created_at_ts=20.000))
    buffer.write_playback(_frame(11, pcm=b"nearest", created_at_ts=20.040))

    match = buffer.reference_for_capture(_frame(99, pcm=b"mic", created_at_ts=20.060))

    assert match is not None
    assert match.frame.pcm == b"nearest"
    assert match.matched_by == "time"
    assert match.age_ms == pytest.approx(20.0)


def test_acoustic_frontend_config_reports_route_and_frame_diagnostics() -> None:
    config = AcousticFrontendConfig(
        capture_device="hw:2,0",
        playback_device="hw:4,0",
        loopback_device="alsa_output.usb-SPA3700.monitor",
        sample_rate=16_000,
        frame_ms=20,
        channels=1,
        aec_enabled=True,
        aec_available=False,
        loopback_enabled=True,
        loopback_available=True,
        mode="passthrough",
    )

    diagnostics = config.diagnostics()

    assert diagnostics["devices"] == {
        "capture": "hw:2,0",
        "playback": "hw:4,0",
        "loopback": "alsa_output.usb-SPA3700.monitor",
    }
    assert diagnostics["audio_format"] == {
        "sample_rate": 16_000,
        "frame_ms": 20,
        "channels": 1,
    }
    assert diagnostics["aec_status"] == "unavailable"
    assert diagnostics["loopback"]["state"] == "ready"


def test_noop_frontend_does_not_claim_aec_when_webrtc_is_unavailable() -> None:
    frontend = NoOpAcousticFrontend(
        AcousticFrontendConfig(
            aec_enabled=True,
            aec_available=False,
            aec_backend="webrtc",
            loopback_enabled=True,
            loopback_available=True,
        )
    )
    capture = _frame(7, pcm=b"mic-with-echo", created_at_ts=30.060)
    reference = _frame(7, pcm=b"speaker-reference", created_at_ts=30.000)

    processed = frontend.process_capture(capture, playback_reference=reference)

    assert processed.frame.pcm == b"mic-with-echo"
    assert processed.diagnostics["aec_backend"] == "webrtc"
    assert processed.diagnostics["aec_status"] == "unavailable"
    assert processed.diagnostics["aec_applied"] is False
    assert processed.diagnostics["fallback_reason"] == "aec_unavailable"
    assert processed.diagnostics["reference_age_ms"] == pytest.approx(60.0)
    assert frontend.readiness()["last_capture"]["fallback_reason"] == "aec_unavailable"


def test_loopback_reference_diagnostics_reports_ready_reference() -> None:
    config = AcousticFrontendConfig(
        aec_enabled=True,
        aec_available=True,
        loopback_enabled=True,
        loopback_available=True,
        loopback_device="alsa_output.usb-speaker.monitor",
    )
    match = LoopbackReferenceMatch(
        frame=_frame(17, created_at_ts=40.000),
        age_ms=45.0,
        matched_by="sequence",
    )

    diagnostics = build_loopback_reference_diagnostics(config, match, max_age_ms=120)

    assert diagnostics == {
        "ready": True,
        "state": "ready",
        "reason": "loopback_reference_ready",
        "reference_age_ms": pytest.approx(45.0),
        "matched_by": "sequence",
        "max_age_ms": 120,
        "device": "alsa_output.usb-speaker.monitor",
        "device_ready": True,
        "device_state": "ready",
        "dry_run": True,
        "aec_status": "passthrough",
    }


def test_loopback_reference_diagnostics_reports_missing_reference() -> None:
    config = AcousticFrontendConfig(
        aec_enabled=True,
        aec_available=True,
        loopback_enabled=True,
        loopback_available=True,
        loopback_device="alsa_output.usb-speaker.monitor",
    )

    diagnostics = build_loopback_reference_diagnostics(config, None, max_age_ms=120)

    assert diagnostics["ready"] is False
    assert diagnostics["state"] == "missing"
    assert diagnostics["reason"] == "missing_playback_reference"
    assert diagnostics["reference_age_ms"] is None
    assert diagnostics["matched_by"] is None
    assert diagnostics["max_age_ms"] == 120
    assert diagnostics["device"] == "alsa_output.usb-speaker.monitor"
    assert diagnostics["aec_status"] == "passthrough"


def test_loopback_reference_diagnostics_reports_stale_reference() -> None:
    config = AcousticFrontendConfig(
        aec_enabled=True,
        aec_available=True,
        loopback_enabled=True,
        loopback_available=True,
        loopback_device="alsa_output.usb-speaker.monitor",
    )
    match = LoopbackReferenceMatch(
        frame=_frame(18, created_at_ts=41.000),
        age_ms=250.0,
        matched_by="explicit",
    )

    diagnostics = build_loopback_reference_diagnostics(config, match, max_age_ms=120)

    assert diagnostics["ready"] is False
    assert diagnostics["state"] == "stale"
    assert diagnostics["reason"] == "stale_playback_reference"
    assert diagnostics["reference_age_ms"] == pytest.approx(250.0)
    assert diagnostics["matched_by"] == "explicit"


def test_loopback_reference_diagnostics_reports_aec_unavailable() -> None:
    config = AcousticFrontendConfig(
        aec_enabled=True,
        aec_available=False,
        aec_backend="webrtc",
        loopback_enabled=True,
        loopback_available=True,
        loopback_device="alsa_output.usb-speaker.monitor",
    )
    match = LoopbackReferenceMatch(
        frame=_frame(19, created_at_ts=42.000),
        age_ms=30.0,
        matched_by="sequence",
    )

    diagnostics = build_loopback_reference_diagnostics(config, match, max_age_ms=120)

    assert diagnostics["ready"] is False
    assert diagnostics["state"] == "aec_unavailable"
    assert diagnostics["reason"] == "aec_unavailable"
    assert diagnostics["reference_age_ms"] == pytest.approx(30.0)
    assert diagnostics["matched_by"] == "sequence"
    assert diagnostics["aec_status"] == "unavailable"


def test_loopback_reference_diagnostics_report_dry_run_and_missing_device() -> None:
    config = AcousticFrontendConfig(
        aec_enabled=True,
        aec_available=True,
        loopback_enabled=True,
        loopback_available=True,
        mode="noop",
        loopback_device=None,
    )
    match = LoopbackReferenceMatch(
        frame=_frame(21, created_at_ts=43.000),
        age_ms=24.0,
        matched_by="time",
    )

    diagnostics = build_loopback_reference_diagnostics(config, match, max_age_ms=120)

    assert diagnostics["ready"] is True
    assert diagnostics["state"] == "ready"
    assert diagnostics["device"] is None
    assert diagnostics["device_ready"] is False
    assert diagnostics["device_state"] == "missing"
    assert diagnostics["dry_run"] is True


def test_noop_frontend_last_capture_reuses_loopback_reference_diagnostics() -> None:
    frontend = NoOpAcousticFrontend(
        AcousticFrontendConfig(
            aec_enabled=True,
            aec_available=True,
            loopback_enabled=True,
            loopback_available=True,
            loopback_device="alsa_output.usb-speaker.monitor",
        )
    )

    processed = frontend.process_capture(
        _frame(20, pcm=b"mic", created_at_ts=50.200),
        playback_reference=_frame(20, pcm=b"speaker", created_at_ts=50.000),
    )

    assert processed.diagnostics["loopback_reference"]["ready"] is True
    assert processed.diagnostics["loopback_reference_state"] == "ready"
    assert processed.diagnostics["loopback_reference_reason"] == "loopback_reference_ready"
    assert processed.diagnostics["loopback_reference"]["dry_run"] is True
    assert frontend.readiness()["last_capture"]["loopback_reference"]["matched_by"] == "explicit"
