from eihead.devices.audio import (
    AcousticFrontendReadiness,
    AudioDeviceCandidate,
    AudioRoutePlan,
    PlaybackInterruptionPlan,
    build_aplay_command,
    build_loopback_readiness,
    build_playback_stop_plan,
    build_arecord_command,
    choose_audio_routes,
    evaluate_audio_frontend_readiness,
    parse_aplay_devices,
    parse_arecord_devices,
    parse_pactl_sources,
    select_preferred_input,
)


def test_select_preferred_input_prefers_u4k_over_spa3700() -> None:
    spa3700 = AudioDeviceCandidate(
        name="SPA3700 USB Audio",
        kind="input",
        device="hw:1,0",
        score=100,
        reason="appears first in ALSA list",
    )
    u4k = AudioDeviceCandidate(
        name="U4K capture",
        kind="input",
        device="hw:2,0",
        score=10,
        reason="confirmed usable field microphone",
    )

    selected = select_preferred_input([spa3700, u4k])

    assert selected.name == u4k.name
    assert selected.device == u4k.device
    assert selected.metadata["preferred_keyword"] == "U4K"
    assert selected.score > spa3700.score


def test_audio_device_helpers_are_exported_from_devices_package() -> None:
    from eihead.devices import AudioDeviceCandidate, select_preferred_input

    selected = select_preferred_input(
        [
            AudioDeviceCandidate(
                name="U4K capture",
                kind="input",
                device="hw:2,0",
                score=10,
                reason="confirmed usable field microphone",
            )
        ]
    )

    assert selected.device == "hw:2,0"


def test_select_preferred_input_degrades_spa3700_when_it_is_only_input() -> None:
    spa3700 = AudioDeviceCandidate(
        name="SPA3700 USB Audio",
        kind="input",
        device="hw:1,0",
        score=80,
        reason="listed input",
    )

    selected = select_preferred_input([spa3700])

    assert selected.name == "SPA3700 USB Audio"
    assert selected.score < 0
    assert "not confirmed usable" in selected.reason
    assert selected.metadata["degraded"] is True


def test_select_preferred_input_without_u4k_chooses_best_non_spa3700() -> None:
    candidates = [
        AudioDeviceCandidate("USB PnP Sound Device", "input", "hw:3,0", 25, "generic mic"),
        AudioDeviceCandidate("SPA3700 USB Audio", "input", "hw:1,0", 100, "listed input"),
        AudioDeviceCandidate("HD Webcam Microphone", "input", "hw:4,0", 40, "fallback mic"),
    ]

    selected = select_preferred_input(candidates)

    assert selected.name == "HD Webcam Microphone"
    assert "SPA3700" not in selected.name


def test_arecord_and_aplay_commands_are_16k_mono_pcm_without_execution() -> None:
    arecord = build_arecord_command("hw:2,0")
    aplay = build_aplay_command("hw:5,0")

    assert arecord == [
        "arecord",
        "-D",
        "hw:2,0",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "--period-time",
        "60000",
    ]
    assert aplay == [
        "aplay",
        "-D",
        "hw:5,0",
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
    ]


def test_audio_frontend_readiness_degraded_without_loopback_or_aec() -> None:
    readiness = evaluate_audio_frontend_readiness(
        capture_device="hw:2,0",
        loopback_device=None,
        supports_aec=False,
        supports_ns=True,
        supports_vad=True,
    )

    assert isinstance(readiness, AcousticFrontendReadiness)
    assert readiness.capture is True
    assert readiness.healthy is False
    assert "loopback unavailable; speaker echo reference is degraded" in readiness.warnings
    assert "AEC unavailable; echo cancellation is degraded" in readiness.warnings
    assert readiness.to_dict()["healthy"] is False


def test_audio_frontend_readiness_capture_is_required() -> None:
    readiness = evaluate_audio_frontend_readiness(capture_device="")

    assert readiness.capture is False
    assert readiness.healthy is False
    assert "capture unavailable; microphone input is blocked" in readiness.warnings


def test_playback_interruption_plan_defaults_to_300ms_stop_command() -> None:
    plan = PlaybackInterruptionPlan(stop_command=["systemctl", "stop", "eivoice-playback"])

    assert plan.expected_max_ms == 300
    assert plan.reason == "barge-in requires playback stop before capture continues"
    assert plan.to_dict() == {
        "stop_command": ["systemctl", "stop", "eivoice-playback"],
        "reason": "barge-in requires playback stop before capture continues",
        "expected_max_ms": 300,
    }


def test_parse_arecord_devices_returns_input_candidates_from_text_only() -> None:
    text = """
**** List of CAPTURE Hardware Devices ****
card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: U4K [U4K Microphone], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""

    candidates = parse_arecord_devices(text)

    assert [candidate.device for candidate in candidates] == ["hw:1,0", "hw:2,0"]
    assert all(candidate.kind == "input" for candidate in candidates)
    assert candidates[1].name == "U4K Microphone USB Audio"
    assert candidates[1].metadata["card_name"] == "U4K"
    assert candidates[1].metadata["parser"] == "arecord"


def test_parse_aplay_devices_returns_output_candidates_from_text_only() -> None:
    text = """
**** List of PLAYBACK Hardware Devices ****
card 0: PCH [HDA Intel PCH], device 0: ALC887-VD Analog [ALC887-VD Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 3: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""

    candidates = parse_aplay_devices(text)

    assert [candidate.device for candidate in candidates] == ["hw:0,0", "hw:3,0"]
    assert all(candidate.kind == "output" for candidate in candidates)
    assert candidates[0].metadata["parser"] == "aplay"


def test_parse_pactl_sources_extracts_monitor_and_source_candidates() -> None:
    text = """
Source #52
    Name: alsa_input.usb-GeneralPlus_USB_Audio_Device-00.analog-mono
    Description: U4K USB Audio Device Mono Input
    State: RUNNING
    Monitor of Sink: n/a
Source #87
    Name: alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor
    Description: Monitor of SPA3700 Analog Stereo
    State: IDLE
    Monitor of Sink: alsa_output.usb-C-Media_SPA3700-00.analog-stereo
"""

    candidates = parse_pactl_sources(text)

    assert [candidate.kind for candidate in candidates] == ["input", "loopback"]
    assert candidates[0].device == "alsa_input.usb-GeneralPlus_USB_Audio_Device-00.analog-mono"
    assert candidates[1].device == "alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor"
    assert candidates[1].metadata["monitor_of_sink"] == "alsa_output.usb-C-Media_SPA3700-00.analog-stereo"


def test_choose_audio_routes_prefers_u4k_and_marks_missing_optional_routes_in_warnings() -> None:
    inputs = parse_arecord_devices(
        """
card 1: Device [SPA3700], device 0: USB Audio [USB Audio]
card 2: U4K [U4K Microphone], device 0: USB Audio [USB Audio]
"""
    )
    outputs = parse_aplay_devices("card 4: Speaker [USB Speaker], device 0: USB Audio [USB Audio]")

    plan = choose_audio_routes(inputs=inputs, outputs=outputs, loopbacks=[])

    assert isinstance(plan, AudioRoutePlan)
    assert plan.capture.device == "hw:2,0"
    assert plan.playback.device == "hw:4,0"
    assert plan.loopback is None
    assert plan.readiness.capture is True
    assert plan.readiness.loopback is False
    assert "loopback candidate unavailable; echo reference will be optional" in plan.status["warnings"]


def test_choose_audio_routes_degrades_to_spa3700_when_u4k_missing() -> None:
    inputs = parse_arecord_devices("card 1: Device [SPA3700], device 0: USB Audio [USB Audio]")

    plan = choose_audio_routes(inputs=inputs, outputs=[], loopbacks=[])

    assert plan.capture is not None
    assert plan.capture.device == "hw:1,0"
    assert plan.capture.metadata["degraded"] is True
    assert "SPA3700 input not confirmed usable" in plan.capture.reason
    assert "using degraded SPA3700 input fallback" in plan.status["warnings"]
    assert plan.playback is None


def test_build_playback_stop_plan_uses_route_status_for_reason() -> None:
    route_plan = choose_audio_routes(
        inputs=parse_arecord_devices("card 2: U4K [U4K Microphone], device 0: USB Audio [USB Audio]"),
        outputs=parse_aplay_devices("card 4: Speaker [USB Speaker], device 0: USB Audio [USB Audio]"),
        loopbacks=parse_pactl_sources(
            """
Source #87
    Name: alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor
    Description: Monitor of SPA3700 Analog Stereo
    Monitor of Sink: alsa_output.usb-C-Media_SPA3700-00.analog-stereo
"""
        ),
    )

    plan = build_playback_stop_plan(route_plan, service_name="eivoice-playback")

    assert plan.stop_command == ["systemctl", "stop", "eivoice-playback"]
    assert "capture=hw:2,0" in plan.reason
    assert "loopback=alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor" in plan.reason


def test_choose_audio_routes_does_not_mark_frontend_healthy_from_discovery_only() -> None:
    plan = choose_audio_routes(
        inputs=parse_arecord_devices("card 2: U4K [U4K Microphone], device 0: USB Audio [USB Audio]"),
        outputs=parse_aplay_devices("card 4: Speaker [USB Speaker], device 0: USB Audio [USB Audio]"),
        loopbacks=parse_pactl_sources(
            """
Source #87
    Name: alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor
    Description: Monitor of SPA3700 Analog Stereo
    Monitor of Sink: alsa_output.usb-C-Media_SPA3700-00.analog-stereo
"""
        ),
    )

    assert plan.readiness.capture is True
    assert plan.readiness.loopback is True
    assert plan.readiness.healthy is False
    assert "AEC unavailable; echo cancellation is degraded" in plan.status["warnings"]
    assert "NS unavailable; noise suppression is degraded" in plan.status["warnings"]
    assert "VAD unavailable; endpointing is degraded" in plan.status["warnings"]


def test_choose_audio_routes_accepts_verified_frontend_capabilities() -> None:
    plan = choose_audio_routes(
        inputs=parse_arecord_devices("card 2: U4K [U4K Microphone], device 0: USB Audio [USB Audio]"),
        outputs=parse_aplay_devices("card 4: Speaker [USB Speaker], device 0: USB Audio [USB Audio]"),
        loopbacks=parse_pactl_sources(
            """
Source #87
    Name: alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor
    Description: Monitor of SPA3700 Analog Stereo
    Monitor of Sink: alsa_output.usb-C-Media_SPA3700-00.analog-stereo
"""
        ),
        supports_aec=True,
        supports_ns=True,
        supports_vad=True,
    )

    assert plan.readiness.healthy is True
    assert plan.status["warnings"] == []


def test_build_loopback_readiness_marks_optional_loopback_warning() -> None:
    readiness = build_loopback_readiness(capture_device="hw:2,0", loopback_device=None)

    assert readiness.capture is True
    assert readiness.loopback is False
    assert "loopback unavailable; speaker echo reference is degraded" in readiness.warnings


def test_build_loopback_readiness_requires_explicit_vad_verification_for_healthy() -> None:
    readiness = build_loopback_readiness(
        capture_device="hw:2,0",
        loopback_device="alsa_output.usb-C-Media_SPA3700-00.analog-stereo.monitor",
        supports_aec=True,
        supports_ns=True,
    )

    assert readiness.aec is True
    assert readiness.ns is True
    assert readiness.vad is False
    assert readiness.healthy is False
    assert "VAD unavailable; endpointing is degraded" in readiness.warnings


def test_audio_device_discovery_helpers_are_exported_from_devices_package() -> None:
    from eihead.devices import (
        AudioRoutePlan,
        build_loopback_readiness,
        build_playback_stop_plan,
        choose_audio_routes,
        parse_arecord_devices,
    )

    inputs = parse_arecord_devices("card 2: U4K [U4K Microphone], device 0: USB Audio [USB Audio]")
    plan = choose_audio_routes(inputs=inputs, outputs=[], loopbacks=[])

    assert isinstance(plan, AudioRoutePlan)
    assert build_loopback_readiness(capture_device="hw:2,0", loopback_device=None).loopback is False
    assert build_playback_stop_plan(plan).stop_command == ["systemctl", "stop", "eivoice-playback"]
