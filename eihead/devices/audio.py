from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Sequence


AudioDeviceKind = Literal["input", "output", "loopback"]


@dataclass(frozen=True)
class AudioDeviceCandidate:
    name: str
    kind: AudioDeviceKind
    device: str
    score: int
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def select_preferred_input(
    candidates: Iterable[AudioDeviceCandidate],
    preferred_keywords: Sequence[str] = ("U4K",),
) -> AudioDeviceCandidate:
    input_candidates = [candidate for candidate in candidates if candidate.kind == "input"]
    if not input_candidates:
        raise ValueError("no input audio candidates available")

    ranked = [_rank_input_candidate(candidate, preferred_keywords) for candidate in input_candidates]
    return max(ranked, key=lambda candidate: candidate.score)


def _rank_input_candidate(
    candidate: AudioDeviceCandidate,
    preferred_keywords: Sequence[str],
) -> AudioDeviceCandidate:
    name_upper = candidate.name.upper()
    score = candidate.score
    reason_parts = [candidate.reason]
    metadata = dict(candidate.metadata)

    for keyword in preferred_keywords:
        if keyword.upper() in name_upper:
            score += 10_000
            reason_parts.append(f"preferred field microphone keyword matched: {keyword}")
            metadata["preferred_keyword"] = keyword
            break

    if "SPA3700" in name_upper:
        score -= 1_000
        reason_parts.append("SPA3700 input not confirmed usable")
        metadata["degraded"] = True
        metadata["deprioritized"] = "SPA3700 input not confirmed usable"

    return replace(
        candidate,
        score=score,
        reason="; ".join(part for part in reason_parts if part),
        metadata=metadata,
    )


def build_arecord_command(
    device: str,
    sample_rate: int = 16_000,
    channels: int = 1,
    frame_ms: int = 60,
) -> list[str]:
    return [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "--period-time",
        str(frame_ms * 1_000),
    ]


def build_aplay_command(
    device: str,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> list[str]:
    return [
        "aplay",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
    ]


_ALSA_DEVICE_LINE_RE = re.compile(
    r"^card\s+(?P<card_index>\d+):\s+(?P<card_id>[^\[]+?)\s+\[(?P<card_name>[^\]]+)\],\s+"
    r"device\s+(?P<device_index>\d+):\s+(?P<device_id>[^\[]+?)\s+\[(?P<device_name>[^\]]+)\]$"
)
_PACTL_SOURCE_LINE_RE = re.compile(r"^Source\s+#(?P<source_index>\d+)$")


def parse_arecord_devices(text: str) -> list[AudioDeviceCandidate]:
    return _parse_alsa_device_listing(text, kind="input", parser="arecord")


def parse_aplay_devices(text: str) -> list[AudioDeviceCandidate]:
    return _parse_alsa_device_listing(text, kind="output", parser="aplay")


def _parse_alsa_device_listing(
    text: str,
    *,
    kind: Literal["input", "output"],
    parser: Literal["arecord", "aplay"],
) -> list[AudioDeviceCandidate]:
    candidates: list[AudioDeviceCandidate] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _ALSA_DEVICE_LINE_RE.match(line)
        if not match:
            continue

        card_index = int(match.group("card_index"))
        device_index = int(match.group("device_index"))
        card_name = match.group("card_name").strip()
        device_name = match.group("device_name").strip()
        score = 100 - (card_index * 5) - device_index
        reason = f"parsed from {parser} device listing"
        if "USB" in f"{card_name} {device_name}".upper():
            score += 5
            reason = f"{reason}; usb audio device candidate"

        candidates.append(
            AudioDeviceCandidate(
                name=_candidate_name(card_name, device_name),
                kind=kind,
                device=f"hw:{card_index},{device_index}",
                score=score,
                reason=reason,
                metadata={
                    "card_index": card_index,
                    "card_id": match.group("card_id").strip(),
                    "card_name": match.group("card_id").strip(),
                    "card_label": card_name,
                    "device_index": device_index,
                    "device_id": match.group("device_id").strip(),
                    "device_name": device_name,
                    "parser": parser,
                },
            )
        )
    return candidates


def parse_pactl_sources(text: str) -> list[AudioDeviceCandidate]:
    candidates: list[AudioDeviceCandidate] = []
    current: dict[str, str] = {}

    def flush() -> None:
        if not current.get("name"):
            return
        device = current["name"]
        monitor_of_sink = current.get("monitor_of_sink", "")
        is_monitor = (
            (bool(monitor_of_sink) and monitor_of_sink.lower() not in {"n/a", "na", "none"})
            or device.endswith(".monitor")
        )
        kind: AudioDeviceKind = "loopback" if is_monitor else "input"
        description = current.get("description") or device
        score = 120 if is_monitor else 90
        reason = "parsed from pactl source listing"
        if is_monitor:
            reason = f"{reason}; pulse monitor source available for loopback"
        candidates.append(
            AudioDeviceCandidate(
                name=description,
                kind=kind,
                device=device,
                score=score,
                reason=reason,
                metadata={
                    "description": description,
                    "parser": "pactl",
                    "source_index": _safe_int(current.get("source_index")),
                    "state": current.get("state", ""),
                    "monitor_of_sink": current.get("monitor_of_sink", ""),
                },
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        source_match = _PACTL_SOURCE_LINE_RE.match(stripped)
        if source_match:
            flush()
            current = {"source_index": source_match.group("source_index")}
            continue
        if not current or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        current[normalized_key] = value.strip()

    flush()
    return candidates


@dataclass(frozen=True)
class AcousticFrontendReadiness:
    aec: bool
    ns: bool
    vad: bool
    loopback: bool
    capture: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.capture and self.loopback and self.aec and self.ns and self.vad

    def to_dict(self) -> dict[str, Any]:
        return {
            "aec": self.aec,
            "ns": self.ns,
            "vad": self.vad,
            "loopback": self.loopback,
            "capture": self.capture,
            "healthy": self.healthy,
            "warnings": list(self.warnings),
        }


def evaluate_audio_frontend_readiness(
    capture_device: str | None,
    loopback_device: str | None = None,
    supports_aec: bool = False,
    supports_ns: bool = False,
    supports_vad: bool = False,
) -> AcousticFrontendReadiness:
    capture = bool(capture_device)
    loopback = bool(loopback_device)
    warnings: list[str] = []

    if not capture:
        warnings.append("capture unavailable; microphone input is blocked")
    if not loopback:
        warnings.append("loopback unavailable; speaker echo reference is degraded")
    if not supports_aec:
        warnings.append("AEC unavailable; echo cancellation is degraded")
    if not supports_ns:
        warnings.append("NS unavailable; noise suppression is degraded")
    if not supports_vad:
        warnings.append("VAD unavailable; endpointing is degraded")

    return AcousticFrontendReadiness(
        aec=supports_aec,
        ns=supports_ns,
        vad=supports_vad,
        loopback=loopback,
        capture=capture,
        warnings=warnings,
    )


def build_loopback_readiness(
    capture_device: str | None,
    loopback_device: str | None = None,
    supports_aec: bool = False,
    supports_ns: bool = False,
    supports_vad: bool = False,
) -> AcousticFrontendReadiness:
    return evaluate_audio_frontend_readiness(
        capture_device=capture_device,
        loopback_device=loopback_device,
        supports_aec=supports_aec,
        supports_ns=supports_ns,
        supports_vad=supports_vad,
    )


@dataclass(frozen=True)
class AudioRoutePlan:
    capture: AudioDeviceCandidate | None
    playback: AudioDeviceCandidate | None = None
    loopback: AudioDeviceCandidate | None = None
    readiness: AcousticFrontendReadiness = field(
        default_factory=lambda: evaluate_audio_frontend_readiness(capture_device=None)
    )
    status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": _candidate_to_dict(self.capture),
            "playback": _candidate_to_dict(self.playback),
            "loopback": _candidate_to_dict(self.loopback),
            "readiness": self.readiness.to_dict(),
            "status": dict(self.status),
        }


def choose_audio_routes(
    inputs: Iterable[AudioDeviceCandidate],
    outputs: Iterable[AudioDeviceCandidate] = (),
    loopbacks: Iterable[AudioDeviceCandidate] = (),
    *,
    supports_aec: bool = False,
    supports_ns: bool = False,
    supports_vad: bool = False,
) -> AudioRoutePlan:
    input_candidates = [candidate for candidate in inputs if candidate.kind == "input"]
    output_candidates = [candidate for candidate in outputs if candidate.kind == "output"]
    loopback_candidates = [candidate for candidate in loopbacks if candidate.kind == "loopback"]
    warnings: list[str] = []

    capture: AudioDeviceCandidate | None = None
    if input_candidates:
        capture = select_preferred_input(input_candidates)
        if capture.metadata.get("degraded"):
            warnings.append("using degraded SPA3700 input fallback")
    else:
        warnings.append("no capture candidate parsed from provided discovery output")

    playback = max(output_candidates, key=lambda candidate: candidate.score, default=None)
    if playback is None:
        warnings.append("playback candidate unavailable; output route will remain optional")

    loopback = max(loopback_candidates, key=lambda candidate: candidate.score, default=None)
    if loopback is None:
        warnings.append("loopback candidate unavailable; echo reference will be optional")

    readiness = build_loopback_readiness(
        capture_device=capture.device if capture else None,
        loopback_device=loopback.device if loopback else None,
        supports_aec=supports_aec,
        supports_ns=supports_ns,
        supports_vad=supports_vad,
    )
    status = {
        "capture_device": capture.device if capture else None,
        "playback_device": playback.device if playback else None,
        "loopback_device": loopback.device if loopback else None,
        "warnings": warnings + [
            warning for warning in readiness.warnings if warning not in warnings
        ],
    }
    return AudioRoutePlan(
        capture=capture,
        playback=playback,
        loopback=loopback,
        readiness=readiness,
        status=status,
    )


@dataclass(frozen=True)
class PlaybackInterruptionPlan:
    stop_command: list[str]
    reason: str = "barge-in requires playback stop before capture continues"
    expected_max_ms: int = 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_command": list(self.stop_command),
            "reason": self.reason,
            "expected_max_ms": self.expected_max_ms,
        }


def build_playback_stop_plan(
    route_plan: AudioRoutePlan,
    service_name: str = "eivoice-playback",
) -> PlaybackInterruptionPlan:
    capture_device = route_plan.capture.device if route_plan.capture else "none"
    loopback_device = route_plan.loopback.device if route_plan.loopback else "none"
    warnings = route_plan.status.get("warnings") or []
    reason = (
        f"barge-in requires playback stop before capture continues; "
        f"capture={capture_device}; loopback={loopback_device}"
    )
    if warnings:
        reason = f"{reason}; warnings={', '.join(str(item) for item in warnings)}"
    return PlaybackInterruptionPlan(
        stop_command=["systemctl", "stop", service_name],
        reason=reason,
    )


def _candidate_name(card_name: str, device_name: str) -> str:
    return " ".join(part for part in (card_name.strip(), device_name.strip()) if part)


def _candidate_to_dict(candidate: AudioDeviceCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "name": candidate.name,
        "kind": candidate.kind,
        "device": candidate.device,
        "score": candidate.score,
        "reason": candidate.reason,
        "metadata": dict(candidate.metadata),
    }


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "AcousticFrontendReadiness",
    "AudioDeviceCandidate",
    "AudioRoutePlan",
    "PlaybackInterruptionPlan",
    "build_aplay_command",
    "build_arecord_command",
    "build_loopback_readiness",
    "build_playback_stop_plan",
    "choose_audio_routes",
    "evaluate_audio_frontend_readiness",
    "parse_aplay_devices",
    "parse_arecord_devices",
    "parse_pactl_sources",
    "select_preferred_input",
]
