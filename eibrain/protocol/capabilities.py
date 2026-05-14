"""Capability contracts for eihead registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable
from typing import Any

from .base import ProtocolMessage


def _without_kind(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned.pop("kind", None)
    return cleaned


@dataclass(slots=True)
class HeadLimit:
    """Numeric or enumerated runtime limit for a head device/backend."""

    name: str = ""
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""
    step: float | None = None
    values: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadLimit":
        payload = dict(data)
        payload["values"] = list(payload.get("values", []))
        payload["metadata"] = dict(payload.get("metadata", {}))
        return cls(**payload)


@dataclass(slots=True)
class HeadHealth:
    """Health snapshot reported by eihead."""

    status: str = "unknown"
    message: str = ""
    checked_at_ms: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HeadHealth":
        if data is None:
            return cls()
        payload = dict(data)
        payload["metrics"] = dict(payload.get("metrics", {}))
        return cls(**payload)


@dataclass(slots=True)
class HeadDevice:
    """Physical or logical device exposed by eihead."""

    device_id: str = ""
    kind: str = ""
    name: str = ""
    path: str = ""
    enabled: bool = True
    capabilities: list[str] = field(default_factory=list)
    limits: list[HeadLimit] = field(default_factory=list)
    health: HeadHealth = field(default_factory=HeadHealth)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadDevice":
        payload = dict(data)
        payload["capabilities"] = list(payload.get("capabilities", []))
        payload["limits"] = [
            item if isinstance(item, HeadLimit) else HeadLimit.from_dict(item)
            for item in payload.get("limits", [])
        ]
        payload["health"] = (
            payload["health"]
            if isinstance(payload.get("health"), HeadHealth)
            else HeadHealth.from_dict(payload.get("health"))
        )
        payload["metadata"] = dict(payload.get("metadata", {}))
        return cls(**payload)


@dataclass(slots=True)
class HeadBackend:
    """Runtime backend exposed by eihead, such as ASR, TTS, vision, or embedding."""

    backend_id: str = ""
    kind: str = ""
    provider: str = ""
    model: str = ""
    version: str = ""
    enabled: bool = True
    capabilities: list[str] = field(default_factory=list)
    limits: list[HeadLimit] = field(default_factory=list)
    health: HeadHealth = field(default_factory=HeadHealth)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadBackend":
        payload = dict(data)
        payload["capabilities"] = list(payload.get("capabilities", []))
        payload["limits"] = [
            item if isinstance(item, HeadLimit) else HeadLimit.from_dict(item)
            for item in payload.get("limits", [])
        ]
        payload["health"] = (
            payload["health"]
            if isinstance(payload.get("health"), HeadHealth)
            else HeadHealth.from_dict(payload.get("health"))
        )
        payload["metadata"] = dict(payload.get("metadata", {}))
        return cls(**payload)


@dataclass(slots=True)
class CapabilityManifest(ProtocolMessage):
    """Startup registration payload sent from eihead to eibrain."""

    trace_id: str = ""
    target: str = "eibrain"
    timestamp_ms: int | None = None
    node_id: str = ""
    node_role: str = "eihead"
    protocol_version: str = "head.v1"
    devices: list[HeadDevice] = field(default_factory=list)
    backends: list[HeadBackend] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    health: HeadHealth = field(default_factory=HeadHealth)
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="capability_manifest")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityManifest":
        payload = _without_kind(data)
        payload["devices"] = [
            item if isinstance(item, HeadDevice) else HeadDevice.from_dict(item)
            for item in payload.get("devices", [])
        ]
        payload["backends"] = [
            item if isinstance(item, HeadBackend) else HeadBackend.from_dict(item)
            for item in payload.get("backends", [])
        ]
        payload["capabilities"] = list(payload.get("capabilities", []))
        payload["health"] = (
            payload["health"]
            if isinstance(payload.get("health"), HeadHealth)
            else HeadHealth.from_dict(payload.get("health"))
        )
        payload["metadata"] = dict(payload.get("metadata", {}))
        return cls(**payload)


def build_modality_inventory(
    devices: Iterable[HeadDevice],
    backends: Iterable[HeadBackend],
) -> dict[str, dict[str, Any]]:
    audio = {
        "available": False,
        "microphones": [],
        "speakers": [],
        "asr": [],
        "tts": [],
    }
    vision = {"available": False, "cameras": [], "backends": []}
    actuation = {"available": False, "neck": []}
    embedding = {"available": False, "backends": []}

    for device in devices:
        bucket = classify_device_bucket(device)
        if bucket == "microphones":
            audio["microphones"].append(device.device_id)
        elif bucket == "speakers":
            audio["speakers"].append(device.device_id)
        elif bucket == "cameras":
            vision["cameras"].append(device.device_id)
        elif bucket == "neck":
            actuation["neck"].append(device.device_id)

    for backend in backends:
        bucket = classify_backend_bucket(backend)
        if bucket == "asr":
            audio["asr"].append(backend.backend_id)
        elif bucket == "tts":
            audio["tts"].append(backend.backend_id)
        elif bucket == "vision":
            vision["backends"].append(backend.backend_id)
        elif bucket == "embedding":
            embedding["backends"].append(backend.backend_id)

    audio["available"] = any(audio[key] for key in ("microphones", "speakers", "asr", "tts"))
    vision["available"] = bool(vision["cameras"] or vision["backends"])
    actuation["available"] = bool(actuation["neck"])
    embedding["available"] = bool(embedding["backends"])

    modalities: dict[str, dict[str, Any]] = {}
    if audio["available"]:
        modalities["audio"] = audio
    if vision["available"]:
        modalities["vision"] = vision
    if actuation["available"]:
        modalities["actuation"] = actuation
    if embedding["available"]:
        modalities["embedding"] = embedding
    return modalities


def classify_device_bucket(device: HeadDevice) -> str | None:
    kind = str(device.kind or "").strip().lower()
    capabilities = {str(item).strip().lower() for item in device.capabilities}

    if kind == "camera":
        return "cameras"
    if kind in {"microphone", "mic", "audio_input"}:
        return "microphones"
    if kind in {"speaker", "audio_output"}:
        return "speakers"
    if kind in {"neck", "actuator", "servo"} or {"move_head", "neck_follow"} & capabilities:
        return "neck"
    return None


def classify_backend_bucket(backend: HeadBackend) -> str | None:
    kind = str(backend.kind or "").strip().lower()
    if kind in {"asr", "speech_to_text"}:
        return "asr"
    if kind in {"tts", "text_to_speech"}:
        return "tts"
    if kind == "vision":
        return "vision"
    if kind == "embedding":
        return "embedding"
    return None
