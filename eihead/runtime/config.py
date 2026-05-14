"""Lightweight YAML config loader for the standalone eihead runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by monkeypatch in tests
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR: ImportError | None = exc
else:
    _YAML_IMPORT_ERROR = None


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")
DEFAULT_EIHEAD_CONFIG_PATH = Path("config") / "eihead.honjia.yaml"


class EiheadConfigError(RuntimeError):
    """Raised when an eihead runtime config cannot be loaded."""


@dataclass(slots=True)
class EndpointConfig:
    host: str
    port: int

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}


@dataclass(slots=True)
class DevicePathsConfig:
    camera: str = "/dev/video0"
    hailo: str = "/dev/hailo0"
    i2c: str = "/dev/i2c-1"
    microphone: str = "/dev/snd"
    speaker: str = ""
    neck: str = "/dev/i2c-1"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera": self.camera,
            "hailo": self.hailo,
            "i2c": self.i2c,
            "microphone": self.microphone,
            "speaker": self.speaker,
            "neck": self.neck,
        }
        payload.update(self.extra)
        return payload


@dataclass(slots=True)
class SoftwareCapabilityConfig:
    enabled: bool = True
    provider: str = ""
    backend: str = ""
    model: str = ""
    model_dir: str = ""
    limits: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["enabled"] = self.enabled
        if self.provider:
            payload["provider"] = self.provider
        if self.backend:
            payload["backend"] = self.backend
        if self.model:
            payload["model"] = self.model
        if self.model_dir:
            payload["model_dir"] = self.model_dir
        if self.limits:
            payload["limits"] = dict(self.limits)
        return payload


@dataclass(slots=True)
class CapabilityDeclarationsConfig:
    software: dict[str, SoftwareCapabilityConfig] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload["software"] = {name: declaration.to_dict() for name, declaration in self.software.items()}
        return payload


@dataclass(slots=True)
class LegacyConfig:
    eibrain_config_path: str = "config/eibrain.honjia.yaml"
    body_runtime_config_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.body_runtime_config_path:
            self.body_runtime_config_path = self.eibrain_config_path

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "eibrain_config_path": self.eibrain_config_path,
                "body_runtime_config_path": self.body_runtime_config_path,
            }
        )
        return payload


@dataclass(slots=True)
class EiheadConfig:
    node_id: str = "honjia"
    runtime: EndpointConfig = field(default_factory=lambda: EndpointConfig("127.0.0.1", 18081))
    monitor: EndpointConfig = field(default_factory=lambda: EndpointConfig("0.0.0.0", 18080))
    devices: DevicePathsConfig = field(default_factory=DevicePathsConfig)
    capabilities: CapabilityDeclarationsConfig = field(default_factory=CapabilityDeclarationsConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "runtime": self.runtime.to_dict(),
            "monitor": self.monitor.to_dict(),
            "devices": self.devices.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "legacy": self.legacy.to_dict(),
        }

    def as_capability_registry_config(self) -> dict[str, Any]:
        capabilities: dict[str, Any] = {
            "camera": {"path": self.devices.camera, "limits": {"streams": 1}},
            "hailo": {"path": self.devices.hailo, "limits": {"device_count": 1}},
            "i2c": {"path": self.devices.i2c, "limits": {"bus": _i2c_bus_from_path(self.devices.i2c)}},
            "microphone": {"path": self.devices.microphone, "limits": {"channels": 1}},
            "speaker": {"enabled": True},
            "neck": {
                "path": self.devices.neck,
                "limits": {"pan_deg": [0, 180], "tilt_deg": None},
            },
        }
        if self.devices.speaker:
            capabilities["speaker"]["path"] = self.devices.speaker
        for name, declaration in self.capabilities.software.items():
            capabilities[name] = declaration.to_dict()
        return {"node_id": self.node_id, "capabilities": capabilities}


def load_eihead_config(path: str | Path | None = None) -> EiheadConfig:
    config_path = Path(path) if path is not None else DEFAULT_EIHEAD_CONFIG_PATH
    raw = _read_yaml_mapping(config_path)
    return parse_eihead_config(_expand_env(raw))


def parse_eihead_config(raw: Mapping[str, Any] | None = None) -> EiheadConfig:
    payload = dict(raw or {})
    runtime = _parse_endpoint(payload.get("runtime"), default_host="127.0.0.1", default_port=18081)
    monitor = _parse_endpoint(
        payload.get("monitor", payload.get("monitoring")),
        default_host="0.0.0.0",
        default_port=18080,
    )
    return EiheadConfig(
        node_id=str(payload.get("node_id") or "honjia"),
        runtime=runtime,
        monitor=monitor,
        devices=_parse_devices(payload.get("devices")),
        capabilities=_parse_capabilities(payload.get("capabilities")),
        legacy=_parse_legacy(payload.get("legacy")),
        raw=payload,
    )


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise EiheadConfigError(
            "PyYAML is required to load eihead YAML config; install PyYAML in the eihead runtime environment."
        ) from _YAML_IMPORT_ERROR

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EiheadConfigError(f"Failed to read eihead config at {path}: {exc}") from exc
    except Exception as exc:
        raise EiheadConfigError(f"Failed to parse eihead YAML config at {path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise EiheadConfigError(f"eihead config at {path} must be a YAML mapping")
    return dict(loaded)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(3) or ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _parse_endpoint(raw: Any, *, default_host: str, default_port: int) -> EndpointConfig:
    payload = _mapping(raw)
    return EndpointConfig(
        host=str(payload.get("host") or default_host),
        port=_as_int(payload.get("port", default_port), field_name="port"),
    )


def _parse_devices(raw: Any) -> DevicePathsConfig:
    payload = _mapping(raw)
    known = {
        "camera": _device_path(payload.pop("camera", None), "/dev/video0"),
        "hailo": _device_path(payload.pop("hailo", None), "/dev/hailo0"),
        "i2c": _device_path(payload.pop("i2c", None), "/dev/i2c-1"),
        "microphone": _device_path(payload.pop("microphone", None), "/dev/snd"),
        "speaker": _device_path(payload.pop("speaker", None), ""),
        "neck": _device_path(payload.pop("neck", None), "/dev/i2c-1"),
    }
    return DevicePathsConfig(**known, extra=dict(payload))


def _parse_capabilities(raw: Any) -> CapabilityDeclarationsConfig:
    payload = _mapping(raw)
    software_payload = _mapping(payload.pop("software", {}))
    for alias in ("asr", "tts", "vision_backend", "embedding"):
        if alias in payload and isinstance(payload[alias], Mapping):
            software_payload.setdefault(alias, payload.pop(alias))
    software = {
        str(name): _parse_software_capability(config)
        for name, config in software_payload.items()
        if isinstance(config, Mapping)
    }
    return CapabilityDeclarationsConfig(software=software, extra=dict(payload))


def _parse_software_capability(raw: Mapping[str, Any]) -> SoftwareCapabilityConfig:
    payload = dict(raw)
    return SoftwareCapabilityConfig(
        enabled=bool(payload.pop("enabled", True)),
        provider=str(payload.pop("provider", "")),
        backend=str(payload.pop("backend", "")),
        model=str(payload.pop("model", "")),
        model_dir=str(payload.pop("model_dir", "")),
        limits=dict(payload.pop("limits", {}) or {}),
        extra=payload,
    )


def _parse_legacy(raw: Any) -> LegacyConfig:
    payload = _mapping(raw)
    eibrain_config_path = str(
        payload.pop("eibrain_config_path", payload.pop("config_path", "config/eibrain.honjia.yaml"))
    )
    return LegacyConfig(
        eibrain_config_path=eibrain_config_path,
        body_runtime_config_path=str(payload.pop("body_runtime_config_path", "")),
        extra=dict(payload),
    )


def _device_path(raw: Any, default: str) -> str:
    if raw is None:
        return default
    if isinstance(raw, Mapping):
        if "path" in raw:
            return str(raw["path"])
        if "device_path" in raw:
            return str(raw["device_path"])
        if "bus" in raw:
            return f"/dev/i2c-{_as_int(raw['bus'], field_name='bus')}"
        return default
    return str(raw)


def _mapping(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _as_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EiheadConfigError(f"eihead config field {field_name!r} must be an integer") from exc


def _i2c_bus_from_path(path: str) -> int | None:
    match = re.search(r"/dev/i2c-(\d+)$", path)
    if not match:
        return None
    return int(match.group(1))
