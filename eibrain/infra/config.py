"""Configuration helpers for deployable runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}")


@dataclass(slots=True)
class DriverConfig:
    kind: str = "noop"
    command: list[str] = field(default_factory=list)
    endpoint: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 5.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubfunctionConfig:
    enabled: bool = True
    health: str = "healthy"
    driver: DriverConfig = field(default_factory=DriverConfig)


@dataclass(slots=True)
class OrganConfig:
    enabled: bool = True
    subfunctions: dict[str, SubfunctionConfig] = field(default_factory=dict)


@dataclass(slots=True)
class BodyConfig:
    node_id: str = "honjia"
    foundation: dict[str, str] = field(default_factory=dict)
    organs: dict[str, OrganConfig] = field(default_factory=dict)


@dataclass(slots=True)
class LLMConfig:
    provider: str = "echo"
    model: str = ""
    endpoint: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 256
    supports_vision: bool = False
    experimental: bool = False


@dataclass(slots=True)
class CognitionConfig:
    node_id: str = "honxin"
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision_llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass(slots=True)
class MiniMaxMCPConfig:
    enabled: bool = False
    command: list[str] = field(default_factory=lambda: ["uvx", "minimax-coding-plan-mcp", "-y"])
    api_key: str = ""
    api_host: str = "https://api.minimaxi.com"
    base_path: str = ""
    resource_mode: str = "url"


@dataclass(slots=True)
class MiniMaxCLIConfig:
    enabled: bool = False
    command: list[str] = field(default_factory=lambda: ["mmx"])
    api_key: str = ""
    base_url: str = "https://api.minimaxi.com"


@dataclass(slots=True)
class VisionConfig:
    provider: str = "disabled"
    cli: MiniMaxCLIConfig = field(default_factory=MiniMaxCLIConfig)
    mcp: MiniMaxMCPConfig = field(default_factory=MiniMaxMCPConfig)


@dataclass(slots=True)
class OpenClawConfig:
    provider: str = "in_memory"
    endpoint: str = ""
    api_key: str = ""
    timeout_s: float = 5.0
    tenant_id: str = "default"
    agent_id: str = ""
    workspace_id: str = ""


@dataclass(slots=True)
class MemoryConfig:
    openclaw: OpenClawConfig = field(default_factory=OpenClawConfig)


@dataclass(slots=True)
class MonitoringConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 18081


@dataclass(slots=True)
class SystemConfig:
    project_name: str = "eibrain"
    environment: str = "development"


@dataclass(slots=True)
class DeploymentConfig:
    root_dir: str = field(default_factory=lambda: f"/home/{os.environ.get('USER', '$user')}/eibrain")
    body_runtime_dir: str = ""
    cognitive_runtime_dir: str = ""
    allow_override: bool = True

    def __post_init__(self) -> None:
        if not self.body_runtime_dir:
            self.body_runtime_dir = self.root_dir
        if not self.cognitive_runtime_dir:
            self.cognitive_runtime_dir = self.root_dir


@dataclass(slots=True)
class EIBrainConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def _default_config_path() -> Path:
    return Path.cwd() / "config" / "eibrain.yaml"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(3) or ""),
            value,
        )
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _parse_driver(raw: dict[str, Any] | None) -> DriverConfig:
    payload = dict(raw or {})
    if "health_command" in payload:
        payload["health_command"] = _parse_command(payload["health_command"])
    return DriverConfig(
        kind=str(payload.pop("kind", "noop")),
        command=_parse_command(payload.pop("command", [])),
        endpoint=str(payload.pop("endpoint", "")),
        method=str(payload.pop("method", "POST")),
        headers={str(key): str(value) for key, value in dict(payload.pop("headers", {})).items()},
        timeout_s=float(payload.pop("timeout_s", 5.0)),
        extra=payload,
    )


def _parse_subfunction(raw: dict[str, Any] | None) -> SubfunctionConfig:
    payload = dict(raw or {})
    return SubfunctionConfig(
        enabled=bool(payload.pop("enabled", True)),
        health=str(payload.pop("health", "healthy")),
        driver=_parse_driver(payload.pop("driver", None)),
    )


def _parse_organ(raw: dict[str, Any] | None) -> OrganConfig:
    payload = dict(raw or {})
    enabled = bool(payload.pop("enabled", True))
    subfunctions = {name: _parse_subfunction(config) for name, config in payload.items()}
    return OrganConfig(enabled=enabled, subfunctions=subfunctions)


def _parse_command(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def load_config(path: str | Path | None = None) -> EIBrainConfig:
    config_path = Path(path) if path is not None else _default_config_path()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expanded = _expand_env(raw)

    system = SystemConfig(**dict(expanded.get("system", {})))
    body_payload = dict(expanded.get("body", {}))
    cognition_payload = dict(expanded.get("cognition", {}))
    vision_payload = dict(expanded.get("vision", {}))
    memory_payload = dict(expanded.get("memory", {}))
    monitoring_payload = dict(expanded.get("monitoring", {}))

    body = BodyConfig(
        node_id=str(body_payload.pop("node_id", "honjia")),
        foundation={str(key): str(value) for key, value in dict(body_payload.pop("foundation", {})).items()},
        organs={name: _parse_organ(config) for name, config in dict(body_payload.pop("organs", {})).items()},
    )
    cognition = CognitionConfig(
        node_id=str(cognition_payload.pop("node_id", "honxin")),
        llm=LLMConfig(**dict(cognition_payload.pop("llm", {}))),
        vision_llm=LLMConfig(**dict(cognition_payload.pop("vision_llm", {}))),
    )
    vision = VisionConfig(
        provider=str(vision_payload.pop("provider", "disabled")),
        cli=MiniMaxCLIConfig(
            **{
                **dict(vision_payload.pop("cli", {})),
                "command": _parse_command(dict(expanded.get("vision", {})).get("cli", {}).get("command", ["mmx"])),
            }
        ),
        mcp=MiniMaxMCPConfig(
            **{
                **dict(vision_payload.pop("mcp", {})),
                "command": _parse_command(dict(expanded.get("vision", {})).get("mcp", {}).get("command", ["uvx", "minimax-coding-plan-mcp", "-y"])),
            }
        ),
    )
    memory = MemoryConfig(
        openclaw=OpenClawConfig(**dict(memory_payload.pop("openclaw", {}))),
    )
    deployment = DeploymentConfig(**dict(expanded.get("deployment", {})))
    return EIBrainConfig(
        system=system,
        deployment=deployment,
        body=body,
        cognition=cognition,
        vision=vision,
        memory=memory,
        monitoring=MonitoringConfig(**monitoring_payload),
    )
