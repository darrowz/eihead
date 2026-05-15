"""Native provider state boundary for the head runtime.

The helpers in this module report whether a native provider is wired without
opening devices. Hardware checks must be supplied by an injected probe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import os
from typing import Any, Callable, Mapping


NATIVE_PROVIDER_NAMES = ("eye", "ear", "mouth", "neck")
NATIVE_PROVIDER_STATUSES = {"wired", "unknown", "unavailable", "degraded"}
NativeProviderProbe = Callable[..., Mapping[str, Any] | Any | None]


@dataclass(frozen=True, slots=True)
class NativeProviderStatus:
    status: str
    provider: str = ""
    reason: str = ""
    source: str = ""
    checked_at: float | None = None
    last_checked: float | None = None
    hardware_verified: bool | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.provider:
            payload["provider"] = self.provider
        if self.reason:
            payload["reason"] = self.reason
        if self.source:
            payload["source"] = self.source
        if self.checked_at is not None:
            payload["checked_at"] = self.checked_at
        if self.last_checked is not None:
            payload["last_checked"] = self.last_checked
        if self.hardware_verified is not None:
            payload["hardware_verified"] = self.hardware_verified
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def build_native_provider_statuses(
    *,
    config: Any | None = None,
    environ: Mapping[str, str] | None = None,
    probe: NativeProviderProbe | None = None,
    neck_servo_adapter: Any | None = None,
) -> dict[str, dict[str, Any]]:
    env = dict(os.environ if environ is None else environ)
    statuses: dict[str, NativeProviderStatus] = {}

    for provider_name in NATIVE_PROVIDER_NAMES:
        env_status = _status_from_env(provider_name, env)
        if env_status is not None:
            statuses[provider_name] = env_status
            continue

        if provider_name == "neck" and neck_servo_adapter is None:
            statuses[provider_name] = NativeProviderStatus(
                "unavailable",
                reason="neck_servo_adapter_missing",
            )
            continue

        probed_status = _status_from_probe(provider_name, config=config, environ=env, probe=probe)
        if probed_status is not None:
            statuses[provider_name] = probed_status
            continue

        statuses[provider_name] = _status_from_config(provider_name, config) or NativeProviderStatus("unknown")

    return {name: status.to_dict() for name, status in statuses.items()}


def normalize_native_provider_statuses(
    statuses: Mapping[str, Any] | None,
    *,
    neck_servo_adapter: Any | None = None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    raw_statuses = statuses if isinstance(statuses, Mapping) else {}
    for provider_name in NATIVE_PROVIDER_NAMES:
        raw_status = raw_statuses.get(provider_name)
        if provider_name == "neck" and neck_servo_adapter is None and raw_status is None:
            normalized[provider_name] = NativeProviderStatus(
                "unavailable",
                reason="neck_servo_adapter_missing",
            ).to_dict()
            continue
        normalized[provider_name] = _normalize_status(raw_status).to_dict()
    return normalized


def _status_from_env(provider_name: str, environ: Mapping[str, str]) -> NativeProviderStatus | None:
    prefix = f"EIHEAD_NATIVE_{provider_name.upper()}"
    status = _normalize_status_text(environ.get(f"{prefix}_STATUS"))
    if status is None:
        return None
    provider = _string_value(environ.get(f"{prefix}_PROVIDER"))
    reason = _string_value(environ.get(f"{prefix}_REASON"))
    return NativeProviderStatus(status, provider=provider, reason=reason)


def _status_from_probe(
    provider_name: str,
    *,
    config: Any | None,
    environ: Mapping[str, str],
    probe: NativeProviderProbe | None,
) -> NativeProviderStatus | None:
    if probe is None:
        return None
    raw_status = probe(provider_name, config=config, environ=dict(environ))
    if raw_status is None:
        return None
    return _normalize_status(raw_status)


def _status_from_config(provider_name: str, config: Any | None) -> NativeProviderStatus | None:
    declaration = _provider_declaration(provider_name, config)
    if not declaration:
        return None
    if declaration.get("enabled") is False:
        return NativeProviderStatus("unavailable", reason=f"{provider_name}_disabled_in_config")
    status = _normalize_status_text(declaration.get("status"))
    if status is None:
        return NativeProviderStatus("unknown")
    return NativeProviderStatus(
        status,
        provider=_string_value(declaration.get("provider") or declaration.get("backend")),
        reason=_string_value(declaration.get("reason")),
        details=_details_without_status_fields(declaration),
    )


def _provider_declaration(provider_name: str, config: Any | None) -> dict[str, Any] | None:
    raw = getattr(config, "raw", None)
    if not isinstance(raw, Mapping):
        return None

    for section_name in ("native_providers", "providers"):
        section = raw.get(section_name)
        if isinstance(section, Mapping) and isinstance(section.get(provider_name), Mapping):
            return dict(section[provider_name])

    native = raw.get("native")
    if isinstance(native, Mapping):
        section = native.get("providers")
        if isinstance(section, Mapping) and isinstance(section.get(provider_name), Mapping):
            return dict(section[provider_name])
    return None


def _normalize_status(raw_status: Any) -> NativeProviderStatus:
    if isinstance(raw_status, NativeProviderStatus):
        return raw_status
    if isinstance(raw_status, Mapping):
        status = _normalize_mapping_status(raw_status)
        return NativeProviderStatus(
            status,
            provider=_string_value(raw_status.get("provider") or raw_status.get("backend")),
            reason=_string_value(raw_status.get("reason")),
            source=_string_value(raw_status.get("source") or raw_status.get("status_source")),
            checked_at=_optional_float(raw_status.get("checked_at") or raw_status.get("checked_at_ts")),
            last_checked=_native_last_checked(raw_status),
            hardware_verified=_optional_bool(raw_status.get("hardware_verified")),
            details=_details_without_status_fields(raw_status),
        )
    if hasattr(raw_status, "to_dict") and callable(raw_status.to_dict):
        data = raw_status.to_dict()
        if isinstance(data, Mapping):
            return _normalize_status(data)
    if is_dataclass(raw_status):
        return _normalize_status(asdict(raw_status))
    return NativeProviderStatus(_normalize_status_text(raw_status) or "unknown")


def _normalize_mapping_status(raw_status: Mapping[str, Any]) -> str:
    status = _normalize_status_text(raw_status.get("status"))
    if status is not None and status != "unknown":
        return status
    readiness_status = _status_from_readiness(raw_status)
    if readiness_status is not None:
        return readiness_status
    return status or "unknown"


def _status_from_readiness(raw_status: Mapping[str, Any]) -> str | None:
    raw_text = _string_value(raw_status.get("status")).strip().lower()
    not_wired = _truthy(raw_status.get("not_wired")) or raw_text in {
        "not_wired",
        "offline",
        "missing",
        "placeholder",
        "unavailable",
    }
    placeholder = _truthy(raw_status.get("placeholder"))
    stream_ready = _optional_bool(raw_status.get("stream_ready"))
    degraded = _truthy(raw_status.get("degraded")) or raw_text == "degraded"
    mode = _string_value(raw_status.get("mode")).strip().lower()
    kind = _string_value(raw_status.get("kind")).strip().lower()

    if not_wired or placeholder:
        return "unavailable"
    if degraded:
        return "degraded"
    if stream_ready is True:
        return "wired"
    if stream_ready is False and not not_wired and not placeholder:
        return "degraded"
    if stream_ready is False:
        return "unavailable"
    if raw_text in {"tracking", "ready", "running", "online", "ok", "live"} and (
        mode in {"realtime", "realtime_stream"} or kind == "realtime_vision_observation"
    ):
        return "wired"
    return None


def _normalize_status_text(value: Any) -> str | None:
    text = _string_value(value).strip().lower()
    if not text:
        return None
    return text if text in NATIVE_PROVIDER_STATUSES else "unknown"


def _details_without_status_fields(raw_status: Mapping[str, Any]) -> dict[str, Any]:
    details = raw_status.get("details")
    if isinstance(details, Mapping):
        return dict(details)
    status_keys = {
        "status",
        "provider",
        "reason",
        "source",
        "status_source",
        "checked_at",
        "checked_at_ts",
        "last_checked",
        "last_checked_ts",
        "hardware_verified",
    }
    return {
        str(key): value
        for key, value in raw_status.items()
        if key not in status_keys
    }


def _native_last_checked(raw_status: Mapping[str, Any]) -> float | None:
    last_checked = _optional_float(raw_status.get("last_checked") or raw_status.get("last_checked_ts"))
    if last_checked is not None:
        return last_checked
    return _optional_float(raw_status.get("checked_at") or raw_status.get("checked_at_ts"))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
