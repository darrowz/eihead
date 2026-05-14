"""Composition helpers for eihead runtime status/probe shaping."""

from __future__ import annotations

from typing import Any, Mapping

CAPABILITY_NATIVE_PROVIDER_MAP = {
    "camera": "eye",
    "hailo": "eye",
    "vision_backend": "eye",
    "microphone": "ear",
    "asr": "ear",
    "speaker": "mouth",
    "tts": "mouth",
    "neck": "neck",
}


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _provider_status(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return _string_or_default(payload.get("status"), "unknown").strip().lower() or "unknown"
    return "unknown"


def _capability_status_from_native_provider(native_status: str, *, hardware_verified: bool) -> str:
    if native_status == "wired":
        return "live" if hardware_verified else "online"
    if native_status in {"degraded", "unavailable", "unknown"}:
        return native_status
    return "unknown"


def build_native_capability_probe(
    native_providers: Mapping[str, Any],
):
    providers = dict(native_providers)

    def probe(name: str, *, config: dict[str, Any], static_status: dict[str, Any]) -> dict[str, Any] | None:
        provider_name = CAPABILITY_NATIVE_PROVIDER_MAP.get(name)
        if provider_name is None:
            return None
        provider_payload = providers.get(provider_name)
        if not isinstance(provider_payload, Mapping):
            return None

        native_status = _provider_status(provider_payload)
        hardware_verified = _bool_or_none(provider_payload.get("hardware_verified")) is True
        capability_status = _capability_status_from_native_provider(native_status, hardware_verified=hardware_verified)
        details: dict[str, Any] = {
            "native_provider": provider_name,
            "native_status": native_status,
        }
        native_details = provider_payload.get("details")
        if isinstance(native_details, Mapping):
            details.update({f"native_{key}": value for key, value in native_details.items()})

        return {
            "status": capability_status,
            "source": _string_or_default(provider_payload.get("source"), "native_provider"),
            "reason": _string_or_default(
                provider_payload.get("reason"),
                "native_provider_status",
            ),
            "checked_at": _optional_float(provider_payload.get("checked_at")),
            "last_checked": _optional_float(provider_payload.get("last_checked")),
            "hardware_verified": hardware_verified,
            "provider": provider_payload.get("provider"),
            "details": details,
            "native_provider_status": native_status,
            "static_status": static_status.get("status"),
            "config_kind": config.get("kind"),
        }

    return probe


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CAPABILITY_NATIVE_PROVIDER_MAP",
    "build_native_capability_probe",
]

