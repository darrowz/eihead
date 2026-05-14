"""Runtime health/status projection for legacy head runtime checks."""

from __future__ import annotations

from typing import Any, Mapping


def runtime_check_summary(
    *,
    delegate_name: str,
    native_providers: Mapping[str, Any],
    body_snapshot_check: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any], str]:
    delegate_check, delegate_details = _delegate_check(delegate_name)
    native_check, native_details = _native_provider_check(native_providers)
    body_check = _string_or_default(body_snapshot_check.get("status"), "unknown")
    checks = {
        "head_runtime_import": "ok",
        "body_runtime_delegate": delegate_check,
        "body_runtime_snapshot": body_check,
        "native_provider_boundaries": native_check,
    }
    check_details = {
        "body_runtime_delegate": delegate_details,
        "body_runtime_snapshot": dict(body_snapshot_check),
        "native_provider_boundaries": native_details,
    }
    return checks, check_details, _overall_runtime_status(checks.values())


def _delegate_check(delegate_name: str) -> tuple[str, dict[str, Any]]:
    if delegate_name == "apps.body_runtime.BodyRuntime":
        return (
            "ok",
            {
                "delegate": delegate_name,
                "reason": "legacy_body_runtime_delegate_active",
                "compatibility_mode": True,
            },
        )
    if not delegate_name:
        return "unknown", {"delegate": delegate_name, "reason": "delegate_unknown"}
    return "ok", {"delegate": delegate_name}


def _native_provider_check(native_providers: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    provider_states: dict[str, str] = {}
    non_wired: dict[str, str] = {}
    for provider_name, provider_payload in native_providers.items():
        provider_state = _provider_status(provider_payload)
        provider_states[str(provider_name)] = provider_state
        if provider_state != "wired":
            non_wired[str(provider_name)] = provider_state

    if non_wired:
        return (
            "degraded",
            {
                "reason": "native_provider_not_wired",
                "providers": provider_states,
                "non_wired": non_wired,
            },
        )
    return "ok", {"providers": provider_states}


def _provider_status(provider_payload: Any) -> str:
    if isinstance(provider_payload, Mapping):
        return _string_or_default(provider_payload.get("status"), "unknown").strip().lower() or "unknown"
    return "unknown"


def _overall_runtime_status(checks: Any) -> str:
    states = {_string_or_default(state, "unknown").strip().lower() for state in checks}
    if states & {"blocked", "error", "failed"}:
        return "blocked"
    if states & {"degraded", "unknown", "unavailable"}:
        return "degraded"
    return "ok"


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


__all__ = ["runtime_check_summary"]
