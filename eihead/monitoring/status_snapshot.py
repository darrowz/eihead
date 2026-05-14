"""Status snapshot helpers for honjia monitoring and future ingestion."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping

from eihead.services.capability_registry import CapabilityRegistry, DEGRADED, OFFLINE, ONLINE


Clock = Callable[[], float]


def build_status_snapshot(
    registry: CapabilityRegistry | Mapping[str, Any] | None = None,
    *,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready snapshot from a registry or an existing manifest."""

    now = (clock or time.time)()
    manifest = registry.manifest() if isinstance(registry, CapabilityRegistry) else dict(registry or {})
    capabilities = dict(manifest.get("capabilities") or {})
    summary = _summarize(capabilities)

    return {
        "schema": "eihead.status_snapshot.v1",
        "node_id": manifest.get("node_id", "honjia"),
        "captured_at_ts": now,
        "overall_status": _overall_status(summary),
        "summary": summary,
        "capabilities": capabilities,
        "manifest_schema": manifest.get("schema"),
        "manifest_generated_at_ts": manifest.get("generated_at_ts"),
    }


def snapshot_to_json(snapshot: Mapping[str, Any]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)


def _summarize(capabilities: Mapping[str, Any]) -> dict[str, int]:
    summary = {ONLINE: 0, DEGRADED: 0, OFFLINE: 0, "total": 0}
    for payload in capabilities.values():
        if not isinstance(payload, Mapping):
            continue
        status = str(payload.get("status") or OFFLINE)
        if status not in {ONLINE, DEGRADED, OFFLINE}:
            status = DEGRADED
        summary[status] += 1
        summary["total"] += 1
    return summary


def _overall_status(summary: Mapping[str, int]) -> str:
    if summary.get(ONLINE, 0) == 0 and summary.get(DEGRADED, 0) == 0:
        return OFFLINE
    if summary.get(OFFLINE, 0) or summary.get(DEGRADED, 0):
        return DEGRADED
    return ONLINE


__all__ = ["build_status_snapshot", "snapshot_to_json"]
