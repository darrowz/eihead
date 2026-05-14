"""In-memory registry for eihead capability and status ingestion."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class _HeadNode:
    node_id: str
    last_seen_ts: float = 0.0
    overall_status: str = "unknown"
    capabilities: dict[str, JsonObject] = field(default_factory=dict)
    manifest: JsonObject = field(default_factory=dict)
    status: JsonObject = field(default_factory=dict)
    errors: list[JsonObject] = field(default_factory=list)

    def to_dict(self) -> JsonObject:
        return {
            "node_id": self.node_id,
            "last_seen_ts": self.last_seen_ts,
            "overall_status": self.overall_status,
            "capabilities": _copy_jsonish(self.capabilities),
            "manifest": _copy_jsonish(self.manifest),
            "status": _copy_jsonish(self.status),
            "errors": _copy_jsonish(self.errors),
        }


class HeadRegistry:
    """Lightweight eibrain-side cache for eihead runtime capabilities.

    The registry is intentionally process-local and dependency-free. It accepts
    plain dict payloads, dataclass-like objects exposing ``to_dict()``, and
    HeadClient-like clients exposing ``get_capabilities()`` / ``get_status()``.
    """

    def __init__(
        self,
        *,
        default_node_id: str = "honjia",
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        if not default_node_id.strip():
            raise ValueError("default_node_id must not be empty")
        self.default_node_id = default_node_id
        self._time_fn = time_fn or time.time
        self._nodes: dict[str, _HeadNode] = {}

    def update_manifest(
        self,
        manifest: Mapping[str, Any] | object,
        *,
        node_id: str | None = None,
        observed_at: float | None = None,
    ) -> JsonObject:
        """Ingest a capability manifest or /capabilities response."""

        payload = self._coerce_payload(manifest, label="manifest")
        payload = self._unwrap_manifest(payload)
        effective_node_id = self._node_id_from(payload, node_id=node_id)
        node = self._ensure_node(effective_node_id)
        node.last_seen_ts = self._observed_at(observed_at)
        node.manifest = payload
        node.capabilities = self._merge_capability_payload(node.capabilities, payload, source="manifest")
        status = _extract_status(payload)
        if status != "unknown":
            node.overall_status = status
        node.errors = _merge_errors(node.errors, _extract_errors(payload, source="manifest", ts=node.last_seen_ts))
        return node.to_dict()

    def update_status(
        self,
        status: Mapping[str, Any] | object,
        *,
        node_id: str | None = None,
        observed_at: float | None = None,
    ) -> JsonObject:
        """Ingest a status snapshot from eihead."""

        payload = self._coerce_payload(status, label="status")
        effective_node_id = self._node_id_from(payload, node_id=node_id)
        node = self._ensure_node(effective_node_id)
        node.last_seen_ts = self._observed_at(observed_at)
        node.status = payload
        node.capabilities = self._merge_capability_payload(node.capabilities, payload, source="status")
        status_value = _extract_status(payload)
        incoming_errors = _extract_errors(payload, source="status", ts=node.last_seen_ts)
        if status_value != "unknown":
            node.overall_status = status_value
        elif incoming_errors:
            node.overall_status = "degraded"
        node.errors = _merge_errors(node.errors, incoming_errors)
        return node.to_dict()

    def update_from_client(self, client: object, *, node_id: str | None = None) -> JsonObject:
        """Pull capabilities and status from a HeadClient-like object.

        Client errors are recorded on the node instead of being raised so that
        monitoring can still show partial data when one endpoint fails.
        """

        effective_node_id = node_id or self.default_node_id
        capabilities_payload: JsonObject | None = None
        try:
            capabilities_payload = self._call_client(client, "get_capabilities")
            manifest_node = self.update_manifest(capabilities_payload, node_id=node_id)
            effective_node_id = str(manifest_node["node_id"])
        except Exception as exc:  # pragma: no cover - exercised by tests via behavior
            self._record_client_error(effective_node_id, "get_capabilities", exc)

        try:
            status_payload = self._call_client(client, "get_status")
            status_node_id = node_id or _safe_node_id(capabilities_payload) or effective_node_id
            return self.update_status(status_payload, node_id=status_node_id)
        except Exception as exc:  # pragma: no cover - exercised by tests via behavior
            self._record_client_error(effective_node_id, "get_status", exc)
            return self.get_node(effective_node_id) or self._ensure_node(effective_node_id).to_dict()

    def get_node(self, node_id: str | None = None) -> JsonObject | None:
        """Return a copy of one node snapshot."""

        node = self._nodes.get(node_id or self.default_node_id)
        return node.to_dict() if node else None

    def get_capability(self, capability_id: str, *, node_id: str | None = None) -> JsonObject | None:
        """Return a copy of one indexed capability for a node."""

        if not capability_id:
            raise ValueError("capability_id must not be empty")
        node = self._nodes.get(node_id or self.default_node_id)
        if not node:
            return None
        capability = node.capabilities.get(capability_id)
        return _copy_jsonish(capability) if capability is not None else None

    def summary(self) -> JsonObject:
        """Return a compact registry summary for monitoring surfaces."""

        node_summaries: list[JsonObject] = []
        error_count = 0
        online_count = 0
        degraded_count = 0
        for node_id in sorted(self._nodes):
            node = self._nodes[node_id]
            error_count += len(node.errors)
            if _is_online_status(node.overall_status):
                online_count += 1
            if node.overall_status.lower() == "degraded":
                degraded_count += 1
            node_summaries.append(
                {
                    "node_id": node.node_id,
                    "last_seen_ts": node.last_seen_ts,
                    "overall_status": node.overall_status,
                    "capability_count": len(node.capabilities),
                    "error_count": len(node.errors),
                }
            )
        return {
            "default_node_id": self.default_node_id,
            "node_count": len(self._nodes),
            "online_count": online_count,
            "degraded_count": degraded_count,
            "error_count": error_count,
            "nodes": node_summaries,
        }

    def to_dict(self) -> JsonObject:
        """Return the full registry snapshot."""

        return {
            "default_node_id": self.default_node_id,
            "nodes": {node_id: node.to_dict() for node_id, node in sorted(self._nodes.items())},
            "summary": self.summary(),
        }

    def _ensure_node(self, node_id: str) -> _HeadNode:
        if node_id not in self._nodes:
            self._nodes[node_id] = _HeadNode(node_id=node_id)
        return self._nodes[node_id]

    def _node_id_from(self, payload: Mapping[str, Any], *, node_id: str | None) -> str:
        return node_id or _safe_node_id(payload) or self.default_node_id

    def _observed_at(self, observed_at: float | None) -> float:
        return float(self._time_fn() if observed_at is None else observed_at)

    def _record_client_error(self, node_id: str, operation: str, exc: BaseException) -> None:
        node = self._ensure_node(node_id)
        node.last_seen_ts = self._observed_at(None)
        node.overall_status = "degraded"
        node.errors = _merge_errors(node.errors, [_client_error_to_dict(operation, exc, node.last_seen_ts)])

    @staticmethod
    def _call_client(client: object, method_name: str) -> JsonObject:
        method = getattr(client, method_name, None)
        if not callable(method):
            raise TypeError(f"client must provide {method_name}()")
        payload = method()
        if not isinstance(payload, Mapping):
            raise TypeError(f"{method_name}() must return a mapping")
        return _copy_jsonish(dict(payload))

    @staticmethod
    def _coerce_payload(payload: Mapping[str, Any] | object, *, label: str) -> JsonObject:
        to_dict = getattr(payload, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError(f"{label} must be a mapping or expose to_dict()")
        return _copy_jsonish(dict(payload))

    @staticmethod
    def _unwrap_manifest(payload: JsonObject) -> JsonObject:
        for key in ("manifest", "capability_manifest"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return _copy_jsonish(dict(value))
        return payload

    @classmethod
    def _merge_capability_payload(
        cls,
        existing: Mapping[str, JsonObject],
        payload: Mapping[str, Any],
        *,
        source: str,
    ) -> dict[str, JsonObject]:
        index = {key: _copy_jsonish(value) for key, value in existing.items()}
        cls._index_declared_capabilities(index, payload.get("capabilities"), source=source)
        cls._index_component_collection(index, payload.get("devices"), id_field="device_id", category="device", source=source)
        cls._index_component_collection(index, payload.get("backends"), id_field="backend_id", category="backend", source=source)
        return index

    @classmethod
    def _index_declared_capabilities(
        cls,
        index: dict[str, JsonObject],
        raw: Any,
        *,
        source: str,
    ) -> None:
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                if isinstance(value, Mapping):
                    entry = _copy_jsonish(dict(value))
                elif isinstance(value, list):
                    entry = {"capabilities": _copy_jsonish(value)}
                else:
                    entry = {"value": _copy_jsonish(value)}
                entry.setdefault("id", str(key))
                entry.setdefault("name", str(key))
                entry.setdefault("kind", str(key))
                entry.setdefault("category", "capability")
                entry["source"] = source
                _upsert_capability(index, str(key), entry)
            return
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if isinstance(item, Mapping):
                    entry = _copy_jsonish(dict(item))
                    key = _component_key(entry, "capability_id") or entry.get("name") or entry.get("kind")
                    if not key:
                        continue
                    entry.setdefault("id", str(key))
                    entry.setdefault("name", str(key))
                    entry.setdefault("category", "capability")
                    entry["source"] = source
                    _upsert_capability(index, str(key), entry)
                elif item:
                    key = str(item)
                    _upsert_capability(
                        index,
                        key,
                        {
                            "id": key,
                            "name": key,
                            "kind": key,
                            "category": "capability",
                            "source": source,
                            "online": True,
                        },
                    )

    @classmethod
    def _index_component_collection(
        cls,
        index: dict[str, JsonObject],
        raw: Any,
        *,
        id_field: str,
        category: str,
        source: str,
    ) -> None:
        for item in _iter_component_items(raw, id_field=id_field):
            key = _component_key(item, id_field)
            if not key:
                continue
            entry = _capability_entry(item, key=key, category=category, source=source)
            _upsert_capability(index, key, entry)
            kind = item.get("kind")
            if kind:
                _upsert_capability(index, str(kind), entry)


def _iter_component_items(raw: Any, *, id_field: str) -> Iterable[JsonObject]:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                item = _copy_jsonish(dict(value))
            else:
                item = {"value": _copy_jsonish(value)}
            item.setdefault(id_field, str(key))
            yield item
        return
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        for value in raw:
            if isinstance(value, Mapping):
                yield _copy_jsonish(dict(value))


def _capability_entry(item: Mapping[str, Any], *, key: str, category: str, source: str) -> JsonObject:
    entry = _copy_jsonish(dict(item))
    entry.setdefault("id", key)
    entry.setdefault("category", category)
    entry["source"] = source
    online = _derive_online(entry)
    if online is not None:
        entry["online"] = online
    return entry


def _upsert_capability(index: dict[str, JsonObject], key: str, entry: Mapping[str, Any]) -> None:
    payload = _copy_jsonish(dict(entry))
    existing = index.get(key)
    if existing is None:
        index[key] = payload
        return
    if existing.get("id") == payload.get("id") or existing.get("category") == payload.get("category"):
        merged = {**existing, **payload}
        if isinstance(existing.get("raw"), Mapping) and isinstance(payload.get("raw"), Mapping):
            merged["raw"] = {**existing["raw"], **payload["raw"]}
        index[key] = merged
        return
    if existing.get("category") == "group":
        items = list(existing.get("items", []))
        if not any(isinstance(item, Mapping) and item.get("id") == payload.get("id") for item in items):
            items.append(payload)
        existing["items"] = items
        index[key] = existing
        return
    index[key] = {"id": key, "category": "group", "items": [existing, payload]}


def _component_key(item: Mapping[str, Any], id_field: str) -> str:
    for key in (id_field, "id", "name", "kind"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def _extract_status(payload: Mapping[str, Any]) -> str:
    for key in ("overall_status", "status", "state"):
        value = payload.get(key)
        if value:
            return str(value)
    health = payload.get("health")
    if isinstance(health, Mapping):
        value = health.get("status")
        if value:
            return str(value)
    return "unknown"


def _extract_errors(payload: Mapping[str, Any], *, source: str, ts: float) -> list[JsonObject]:
    errors: list[JsonObject] = []
    raw_errors = payload.get("errors", [])
    if isinstance(raw_errors, Mapping):
        raw_errors = [raw_errors]
    elif isinstance(raw_errors, str):
        raw_errors = [{"message": raw_errors}]
    elif not isinstance(raw_errors, Iterable):
        raw_errors = []

    for item in raw_errors:
        if isinstance(item, Mapping):
            error = _copy_jsonish(dict(item))
        else:
            error = {"message": str(item)}
        error.setdefault("source", source)
        error.setdefault("ts", ts)
        errors.append(error)

    raw_error = payload.get("error")
    if raw_error:
        if isinstance(raw_error, Mapping):
            error = _copy_jsonish(dict(raw_error))
        else:
            error = {"message": str(raw_error)}
        error.setdefault("source", source)
        error.setdefault("ts", ts)
        errors.append(error)
    return errors


def _client_error_to_dict(operation: str, exc: BaseException, ts: float) -> JsonObject:
    to_dict = getattr(exc, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if not isinstance(payload, Mapping):
            payload = {"message": str(exc)}
        else:
            payload = _copy_jsonish(dict(payload))
    else:
        payload = {"kind": exc.__class__.__name__, "message": str(exc)}
    payload.setdefault("operation", operation)
    payload.setdefault("source", "head_client")
    payload.setdefault("ts", ts)
    return payload


def _safe_node_id(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in ("node_id", "id", "head_id"):
        value = payload.get(key)
        if value:
            return str(value)
    manifest = payload.get("manifest")
    if isinstance(manifest, Mapping):
        return _safe_node_id(manifest)
    return ""


def _derive_online(payload: Mapping[str, Any]) -> bool | None:
    if payload.get("enabled") is False:
        return False
    status = _extract_status(payload).lower()
    if _is_online_status(status):
        return True
    if status in {"disabled", "down", "error", "failed", "offline", "unavailable"}:
        return False
    return None


def _is_online_status(status: str) -> bool:
    return status.lower() in {"ok", "online", "ready", "healthy", "available", "up"}


def _merge_errors(existing: list[JsonObject], incoming: list[JsonObject]) -> list[JsonObject]:
    merged = [_copy_jsonish(item) for item in existing]
    seen = {_stable_error_key(item) for item in merged}
    for item in incoming:
        key = _stable_error_key(item)
        if key not in seen:
            merged.append(_copy_jsonish(item))
            seen.add(key)
    return merged


def _stable_error_key(error: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(error.get("source", "")),
        str(error.get("operation", "")),
        str(error.get("message", error.get("kind", ""))),
    )


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_jsonish(item) for item in value]
    return value


__all__ = ["HeadRegistry"]
