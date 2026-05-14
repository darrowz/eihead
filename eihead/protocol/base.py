"""Shared local protocol helpers for eihead messages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, TypeVar


MessageT = TypeVar("MessageT", bound="ProtocolMessage")


def message_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return dataclass constructor kwargs from a serialized message."""

    payload = dict(data)
    payload.pop("kind", None)
    return payload


def copy_dict_field(payload: dict[str, Any], field_name: str) -> None:
    payload[field_name] = dict(payload.get(field_name, {}))


def copy_dict_list_field(payload: dict[str, Any], field_name: str) -> None:
    payload[field_name] = [
        dict(item) if isinstance(item, Mapping) else item
        for item in payload.get(field_name, [])
    ]


@dataclass(slots=True)
class ProtocolMessage:
    """Base dataclass for protocol messages with stable JSON conversion."""

    ts: float
    source: str
    session_id: str | None = None
    actor_id: str | None = None
    target_id: str | None = None
    kind: str = field(init=False, default="protocol_message")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[MessageT], payload: Mapping[str, Any]) -> MessageT:
        init_fields = {field.name for field in fields(cls) if field.init}
        cleaned = message_payload(payload)
        return cls(
            **{key: value for key, value in cleaned.items() if key in init_fields}
        )


def serialize_message(message: Any) -> dict[str, Any]:
    """Serialize protocol-like messages, mappings, and dataclasses."""

    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "to_dict") and callable(message.to_dict):
        payload = message.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if is_dataclass(message):
        return asdict(message)
    raise TypeError(f"cannot serialize message of type {type(message).__name__}")

