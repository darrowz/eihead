"""JSON-ready realtime observation helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


OBSERVATION_KINDS: tuple[str, ...] = (
    "asr_partial",
    "asr_final",
    "vision",
    "prosody",
    "environment",
    "user_interrupt",
)


def to_json_ready(value: Any) -> Any:
    """Return a structure that can be passed to json.dumps without adapters."""

    if is_dataclass(value) and not isinstance(value, type):
        return to_json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _without_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def observation(
    *,
    kind: str,
    round_id: str,
    cancellation_token: str | None = None,
    payload: Mapping[str, Any] | None = None,
    observed_at_ts: float | None = None,
    source: str | None = None,
    stable: bool = False,
    **payload_fields: Any,
) -> dict[str, Any]:
    """Build a normalized observation envelope for realtime lanes."""

    if kind not in OBSERVATION_KINDS:
        raise ValueError(f"unsupported realtime observation kind: {kind}")

    merged_payload: dict[str, Any] = {}
    if payload:
        merged_payload.update(dict(payload))
    merged_payload.update(payload_fields)

    event: dict[str, Any] = {
        "type": "observation",
        "kind": kind,
        "round_id": round_id,
        "stable": bool(stable),
        "payload": to_json_ready(_without_none(merged_payload)),
    }
    if cancellation_token is not None:
        event["cancellation_token"] = cancellation_token
    if observed_at_ts is not None:
        event["observed_at_ts"] = float(observed_at_ts)
    if source is not None:
        event["source"] = source
    return event


def asr_partial(
    *,
    round_id: str,
    text: str,
    cancellation_token: str | None = None,
    confidence: float | None = None,
    observed_at_ts: float | None = None,
    source: str = "asr",
) -> dict[str, Any]:
    return observation(
        kind="asr_partial",
        round_id=round_id,
        cancellation_token=cancellation_token,
        text=text.strip(),
        confidence=confidence,
        observed_at_ts=observed_at_ts,
        source=source,
        stable=False,
    )


def asr_final(
    *,
    round_id: str,
    text: str,
    cancellation_token: str | None = None,
    confidence: float | None = None,
    observed_at_ts: float | None = None,
    source: str = "asr",
) -> dict[str, Any]:
    return observation(
        kind="asr_final",
        round_id=round_id,
        cancellation_token=cancellation_token,
        text=text.strip(),
        confidence=confidence,
        observed_at_ts=observed_at_ts,
        source=source,
        stable=True,
    )


def vision(
    *,
    round_id: str,
    hints: Mapping[str, Any],
    cancellation_token: str | None = None,
    observed_at_ts: float | None = None,
    source: str = "vision",
) -> dict[str, Any]:
    return observation(
        kind="vision",
        round_id=round_id,
        cancellation_token=cancellation_token,
        payload=hints,
        observed_at_ts=observed_at_ts,
        source=source,
        stable=False,
    )


def prosody(
    *,
    round_id: str,
    hints: Mapping[str, Any],
    cancellation_token: str | None = None,
    observed_at_ts: float | None = None,
    source: str = "prosody",
) -> dict[str, Any]:
    return observation(
        kind="prosody",
        round_id=round_id,
        cancellation_token=cancellation_token,
        payload=hints,
        observed_at_ts=observed_at_ts,
        source=source,
        stable=False,
    )


def environment(
    *,
    round_id: str,
    hints: Mapping[str, Any],
    cancellation_token: str | None = None,
    observed_at_ts: float | None = None,
    source: str = "environment",
) -> dict[str, Any]:
    return observation(
        kind="environment",
        round_id=round_id,
        cancellation_token=cancellation_token,
        payload=hints,
        observed_at_ts=observed_at_ts,
        source=source,
        stable=False,
    )


def user_interrupt(
    *,
    round_id: str,
    reason: str,
    interrupted_round_id: str | None = None,
    cancellation_token: str | None = None,
    observed_at_ts: float | None = None,
    source: str = "user",
) -> dict[str, Any]:
    return observation(
        kind="user_interrupt",
        round_id=round_id,
        cancellation_token=cancellation_token,
        reason=reason,
        interrupted_round_id=interrupted_round_id,
        observed_at_ts=observed_at_ts,
        source=source,
        stable=True,
    )


__all__ = [
    "OBSERVATION_KINDS",
    "asr_final",
    "asr_partial",
    "environment",
    "observation",
    "prosody",
    "to_json_ready",
    "user_interrupt",
    "vision",
]
