from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


VOICE_CHAIN_READINESS_SCHEMA = "eibrain.voice_chain_readiness.v1"


def build_voice_chain_readiness(
    *,
    benchmark: Mapping[str, Any] | None = None,
    explicit: Mapping[str, Any] | None = None,
    scenario_targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize voice-chain benchmark data into a monitor-facing readiness payload."""

    if isinstance(explicit, Mapping):
        payload = _json_mapping(explicit)
        payload.setdefault("schema", VOICE_CHAIN_READINESS_SCHEMA)
        payload.setdefault("source", "explicit")
        payload.setdefault("live", _truthy(payload.get("live")))
        payload.setdefault("honjiaReady", _truthy(payload.get("honjiaReady")))
        payload.setdefault("turnCount", _int_or_none(payload.get("turnCount")) or 0)
        payload.setdefault("failedMetrics", _as_text_list(payload.get("failedMetrics")))
        payload.setdefault("roundLeakFree", _truthy(payload.get("roundLeakFree")))
        payload.setdefault("summary", _summary(payload))
        payload.setdefault("readinessMessage", _readiness_message(payload))
        return payload

    if isinstance(benchmark, Mapping):
        return _readiness_from_benchmark(benchmark)

    targets = _json_mapping(scenario_targets) if isinstance(scenario_targets, Mapping) else {}
    return {
        "schema": VOICE_CHAIN_READINESS_SCHEMA,
        "source": "waiting_for_live_benchmark",
        "live": False,
        "honjiaReady": False,
        "turnCount": 0,
        "failedMetrics": [],
        "roundLeakCount": None,
        "roundLeakFree": False,
        "metrics": {},
        "bottleneck": None,
        "scenarioTargets": targets,
        "summary": "waiting for live benchmark",
        "readinessMessage": "waiting for live voice-chain benchmark; offline scenarios are target baselines",
    }


def _readiness_from_benchmark(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    data = _json_mapping(benchmark)
    metrics = data.get("metrics") if isinstance(data.get("metrics"), Mapping) else {}
    failed_metrics = _failed_metrics(metrics)
    round_leak_count = _int_or_none(data.get("roundLeakCount")) or 0
    turn_count = _int_or_none(data.get("turnCount")) or 0
    round_leak_free = round_leak_count == 0
    readiness_summary = data.get("readinessSummary") if isinstance(data.get("readinessSummary"), Mapping) else {}
    streaming = data.get("streaming") if isinstance(data.get("streaming"), Mapping) else {}
    interrupt_stop = data.get("interruptStop") if isinstance(data.get("interruptStop"), Mapping) else {}
    streaming_ready = _optional_truthy(readiness_summary.get("streamingReady"), streaming.get("ready"))
    interrupt_stop_ready = _optional_truthy(readiness_summary.get("interruptStopReady"), interrupt_stop.get("ready"))
    honjia_ready = (
        bool(turn_count)
        and round_leak_free
        and not failed_metrics
        and streaming_ready is not False
        and interrupt_stop_ready is not False
    )
    payload: dict[str, Any] = {
        "schema": VOICE_CHAIN_READINESS_SCHEMA,
        "source": "live_benchmark",
        "live": True,
        "honjiaReady": honjia_ready,
        "turnCount": turn_count,
        "failedMetrics": failed_metrics,
        "roundLeakCount": round_leak_count,
        "roundLeakFree": round_leak_free,
        "metrics": _json_mapping(metrics),
        "bottleneck": data.get("bottleneck") if isinstance(data.get("bottleneck"), Mapping) else None,
        "streamingReady": streaming_ready if streaming_ready is not None else None,
        "interruptStopReady": interrupt_stop_ready if interrupt_stop_ready is not None else None,
        "summary": "",
        "readinessMessage": "",
    }
    if "roundLeakRate" in data:
        payload["roundLeakRate"] = data["roundLeakRate"]
    if isinstance(data.get("thresholds"), Mapping):
        payload["thresholds"] = _json_mapping(data["thresholds"])
    if isinstance(data.get("stageLatencyMetrics"), Mapping):
        payload["stageLatencyMetrics"] = _json_mapping(data["stageLatencyMetrics"])
    if isinstance(streaming, Mapping) and streaming:
        payload["streaming"] = _json_mapping(streaming)
    if isinstance(interrupt_stop, Mapping) and interrupt_stop:
        payload["interruptStop"] = _json_mapping(interrupt_stop)
    if isinstance(readiness_summary, Mapping) and readiness_summary:
        payload["readinessSummary"] = _json_mapping(readiness_summary)
    payload["summary"] = _summary(payload)
    payload["readinessMessage"] = _readiness_message(payload)
    return payload


def _failed_metrics(metrics: Mapping[str, Any]) -> list[str]:
    failed: list[str] = []
    for name, metric in metrics.items():
        if isinstance(metric, Mapping) and metric.get("pass") is False:
            failed.append(str(name))
    return failed


def _summary(payload: Mapping[str, Any]) -> str:
    source = str(payload.get("source") or "")
    turn_count = _int_or_none(payload.get("turnCount")) or 0
    if source == "waiting_for_live_benchmark":
        return "waiting for live benchmark"
    if _truthy(payload.get("honjiaReady")):
        return f"ready: {turn_count} live {_plural('turn', turn_count)}"
    failed = _as_text_list(payload.get("failedMetrics"))
    if failed:
        return "not ready: " + ", ".join(failed)
    if not _truthy(payload.get("roundLeakFree")):
        return "not ready: round leak"
    if payload.get("streamingReady") is False:
        return "not ready: streaming"
    if payload.get("interruptStopReady") is False:
        return "not ready: interrupt stop"
    if turn_count <= 0:
        return "not ready: no live turns"
    return "not ready"


def _readiness_message(payload: Mapping[str, Any]) -> str:
    if _truthy(payload.get("honjiaReady")):
        return "voice chain benchmark is within thresholds"
    summary = str(payload.get("summary") or _summary(payload))
    if summary.startswith("waiting"):
        return "waiting for live voice-chain benchmark"
    return summary


def _json_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in mapping.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return [str(value)] if value else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "ready", "ok", "wired"}


def _optional_truthy(*values: Any) -> bool | None:
    for value in values:
        if value is not None:
            return _truthy(value)
    return None


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"
