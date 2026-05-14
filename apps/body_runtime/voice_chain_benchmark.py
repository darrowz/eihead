from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import ceil, isfinite
from numbers import Real
from typing import Any


DEFAULT_THRESHOLDS = {
    "asrFinalMs": 800.0,
    "firstTokenMs": 700.0,
    "firstAudioMs": 2000.0,
    "interruptStopMs": 300.0,
}

READINESS_SUMMARY_SCHEMA = "eibrain.voice_chain_readiness_summary.v1"

_METRIC_LABELS = {
    "wakeToListenMs": "wake_to_listen",
    "firstAsrPartialMs": "listen_asr_partial",
    "asrFinalMs": "asr_final",
    "firstLlmDeltaMs": "llm_first_delta",
    "firstTokenMs": "first_token",
    "firstTtsChunkMs": "tts_first_chunk",
    "firstAudioMs": "first_audio",
    "interruptStopMs": "interrupt_stop",
}

_STAGE_LATENCY_KEYS = (
    "stageLatencyMs",
    "stage_latency_ms",
    "lastStageLatencyMs",
    "last_stage_latency_ms",
)

_STREAMING_SIGNALS = {
    "asrPartial": (
        "asrPartial",
        "asr_partial",
        "asrPartialSeen",
        "partialAsr",
        "partial_asr",
        "firstAsrPartialMs",
    ),
    "asrFinal": ("asrFinal", "asr_final", "finalAsr", "final_asr", "asrFinalMs"),
    "llmDelta": (
        "llmDelta",
        "llm_delta",
        "replyDelta",
        "reply_delta",
        "agentDelta",
        "agent_delta",
        "firstLlmDeltaMs",
        "firstTokenMs",
    ),
    "ttsChunk": (
        "ttsChunk",
        "tts_chunk",
        "audioChunk",
        "audio_chunk",
        "ttsAudioChunk",
        "tts_audio_chunk",
        "firstTtsChunkMs",
    ),
    "playback": ("playback", "playbackStarted", "playback_started", "playback_started_seen"),
}


def summarize_voice_chain(turns: Iterable[Mapping[str, Any]], *, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    turn_list = [turn for turn in turns if isinstance(turn, Mapping)]
    threshold_values = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        threshold_values.update(_coerce_thresholds(thresholds))
    fields = _metric_fields(turn_list, threshold_values)

    metrics = {}
    for field in fields:
        values = [_as_float(turn[field]) for turn in turn_list if field in turn]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        threshold = threshold_values.get(field)
        p95 = _nearest_rank_p95(numeric_values)
        metrics[field] = {
            "count": len(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "p95": p95,
            "threshold": threshold,
            "pass": p95 <= threshold if threshold is not None else None,
        }

    round_leak_count = sum(1 for turn in turn_list if _is_round_leak(turn))
    turn_count = len(turn_list)
    rounds = [_round_report(index, turn) for index, turn in enumerate(turn_list)]
    round_leak = _round_leak_summary(round_leak_count, turn_count)
    interrupt_stop = _interrupt_stop_summary(turn_list, threshold_values)
    streaming = _streaming_summary(rounds)
    bottleneck = _bottleneck(metrics)
    failed_metrics = _failed_metrics(metrics)
    readiness_summary = _readiness_summary(
        turn_count=turn_count,
        failed_metrics=failed_metrics,
        round_leak=round_leak,
        interrupt_stop=interrupt_stop,
        streaming=streaming,
        bottleneck=bottleneck,
    )
    summary = {
        "turnCount": turn_count,
        "roundLeakCount": round_leak_count,
        "roundLeakRate": round_leak_count / turn_count if turn_count else 0.0,
        "thresholds": threshold_values,
        "metrics": metrics,
        "stageLatencyMetrics": _stage_latency_metrics(rounds),
        "rounds": rounds,
        "roundLeak": round_leak,
        "interruptStop": interrupt_stop,
        "streaming": streaming,
        "bottleneck": bottleneck,
        "readinessSummary": readiness_summary,
    }
    summary["joyinsideReadiness"] = _joyinside_readiness_from_summary(summary)
    return summary


def summarize_joyinside_voice_readiness(
    turns: Iterable[Mapping[str, Any]], *, thresholds: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the JoyInside voice acceptance gate for direct CLI/Web display."""
    return _joyinside_readiness_from_summary(summarize_voice_chain(turns, thresholds=thresholds))


def _coerce_thresholds(thresholds: Mapping[str, Any]) -> dict[str, float]:
    coerced = {}
    for field, value in thresholds.items():
        threshold = _as_float(value)
        if threshold is not None:
            coerced[str(field)] = threshold
    return coerced


def _metric_fields(turns: list[Mapping[str, Any]], thresholds: Mapping[str, float]) -> list[str]:
    fields = list(_METRIC_LABELS)
    extra_fields = set(thresholds) - set(fields)
    for turn in turns:
        extra_fields.update(str(field) for field in turn if str(field).endswith("Ms") and field not in _METRIC_LABELS)
    fields.extend(sorted(extra_fields))
    return fields


def _round_report(index: int, turn: Mapping[str, Any]) -> dict[str, Any]:
    streaming = _turn_streaming_readiness(turn)
    report: dict[str, Any] = {
        "index": index,
        "roundId": _round_id(turn, index),
        "stageLatencyMs": _stage_latency_ms(turn),
        "roundLeak": _is_round_leak(turn),
        "interrupted": _is_interrupted(turn),
        "streamingReady": streaming["ready"],
    }
    status = turn.get("status")
    if status not in (None, ""):
        report["status"] = str(status)
    if streaming["missingSignals"]:
        report["streamingMissingSignals"] = list(streaming["missingSignals"])
    for field in _METRIC_LABELS:
        value = _as_float(turn.get(field))
        if value is not None:
            report[field] = value
    return report


def _round_id(turn: Mapping[str, Any], index: int) -> str:
    for key in ("roundId", "round_id", "id"):
        value = turn.get(key)
        if value not in (None, ""):
            return str(value)
    return f"turn-{index + 1}"


def _stage_latency_ms(turn: Mapping[str, Any]) -> dict[str, float]:
    stages: dict[str, float] = {}
    for key in _STAGE_LATENCY_KEYS:
        value = turn.get(key)
        if isinstance(value, Mapping):
            stages.update(_coerce_numeric_mapping(value))
    for field, label in _METRIC_LABELS.items():
        value = _as_float(turn.get(field))
        if value is not None:
            stages.setdefault(label, value)
    return stages


def _coerce_numeric_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        number = _as_float(item)
        if number is not None:
            result[str(key)] = number
    return result


def _stage_latency_metrics(rounds: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    stage_names = sorted(
        {
            str(name)
            for item in rounds
            if isinstance(item.get("stageLatencyMs"), Mapping)
            for name in item["stageLatencyMs"]
        }
    )
    metrics: dict[str, dict[str, Any]] = {}
    for name in stage_names:
        values = [
            value
            for item in rounds
            if isinstance(item.get("stageLatencyMs"), Mapping)
            for value in [_as_float(item["stageLatencyMs"].get(name))]
            if value is not None
        ]
        if not values:
            continue
        metrics[name] = {
            "count": len(values),
            "avg": sum(values) / len(values),
            "p95": _nearest_rank_p95(values),
        }
    return metrics


def _round_leak_summary(round_leak_count: int, turn_count: int) -> dict[str, Any]:
    rate = round_leak_count / turn_count if turn_count else 0.0
    return {
        "count": round_leak_count,
        "rate": rate,
        "free": round_leak_count == 0,
        "pass": round_leak_count == 0,
    }


def _interrupt_stop_summary(turns: list[Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    threshold = thresholds.get("interruptStopMs")
    required_count = 0
    confirmed_count = 0
    failures: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not _is_interrupted(turn):
            continue
        required_count += 1
        value = _as_float(turn.get("interruptStopMs"))
        passed = value is not None and (threshold is None or value <= threshold)
        if passed:
            confirmed_count += 1
        else:
            failures.append(
                {
                    "roundId": _round_id(turn, index),
                    "interruptStopMs": value,
                    "threshold": threshold,
                }
            )
    failed_count = required_count - confirmed_count
    return {
        "requiredCount": required_count,
        "confirmedCount": confirmed_count,
        "failedCount": failed_count,
        "threshold": threshold,
        "ready": required_count > 0 and failed_count == 0,
        "failures": failures,
    }


def _streaming_summary(rounds: list[Mapping[str, Any]]) -> dict[str, Any]:
    missing = [
        {
            "roundId": item.get("roundId"),
            "index": item.get("index"),
            "signals": list(item.get("streamingMissingSignals", _STREAMING_SIGNALS)),
        }
        for item in rounds
        if item.get("streamingReady") is not True
    ]
    ready_turn_count = sum(1 for item in rounds if item.get("streamingReady") is True)
    turn_count = len(rounds)
    return {
        "ready": turn_count > 0 and ready_turn_count == turn_count,
        "turnCount": turn_count,
        "readyTurnCount": ready_turn_count,
        "missingSignals": missing,
    }


def _turn_streaming_readiness(turn: Mapping[str, Any]) -> dict[str, Any]:
    explicit = _explicit_bool(turn.get("streamingReady"), turn.get("streaming_ready"))
    if explicit is not None:
        return {"ready": explicit, "missingSignals": [] if explicit else list(_STREAMING_SIGNALS)}

    streaming = turn.get("streaming")
    source = streaming if isinstance(streaming, Mapping) else {}
    missing = [
        signal
        for signal, aliases in _STREAMING_SIGNALS.items()
        if not any(
            _truthy_signal(source.get(alias))
            or _truthy_signal(turn.get(alias))
            for alias in aliases
        )
    ]
    return {"ready": not missing, "missingSignals": missing}


def _explicit_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _truthy_signal(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return isfinite(float(value)) and float(value) != 0.0
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text or text in {"0", "false", "no", "off", "missing", "none", "null"}:
        return False
    return True


def _is_interrupted(turn: Mapping[str, Any]) -> bool:
    status = str(turn.get("status") or "").lower()
    return status == "interrupted" or turn.get("interrupted") is True or turn.get("interruptActive") is True


def _is_round_leak(turn: Mapping[str, Any]) -> bool:
    status = str(turn.get("status") or "").lower()
    return turn.get("roundLeak") is True or status == "stale_round_blocked"


def _failed_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        str(name)
        for name, metric in metrics.items()
        if isinstance(metric, Mapping) and metric.get("pass") is False
    ]


def _readiness_summary(
    *,
    turn_count: int,
    failed_metrics: list[str],
    round_leak: Mapping[str, Any],
    interrupt_stop: Mapping[str, Any],
    streaming: Mapping[str, Any],
    bottleneck: Mapping[str, Any],
) -> dict[str, Any]:
    round_leak_free = bool(round_leak.get("free"))
    interrupt_stop_ready = bool(interrupt_stop.get("ready"))
    streaming_ready = bool(streaming.get("ready"))
    honjia_ready = bool(turn_count) and not failed_metrics and round_leak_free and interrupt_stop_ready and streaming_ready
    reasons: list[str] = []
    if turn_count <= 0:
        reasons.append("no turns")
    if failed_metrics:
        reasons.append("failed metrics: " + ", ".join(failed_metrics))
    if not round_leak_free:
        reasons.append("round leak")
    if not interrupt_stop_ready:
        reasons.append("interrupt stop")
    if not streaming_ready:
        reasons.append("streaming")
    message = "ready: voice-chain acceptance passed" if honjia_ready else "not ready: " + ", ".join(reasons)
    return {
        "schema": READINESS_SUMMARY_SCHEMA,
        "source": "voice_chain_benchmark",
        "live": True,
        "honjiaReady": honjia_ready,
        "codeReady": honjia_ready,
        "turnCount": turn_count,
        "failedMetrics": failed_metrics,
        "roundLeakCount": int(round_leak.get("count") or 0),
        "roundLeakRate": float(round_leak.get("rate") or 0.0),
        "roundLeakFree": round_leak_free,
        "interruptStopReady": interrupt_stop_ready,
        "streamingReady": streaming_ready,
        "bottleneck": dict(bottleneck),
        "summary": message,
        "readinessMessage": message,
    }


def _joyinside_readiness_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    turn_count = int(summary.get("turnCount") or 0)
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), Mapping) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    interrupt_stop = summary.get("interruptStop") if isinstance(summary.get("interruptStop"), Mapping) else {}
    round_leak = summary.get("roundLeak") if isinstance(summary.get("roundLeak"), Mapping) else {}
    streaming = summary.get("streaming") if isinstance(summary.get("streaming"), Mapping) else {}

    failed_metrics = _failed_metrics(metrics)
    blocking_reasons: list[str] = []
    next_actions: list[str] = []

    if turn_count <= 0:
        blocking_reasons.append("no_turns")
        next_actions.append("Run at least one benchmark turn before evaluating JoyInside readiness.")

    for field in failed_metrics:
        metric = metrics.get(field)
        threshold = metric.get("threshold") if isinstance(metric, Mapping) else thresholds.get(field)
        blocking_reasons.append(f"{field}_p95_exceeded")
        if threshold is not None:
            next_actions.append(f"Reduce {field} p95 to <= {threshold}ms.")
        else:
            next_actions.append(f"Reduce {field} p95 below the configured threshold.")

    interrupt_ready = bool(interrupt_stop.get("ready"))
    if not interrupt_ready:
        threshold = interrupt_stop.get("threshold", thresholds.get("interruptStopMs"))
        blocking_reasons.append("interrupt_not_confirmed")
        if threshold is not None:
            next_actions.append(f"Capture at least one interrupted turn with interruptStopMs <= {threshold}ms.")
        else:
            next_actions.append("Capture at least one interrupted turn with confirmed interruptStopMs.")

    round_leak_free = bool(round_leak.get("free")) and turn_count > 0
    if turn_count > 0 and not round_leak_free:
        blocking_reasons.append("round_leak_detected")
        next_actions.append("Fix stale round suppression until roundLeak count is 0.")

    streaming_ready = bool(streaming.get("ready"))
    if not streaming_ready:
        blocking_reasons.append("streaming_signals_missing")
        next_actions.append("Emit ASR partial/final, LLM delta, TTS chunk, and playback streaming signals for every turn.")

    latency_ready = turn_count > 0 and not failed_metrics
    score = 0
    score += 25 if latency_ready else 0
    score += 25 if interrupt_ready else 0
    score += 25 if round_leak_free else 0
    score += 25 if streaming_ready else 0
    ready = score == 100 and not blocking_reasons
    return {
        "ready": ready,
        "score": score,
        "grade": _joyinside_grade(score),
        "blocking_reasons": blocking_reasons,
        "next_actions": next_actions,
    }


def _joyinside_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 50:
        return "C"
    if score >= 25:
        return "D"
    return "F"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(1, ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def _bottleneck(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selected = None
    selected_ratio = None
    for field, metric in metrics.items():
        threshold = metric.get("threshold")
        p95 = metric.get("p95")
        if threshold is None or threshold <= 0 or p95 is None:
            continue
        ratio = p95 / threshold
        if selected_ratio is None or ratio > selected_ratio:
            selected = (field, metric, ratio)
            selected_ratio = ratio

    if selected is None:
        return {"field": None, "label": None, "p95": None, "threshold": None, "ratio": None}

    field, metric, ratio = selected
    return {
        "field": field,
        "label": _METRIC_LABELS.get(field, field),
        "p95": metric["p95"],
        "threshold": metric["threshold"],
        "ratio": ratio,
    }
