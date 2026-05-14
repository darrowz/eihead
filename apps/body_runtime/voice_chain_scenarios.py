from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from apps.body_runtime.voice_chain_benchmark import DEFAULT_THRESHOLDS, summarize_voice_chain


@dataclass(frozen=True, slots=True)
class VoiceScenario:
    name: str
    description: str
    turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "turnCount": len(self.turns),
            "turns": [dict(turn) for turn in self.turns],
        }


DEFAULT_SCENARIOS = (
    VoiceScenario(
        name="short_chinese",
        description="single short wake-to-reply turn",
        turns=[
            {
                "roundId": "rnd-short-001",
                "wakeToListenMs": 120.0,
                "firstAsrPartialMs": 180.0,
                "asrFinalMs": 520.0,
                "firstLlmDeltaMs": 360.0,
                "firstTokenMs": 360.0,
                "firstTtsChunkMs": 1180.0,
                "firstAudioMs": 1280.0,
                "interruptStopMs": 180.0,
                "roundLeak": False,
            }
        ],
    ),
    VoiceScenario(
        name="child_fuzzy",
        description="fuzzy child-like utterance with slower ASR finalization",
        turns=[
            {
                "roundId": "rnd-child-001",
                "wakeToListenMs": 180.0,
                "firstAsrPartialMs": 260.0,
                "asrFinalMs": 760.0,
                "firstLlmDeltaMs": 520.0,
                "firstTokenMs": 520.0,
                "firstTtsChunkMs": 1660.0,
                "firstAudioMs": 1760.0,
                "interruptStopMs": 240.0,
                "roundLeak": False,
            }
        ],
    ),
    VoiceScenario(
        name="playback_barge_in",
        description="user interrupts while TTS playback is active",
        turns=[
            {
                "roundId": "rnd-barge-001",
                "firstAsrPartialMs": 210.0,
                "asrFinalMs": 610.0,
                "firstLlmDeltaMs": 440.0,
                "firstTokenMs": 440.0,
                "firstTtsChunkMs": 1390.0,
                "firstAudioMs": 1490.0,
                "interruptStopMs": 210.0,
                "interrupted": True,
                "roundLeak": False,
            }
        ],
    ),
    VoiceScenario(
        name="follow_up_turn",
        description="two consecutive follow-up turns without stale round leakage",
        turns=[
            {
                "roundId": "rnd-follow-001",
                "firstAsrPartialMs": 180.0,
                "asrFinalMs": 560.0,
                "firstLlmDeltaMs": 390.0,
                "firstTokenMs": 390.0,
                "firstTtsChunkMs": 1220.0,
                "firstAudioMs": 1320.0,
                "interruptStopMs": 190.0,
                "roundLeak": False,
            },
            {
                "roundId": "rnd-follow-002",
                "firstAsrPartialMs": 190.0,
                "asrFinalMs": 590.0,
                "firstLlmDeltaMs": 410.0,
                "firstTokenMs": 410.0,
                "firstTtsChunkMs": 1280.0,
                "firstAudioMs": 1380.0,
                "interruptStopMs": 205.0,
                "roundLeak": False,
            },
        ],
    ),
    VoiceScenario(
        name="network_jitter",
        description="provider/network jitter while staying inside first-audio target",
        turns=[
            {
                "roundId": "rnd-jitter-001",
                "firstAsrPartialMs": 240.0,
                "asrFinalMs": 780.0,
                "firstLlmDeltaMs": 690.0,
                "firstTokenMs": 690.0,
                "firstTtsChunkMs": 1860.0,
                "firstAudioMs": 1960.0,
                "interruptStopMs": 280.0,
                "roundLeak": False,
            }
        ],
    ),
)


def run_voice_chain_scenarios(
    *,
    scenarios: Iterable[VoiceScenario] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    using_default_scenarios = scenarios is None
    scenario_list = list(DEFAULT_SCENARIOS if using_default_scenarios else scenarios)
    threshold_values = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        threshold_values.update(dict(thresholds))
    turns = [
        _benchmark_turn(turn, streaming_ready=using_default_scenarios)
        for scenario in scenario_list
        for turn in scenario.turns
    ]
    summary = summarize_voice_chain(turns, thresholds=threshold_values)
    failed_metrics = [
        str(name)
        for name, metric in summary.get("metrics", {}).items()
        if isinstance(metric, Mapping) and metric.get("pass") is False
    ]
    round_leak_free = int(summary.get("roundLeakCount", 0) or 0) == 0
    streaming = summary.get("streaming") if isinstance(summary.get("streaming"), Mapping) else {}
    interrupt_stop = summary.get("interruptStop") if isinstance(summary.get("interruptStop"), Mapping) else {}
    readiness_summary = dict(summary.get("readinessSummary")) if isinstance(summary.get("readinessSummary"), Mapping) else {}
    readiness_summary.update({"source": "offline_scenarios", "live": False})
    streaming_ready = bool(streaming.get("ready"))
    interrupt_stop_ready = bool(interrupt_stop.get("ready"))
    honjia_ready = bool(readiness_summary.get("honjiaReady"))
    return {
        "schema": "eibrain.voice_chain_scenarios.v1",
        "scenarioCount": len(scenario_list),
        "turnCount": len(turns),
        "thresholds": threshold_values,
        "summary": summary,
        "failedMetrics": failed_metrics,
        "roundLeakFree": round_leak_free,
        "streamingReady": streaming_ready,
        "interruptStopReady": interrupt_stop_ready,
        "readinessSummary": readiness_summary,
        "honjiaReady": honjia_ready,
        "scenarios": [scenario.to_dict() for scenario in scenario_list],
    }


def _benchmark_turn(turn: Mapping[str, Any], *, streaming_ready: bool) -> dict[str, Any]:
    item = dict(turn)
    if "firstLlmDeltaMs" not in item and "firstTokenMs" in item:
        item["firstLlmDeltaMs"] = item["firstTokenMs"]
    if "firstTokenMs" not in item and "firstLlmDeltaMs" in item:
        item["firstTokenMs"] = item["firstLlmDeltaMs"]
    if not any(key in item for key in ("stageLatencyMs", "stage_latency_ms", "lastStageLatencyMs", "last_stage_latency_ms")):
        item["stageLatencyMs"] = _derived_stage_latency_ms(item)
    if streaming_ready and "streaming" not in item and "streamingReady" not in item and "streaming_ready" not in item:
        item["streaming"] = {
            "asrPartial": True,
            "asrFinal": True,
            "llmDelta": True,
            "ttsChunk": True,
            "playback": True,
        }
    return item


def _derived_stage_latency_ms(turn: Mapping[str, Any]) -> dict[str, float]:
    asr_partial = _number_or_none(turn.get("firstAsrPartialMs"))
    asr = _number_or_none(turn.get("asrFinalMs"))
    first_delta = _number_or_none(turn.get("firstLlmDeltaMs"))
    first_token = _number_or_none(turn.get("firstTokenMs"))
    first_tts_chunk = _number_or_none(turn.get("firstTtsChunkMs"))
    first_audio = _number_or_none(turn.get("firstAudioMs"))
    stages: dict[str, float] = {}
    if asr_partial is not None:
        stages["listen_asr_partial"] = asr_partial
    if asr is not None:
        stages["listen_asr"] = asr
    if first_delta is not None:
        stages["llm_first_delta"] = first_delta
    if first_token is not None:
        stages["llm_first_token"] = first_token
    if first_tts_chunk is not None:
        stages["tts_first_chunk"] = first_tts_chunk
    if first_audio is not None:
        llm_boundary = first_delta if first_delta is not None else first_token
        if asr is not None and llm_boundary is not None:
            stages["tts_first_audio"] = max(0.0, first_audio - max(asr, llm_boundary))
        else:
            stages["tts_first_audio"] = first_audio
        stages["first_audio"] = first_audio
        stages["total"] = first_audio
    return stages


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


__all__ = ["DEFAULT_SCENARIOS", "VoiceScenario", "run_voice_chain_scenarios"]
