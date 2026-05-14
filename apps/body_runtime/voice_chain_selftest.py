"""Offline voice-chain selftest for protocol and monitor readiness.

The selftest intentionally avoids real microphones, speakers, ASR, LLM, and TTS
providers. It replays a JoyInside-like streaming event path through the same
gateway and adapter classes used by the runtime so tomorrow's live test can
quickly separate code/protocol regressions from hardware or provider issues.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Iterable, Mapping
import json
import sys
from typing import Any

from apps.body_runtime.voice_chain_benchmark import DEFAULT_THRESHOLDS, summarize_voice_chain
from apps.body_runtime.voice_streaming_adapter import VoiceStreamingAdapter
from apps.head_runtime.eivoice_gateway import EiVoiceGateway
from eibrain.body.realtime_voice import RealtimeVoiceSession
from eibrain.voice.readiness import build_voice_chain_readiness


SELFTEST_SCHEMA = "eibrain.voice_chain_selftest.v1"


class _ManualClock:
    def __init__(self) -> None:
        self._now_s = 1000.0

    def __call__(self) -> float:
        return self._now_s

    def advance_ms(self, value: float) -> None:
        self._now_s += max(0.0, float(value)) / 1000.0


class _FakeCapture:
    def __init__(self, *, barge_in: bool = False) -> None:
        self.frames = [
            {
                "audio_base64": base64.b64encode(b"voice-selftest").decode("ascii"),
                "sample_rate_hz": 16000,
                "channels": 1,
                "format": "pcm16",
                "duration_ms": 60,
                "rms_dbfs": -30.0,
            }
        ]
        self.barge_in = barge_in

    def read_frame(self) -> Mapping[str, Any] | None:
        if not self.frames:
            return None
        return self.frames.pop(0)

    def probe_barge_in(self) -> Mapping[str, Any]:
        return {
            "detected": self.barge_in,
            "reason": "near_field_speech",
            "rmsDbfs": -22.0,
            "latencyMs": 190.0,
        }

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "frames_left": len(self.frames)}


class _FakePlayback:
    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        self.started = 0
        self.stops: list[str] = []

    def enqueue_chunk(self, chunk: Mapping[str, Any]) -> None:
        self.chunks.append(dict(chunk))

    def start(self) -> None:
        self.started += 1

    def stop(self, reason: str = "completed") -> None:
        self.stops.append(reason)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "buffered": len(self.chunks)}


def run_voice_chain_selftest(*, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Replay two synthetic turns and return a JSON-ready diagnostic report."""

    threshold_values = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        threshold_values.update(dict(thresholds))

    normal = _run_turn(
        name="normal_turn",
        round_id="selftest-normal-001",
        transcript="你好鸿途",
        reply="我在，语音链自检正常。",
        timings_ms={
            "audio": 90.0,
            "partial": 120.0,
            "final": 300.0,
            "reply_delta": 390.0,
            "first_speech": 410.0,
            "complete": 360.0,
        },
        interrupted=False,
    )
    interrupted = _run_turn(
        name="interrupt_turn",
        round_id="selftest-interrupt-001",
        transcript="介绍一下你自己",
        reply="我是鸿途，正在准备回答。",
        timings_ms={
            "audio": 100.0,
            "partial": 140.0,
            "final": 350.0,
            "reply_delta": 430.0,
            "first_speech": 420.0,
            "interrupt": 190.0,
            "stop": 210.0,
        },
        interrupted=True,
    )
    turns = [normal["benchmarkTurn"], interrupted["benchmarkTurn"]]
    benchmark = summarize_voice_chain(turns, thresholds=threshold_values)
    failed_metrics = [
        str(name)
        for name, metric in benchmark.get("metrics", {}).items()
        if isinstance(metric, Mapping) and metric.get("pass") is False
    ]
    round_leak_count = int(benchmark.get("roundLeakCount") or 0)
    round_leak_free = round_leak_count == 0
    streaming = benchmark.get("streaming") if isinstance(benchmark.get("streaming"), Mapping) else {}
    interrupt_stop = benchmark.get("interruptStop") if isinstance(benchmark.get("interruptStop"), Mapping) else {}
    readiness_summary = benchmark.get("readinessSummary") if isinstance(benchmark.get("readinessSummary"), Mapping) else {}
    streaming_ready = bool(streaming.get("ready"))
    interrupt_stop_ready = bool(interrupt_stop.get("ready"))
    code_ready = (
        bool(readiness_summary.get("codeReady"))
        and all(bool(turn.get("adapterApplied")) for turn in (normal, interrupted))
    )
    event_names = [*normal["eventNames"], *interrupted["eventNames"]]
    operation_counts = Counter([*normal["operations"], *interrupted["operations"]])

    readiness = build_voice_chain_readiness(
        explicit={
            "source": "offline_selftest",
            "live": False,
            "honjiaReady": False,
            "codeReady": code_ready,
            "turnCount": len(turns),
            "failedMetrics": failed_metrics,
            "roundLeakCount": round_leak_count,
            "roundLeakFree": round_leak_free,
            "roundLeakRate": benchmark.get("roundLeakRate"),
            "streamingReady": streaming_ready,
            "interruptStopReady": interrupt_stop_ready,
            "metrics": benchmark.get("metrics", {}),
            "stageLatencyMetrics": benchmark.get("stageLatencyMetrics", {}),
            "streaming": streaming,
            "interruptStop": interrupt_stop,
            "bottleneck": benchmark.get("bottleneck"),
            "readinessSummary": readiness_summary,
            "summary": "selftest ready: protocol path passed" if code_ready else "selftest not ready",
            "readinessMessage": (
                "offline protocol selftest passed; honjia live benchmark still required"
                if code_ready
                else "offline protocol selftest failed"
            ),
        }
    )
    checks = {
        "code_ready": {"ok": code_ready},
        "streaming_ready": {"ok": streaming_ready},
        "interrupt_stop_ready": {"ok": interrupt_stop_ready},
        "round_leak_free": {"ok": round_leak_free},
    }
    return {
        "schema": SELFTEST_SCHEMA,
        "source": "offline_selftest",
        "honjiaRequired": False,
        "codeReady": code_ready,
        "roundLeakFree": round_leak_free,
        "streamingReady": streaming_ready,
        "interruptStopReady": interrupt_stop_ready,
        "failedMetrics": failed_metrics,
        "thresholds": threshold_values,
        "benchmark": benchmark,
        "readiness": readiness,
        "checks": checks,
        "eventCount": len(event_names),
        "eventNames": event_names,
        "operationCounts": dict(sorted(operation_counts.items())),
        "normalTurn": _public_turn(normal),
        "interruptTurn": _public_turn(interrupted),
    }


def _run_turn(
    *,
    name: str,
    round_id: str,
    transcript: str,
    reply: str,
    timings_ms: Mapping[str, float],
    interrupted: bool,
) -> dict[str, Any]:
    clock = _ManualClock()
    capture = _FakeCapture(barge_in=interrupted)
    playback = _FakePlayback()
    session = RealtimeVoiceSession(
        session_id=f"selftest-{name}",
        actor_id="selftest-user",
        round_id=round_id,
        clock=clock,
    )
    adapter = VoiceStreamingAdapter(session)
    gateway = EiVoiceGateway(
        session_id=session.session_id,
        actor_id=session.actor_id,
        capture=capture,
        playback=playback,
        round_id=str(session.round_id),
        trace_id=f"trace-{round_id}",
    )
    session.start_listening()

    events: list[Mapping[str, Any] | object] = []
    applied: list[dict[str, Any]] = []
    event_names: list[str] = []
    operations: list[str] = []

    def apply_event(event: Mapping[str, Any] | object, *, advance_ms: float) -> None:
        clock.advance_ms(advance_ms)
        result = adapter.apply(event)  # type: ignore[arg-type]
        events.append(event)
        name = _event_name(event)
        event_names.append(name)
        operations.append(str(result["operation"]))
        applied.append(result)

    captured = gateway.capture_audio_frame()
    if captured:
        apply_event(captured[0], advance_ms=float(timings_ms["audio"]))
    apply_event(gateway.accept_asr_partial(transcript[:2]), advance_ms=float(timings_ms["partial"]))
    apply_event(gateway.accept_asr_final(transcript), advance_ms=float(timings_ms["final"]))
    apply_event(
        _round_event("ei.dialogue.agent.delta", round_id=round_id, content={"delta": reply}),
        advance_ms=float(timings_ms["reply_delta"]),
    )
    apply_event(
        _round_event(
            "ei.voice.tts.sentence_start",
            round_id=round_id,
            content={"sentenceId": "sent-1", "text": reply},
        ),
        advance_ms=float(timings_ms["first_speech"]),
    )
    apply_event(gateway.enqueue_tts_chunk("AAEC", sentence_id="sent-1"), advance_ms=20.0)
    apply_event(gateway.start_playback(), advance_ms=0.0)

    if interrupted:
        barge = gateway.probe_barge_in()
        if barge is not None:
            apply_event(barge, advance_ms=float(timings_ms["interrupt"]))
        apply_event(gateway.stop_playback(reason="user_barge_in"), advance_ms=float(timings_ms["stop"]))
    else:
        apply_event(gateway.stop_playback(reason="completed"), advance_ms=float(timings_ms["complete"]))

    snapshot = session.snapshot()
    latency = snapshot.get("latency_ms") if isinstance(snapshot.get("latency_ms"), Mapping) else {}
    stage_latency_ms = _stage_latency_ms_from_snapshot(latency)
    benchmark_turn = {
        "roundId": round_id,
        "wakeToListenMs": 0.0,
        "asrFinalMs": latency.get("final_asr", 0.0),
        "firstTokenMs": latency.get("final_asr_to_first_reply_token", 0.0),
        "firstAudioMs": latency.get("first_speech", 0.0),
        "interruptStopMs": timings_ms.get("stop", 0.0) if interrupted else 0.0,
        "stageLatencyMs": stage_latency_ms,
        "streaming": {
            "asrPartial": "ei.voice.asr.partial" in event_names,
            "llmDelta": "ei.dialogue.agent.delta" in event_names,
            "ttsChunk": "ei.voice.tts.chunk" in event_names,
            "playback": "ei.voice.playback.started" in event_names,
        },
        "roundLeak": False,
        "interrupted": interrupted,
    }
    return {
        "name": name,
        "adapterApplied": all(item.get("applied") for item in applied),
        "eventNames": event_names,
        "operations": operations,
        "snapshot": snapshot,
        "benchmarkTurn": benchmark_turn,
        "playback": {
            "started": playback.started,
            "stops": list(playback.stops),
            "chunkCount": len(playback.chunks),
        },
    }


def _stage_latency_ms_from_snapshot(latency: Mapping[str, Any]) -> dict[str, float]:
    asr = _float_or_zero(latency.get("final_asr"))
    first_token = _float_or_zero(latency.get("final_asr_to_first_reply_token"))
    first_speech = _float_or_zero(latency.get("first_speech"))
    return {
        "listen_asr": asr,
        "llm_first_token": first_token,
        "tts_first_audio": max(0.0, first_speech - asr - first_token),
        "total": first_speech,
    }


def _float_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number and number not in (float("inf"), float("-inf")) else 0.0


def _round_event(name: str, *, round_id: str, content: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "event_type": "dialogue",
        "roundId": round_id,
        "content": dict(content),
    }


def _event_name(event: Mapping[str, Any] | object) -> str:
    if isinstance(event, Mapping):
        return str(event.get("name") or "")
    return str(getattr(event, "name", "") or "")


def _public_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": turn["name"],
        "adapterApplied": turn["adapterApplied"],
        "eventNames": list(_iter_text(turn.get("eventNames"))),
        "operations": list(_iter_text(turn.get("operations"))),
        "snapshot": turn["snapshot"],
        "benchmarkTurn": turn["benchmarkTurn"],
        "playback": turn["playback"],
    }


def _iter_text(value: Any) -> Iterable[str]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield str(item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline eibrain voice-chain selftest")
    parser.add_argument("--pretty", action="store_true", help="Print indented JSON")
    args = parser.parse_args(argv)
    report = run_voice_chain_selftest()
    indent = 2 if args.pretty else None
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0 if report.get("codeReady") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["SELFTEST_SCHEMA", "run_voice_chain_selftest", "main"]
