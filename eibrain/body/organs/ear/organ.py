"""Ear organ implementation."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from eibrain.body.ear_stream import ArecordStreamCapture, pcm_signal_stats
from eibrain.body.faster_whisper_recognizer import FasterWhisperRecognizer
from eibrain.body.organs.base import BaseOrgan
from eibrain.body.runtime_linux import transcribe_pcm_with_faster_whisper_subprocess
from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer
from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth


class EarOrgan(BaseOrgan):
    name = "ear"
    subfunction_names = ("capture", "vad", "asr")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._cache_ttl_s = self._read_float_config("capture", "refresh_interval_s", default=1.5)
        self._chunk_count = self._read_int_config("capture", "chunk_count", default=2)
        self._capture = self._build_capture()
        self._recognizer = self._build_recognizer()
        self._cached_heartbeat: OrganHealth | None = None
        self._cached_heartbeat_at = 0.0
        self._cached_driver_probes: dict[str, object] = {}
        self._cached_driver_probe_at: dict[str, float] = {}
        self._recognizer_prewarm_error = ""
        self._recognizer_prewarmed = False
        self._heartbeat_lock = threading.Lock()
        self._prewarm_recognizer()

    def heartbeat(self) -> OrganHealth:
        if not self._audio_runtime_enabled():
            return super().heartbeat()
        lock_acquired = self._heartbeat_lock.acquire(blocking=False)
        if not lock_acquired and self._cached_heartbeat is not None:
            return self._cached_heartbeat
        if not lock_acquired:
            self._heartbeat_lock.acquire()
        now_ts = time.time()
        try:
            if self._cached_heartbeat is not None and now_ts - self._cached_heartbeat_at < self._cache_ttl_s:
                return self._cached_heartbeat
            capture_state, chunks = self._capture_health(now_ts=now_ts)
            vad_state = self._vad_health(capture_state=capture_state, chunks=chunks, now_ts=now_ts)
            asr_state = self._asr_health(capture_state=capture_state, chunks=chunks, now_ts=now_ts)
            subfunctions = {
                "capture": capture_state,
                "vad": vad_state,
                "asr": asr_state,
            }
            statuses = [state.health for state in subfunctions.values()]
            if statuses and all(status == "healthy" for status in statuses):
                health = "healthy"
            elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
                health = "degraded"
            else:
                health = "unavailable"
            self._cached_heartbeat = OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)
            self._cached_heartbeat_at = now_ts
            return self._cached_heartbeat
        finally:
            self._heartbeat_lock.release()

    def passive_heartbeat(self) -> OrganHealth:
        if self._cached_heartbeat is not None:
            return self._cached_heartbeat
        subfunctions = {
            name: SubfunctionHealth(
                name=name,
                health="healthy",
                details=dict(self._passive_driver_probe(name).details),
            )
            for name in self.subfunction_names
        }
        return OrganHealth(organ=self.name, health="healthy", subfunctions=subfunctions)

    def _audio_runtime_enabled(self) -> bool:
        return (
            self._capture is not None
            and self._driver_kind("capture") != "noop"
            and self._driver_kind("asr") != "noop"
            and self._asr_provider() in {"sherpa_onnx", "faster_whisper"}
        )

    def _build_capture(self) -> ArecordStreamCapture | None:
        capture_cfg = self.config.subfunctions.get("capture")
        if capture_cfg is None or capture_cfg.driver.kind == "noop":
            return None
        return ArecordStreamCapture(
            device=str(capture_cfg.driver.extra.get("device", "default")),
            sample_rate=int(capture_cfg.driver.extra.get("sample_rate", 16000)),
            channels=int(capture_cfg.driver.extra.get("channels", 1)),
            streaming_vad=bool(capture_cfg.driver.extra.get("streaming_vad", False)),
            vad_frame_ms=int(capture_cfg.driver.extra.get("vad_frame_ms", 80)),
            vad_rms_threshold=float(capture_cfg.driver.extra.get("vad_rms_threshold", 0.028)),
            vad_min_voice_ms=int(capture_cfg.driver.extra.get("vad_min_voice_ms", 160)),
            vad_end_silence_ms=int(capture_cfg.driver.extra.get("vad_end_silence_ms", 360)),
            vad_pre_roll_ms=int(capture_cfg.driver.extra.get("vad_pre_roll_ms", 240)),
            vad_min_capture_ms=int(capture_cfg.driver.extra.get("vad_min_capture_ms", 0)),
            transcribe_vad_miss=bool(capture_cfg.driver.extra.get("transcribe_vad_miss", False)),
            vad_miss_rms_threshold=float(capture_cfg.driver.extra.get("vad_miss_rms_threshold", 0.0)),
            vad_endpoint_policy=bool(capture_cfg.driver.extra.get("vad_endpoint_policy", False)),
            vad_backend=str(capture_cfg.driver.extra.get("vad_backend", "rms")),
            vad_noise_ratio=float(capture_cfg.driver.extra.get("vad_noise_ratio", 1.18)),
            vad_silero_threshold=float(capture_cfg.driver.extra.get("vad_silero_threshold", 0.5)),
        )

    def _build_recognizer(self) -> SherpaOnnxStreamingRecognizer | FasterWhisperRecognizer | None:
        asr_cfg = self.config.subfunctions.get("asr")
        if asr_cfg is None or asr_cfg.driver.kind == "noop":
            return None
        provider = str(asr_cfg.driver.extra.get("provider", "sherpa_onnx"))
        if provider == "faster_whisper":
            extra = asr_cfg.driver.extra
            return FasterWhisperRecognizer(
                model_name=str(extra.get("model_name", "Systran/faster-whisper-tiny")),
                language=str(extra.get("language", "zh")),
                compute_type=str(extra.get("compute_type", "int8")),
                beam_size=int(extra.get("beam_size", 1)),
                vad_filter=bool(extra.get("vad_filter", False)),
                python_executable=str(extra.get("python_executable", "/usr/bin/python3")),
            )
        if provider != "sherpa_onnx":
            return None
        return SherpaOnnxStreamingRecognizer(
            model_dir=str(asr_cfg.driver.extra.get("model_dir", "")),
            model_type=str(asr_cfg.driver.extra.get("model_type", "") or "") or None,
        )

    def _capture_health(self, *, now_ts: float) -> tuple[SubfunctionHealth, list[bytes]]:
        if self._driver_kind("capture") == "noop":
            return self._subfunction_health("capture"), []
        probe = self._passive_driver_probe("capture")
        started = time.perf_counter()
        chunks: list[bytes] = []
        error = None
        if self._capture is not None:
            try:
                if hasattr(self._capture, "read_window"):
                    chunks = list(self._capture.read_window(self._chunk_count))
                else:
                    chunks = list(self._capture.read_chunks(self._chunk_count))
            except Exception as exc:  # pragma: no cover - hardware path
                error = str(exc)
        stats = pcm_signal_stats(chunks, channels=self._capture.channels if self._capture is not None else 1)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._merge_probe_details(
            probe=probe.details,
            elapsed_ms=elapsed_ms,
            status="healthy" if chunks else "capture_failed",
        )
        details.update(
            {
                "chunk_count": len(chunks),
                "requested_chunk_count": self._chunk_count,
                "sample_rate": self._capture.sample_rate if self._capture is not None else None,
                "channels": self._capture.channels if self._capture is not None else None,
                "capture_device": self._capture.device if self._capture is not None else None,
                "captured_at_ts": now_ts,
                "payload_bytes": sum(len(chunk) for chunk in chunks),
                **stats,
            }
        )
        if self._capture is not None:
            details.update(
                {
                    "capture_returncode": getattr(self._capture, "last_returncode", None),
                    "capture_stderr": getattr(self._capture, "last_stderr", ""),
                    "capture_stdout_bytes": getattr(self._capture, "last_stdout_bytes", None),
                    "capture_command": getattr(self._capture, "last_command", []),
                    "capture_retry_count": getattr(self._capture, "retry_count", None),
                    "streaming_vad": getattr(self._capture, "streaming_vad", False),
                    "vad_rms_threshold": getattr(self._capture, "vad_rms_threshold", None),
                    "vad_triggered": getattr(self._capture, "last_vad_triggered", False),
                    "vad_frame_count": getattr(self._capture, "last_vad_frame_count", None),
                    "vad_voice_frame_count": getattr(self._capture, "last_vad_voice_frame_count", None),
                    "vad_elapsed_ms": getattr(self._capture, "last_vad_elapsed_ms", None),
                }
            )
        if error:
            details["error"] = error
        elif not chunks and details.get("capture_stderr"):
            details["error"] = details["capture_stderr"]
        if chunks:
            health = "healthy"
        else:
            health = "unavailable" if probe.status == "unavailable" else "degraded"
        return SubfunctionHealth(name="capture", health=health, details=details), chunks

    def _vad_health(
        self,
        *,
        capture_state: SubfunctionHealth,
        chunks: list[bytes],
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("vad") == "noop":
            details = dict(self._subfunction_health("vad").details)
            details.update(
                {
                    "captured_at_ts": now_ts,
                    "voice_activity": bool(capture_state.details.get("voice_activity")),
                    "dbfs": capture_state.details.get("dbfs"),
                    "status": "observed" if chunks else "idle",
                    "speech_window_summary": self._summarize_audio(capture_state.details, transcript=""),
                }
            )
            health = "healthy" if chunks else "degraded"
            return SubfunctionHealth(name="vad", health=health, details=details)
        probe = self._passive_driver_probe("vad")
        return SubfunctionHealth(
            name="vad",
            health=self._normalize_status(probe.status),
            details=dict(probe.details),
        )

    def _asr_health(
        self,
        *,
        capture_state: SubfunctionHealth,
        chunks: list[bytes],
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("asr") == "noop":
            return self._subfunction_health("asr")
        started = time.perf_counter()
        transcript = ""
        error = None
        sample_count = int(capture_state.details.get("sample_count", 0) or 0)
        dbfs = float(capture_state.details.get("dbfs", -120.0) or -120.0)
        min_asr_dbfs = self._read_float_config("asr", "min_asr_dbfs", default=-30.0)
        voice_activity = bool(capture_state.details.get("voice_activity")) and dbfs >= min_asr_dbfs
        provider = self._asr_provider()
        if provider == "faster_whisper":
            probe = self._passive_asr_probe()
        else:
            probe = self._driver_probe("asr") if voice_activity else self._passive_asr_probe()
        if (
            chunks
            and voice_activity
            and self._capture is not None
        ):
            try:
                if provider == "sherpa_onnx" and self._recognizer is not None:
                    transcript = self._recognizer.transcribe(
                        chunks,
                        sample_rate=self._capture.sample_rate,
                        channels=self._capture.channels,
                    )
                    asr_cfg = self.config.subfunctions.get("asr")
                    asr_extra = asr_cfg.driver.extra if asr_cfg is not None else {}
                    transcript = self._normalize_transcript(transcript, asr_extra=asr_extra)
                elif provider == "faster_whisper":
                    asr_cfg = self.config.subfunctions.get("asr")
                    asr_extra = asr_cfg.driver.extra if asr_cfg is not None else {}
                    if isinstance(self._recognizer, FasterWhisperRecognizer):
                        transcript = self._recognizer.transcribe(
                            chunks,
                            sample_rate=self._capture.sample_rate,
                            channels=self._capture.channels,
                        )
                        transcript = self._normalize_transcript(transcript, asr_extra=asr_extra)
                    else:
                        result = transcribe_pcm_with_faster_whisper_subprocess(
                            pcm_bytes=b"".join(chunks),
                            model_name=str(asr_extra.get("model_name", "Systran/faster-whisper-tiny")),
                            sample_rate=self._capture.sample_rate,
                            channels=self._capture.channels,
                            language=str(asr_extra.get("language", "zh")),
                            compute_type=str(asr_extra.get("compute_type", "int8")),
                            beam_size=int(asr_extra.get("beam_size", 1)),
                            vad_filter=bool(asr_extra.get("vad_filter", False)),
                            python_executable=str(asr_extra.get("python_executable", "/usr/bin/python3")),
                        )
                        result_details = result.get("details", {})
                        if result.get("status") == "ok" and isinstance(result_details, dict):
                            transcript = str(result_details.get("text", "") or "")
                            transcript = self._normalize_transcript(transcript, asr_extra=asr_extra)
                        elif isinstance(result_details, dict):
                            error = str(result_details.get("stderr") or result_details.get("reason") or "faster_whisper_failed")
            except Exception as exc:  # pragma: no cover - hardware dependency
                error = str(exc)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._merge_probe_details(
            probe=probe.details,
            elapsed_ms=elapsed_ms,
            status=self._asr_status(
                transcript=transcript,
                chunks=chunks,
                raw_voice_activity=bool(capture_state.details.get("voice_activity")),
                gated_voice_activity=voice_activity,
            ),
        )
        details.update(
            {
                "captured_at_ts": now_ts,
                "transcript": transcript,
                "transcript_char_count": len(transcript),
                "voice_activity": capture_state.details.get("voice_activity"),
                "asr_voice_activity": voice_activity,
                "dbfs": capture_state.details.get("dbfs"),
                "min_asr_dbfs": min_asr_dbfs,
                "recognizer_prewarmed": self._recognizer_prewarmed,
                "recognizer_prewarm_error": self._recognizer_prewarm_error,
                "sample_count": sample_count,
                "speech_window_summary": self._summarize_audio(capture_state.details, transcript=transcript),
            }
        )
        if error:
            details["error"] = error
        if (
            isinstance(self._recognizer, SherpaOnnxStreamingRecognizer)
            and sample_count < int(self._recognizer.expected_sample_rate * 0.25)
        ):
            details["status"] = "warming_up"
            details["speech_window_summary"] = "audio window too short for ASR decode"
        if transcript:
            health = "healthy"
        elif details.get("status") == "warming_up":
            health = "degraded"
        elif chunks:
            health = "degraded"
        else:
            health = "unavailable" if probe.status == "unavailable" else "degraded"
        return SubfunctionHealth(name="asr", health=health, details=details)

    def _asr_provider(self) -> str:
        config = self.config.subfunctions.get("asr")
        if config is None:
            return "disabled"
        return str(config.driver.extra.get("provider", "sherpa_onnx"))

    def _driver_probe(self, name: str):
        now_ts = time.time()
        cached = self._cached_driver_probes.get(name)
        cached_at = self._cached_driver_probe_at.get(name, 0.0)
        if cached is not None and now_ts - cached_at < 60.0:
            return cached
        probe = self.drivers[name].heartbeat()
        self._cached_driver_probes[name] = probe
        self._cached_driver_probe_at[name] = now_ts
        return probe

    def _passive_driver_probe(self, name: str):
        if name in self._cached_driver_probes:
            return self._cached_driver_probes[name]
        return SimpleNamespace(
            status="healthy",
            details={
                "driver": self._driver_kind(name),
                "status": "live_probe_skipped",
            },
        )

    def _passive_asr_probe(self):
        return self._passive_driver_probe("asr")

    def _prewarm_recognizer(self) -> None:
        if isinstance(self._recognizer, SherpaOnnxStreamingRecognizer):
            try:
                self._recognizer._get_recognizer(self._recognizer.expected_sample_rate)
                self._recognizer_prewarmed = True
            except Exception as exc:  # pragma: no cover - host dependency
                self._recognizer_prewarm_error = str(exc)
            return
        if not isinstance(self._recognizer, FasterWhisperRecognizer):
            return

        def _load_model() -> None:
            try:
                self._recognizer._get_model()
                self._recognizer_prewarmed = True
            except Exception as exc:  # pragma: no cover - host dependency
                self._recognizer_prewarm_error = str(exc)

        threading.Thread(target=_load_model, name="faster-whisper-prewarm", daemon=True).start()

    @staticmethod
    def _normalize_transcript(transcript: str, *, asr_extra: dict[str, object]) -> str:
        normalized = transcript.strip()
        replacements = asr_extra.get("transcript_replacements", [])
        if isinstance(replacements, dict):
            replacements = [
                {"find": find, "replace": replace}
                for find, replace in replacements.items()
            ]
        if isinstance(replacements, list):
            for replacement in replacements:
                if not isinstance(replacement, dict):
                    continue
                find_text = str(replacement.get("find", ""))
                replace_text = str(replacement.get("replace", ""))
                if find_text:
                    normalized = normalized.replace(find_text, replace_text)
        return EarOrgan._compact_repeated_sentence(normalized)

    @staticmethod
    def _asr_status(
        *,
        transcript: str,
        chunks: list[bytes],
        raw_voice_activity: bool,
        gated_voice_activity: bool,
    ) -> str:
        if transcript:
            return "transcribed"
        if not chunks:
            return "capture_unavailable"
        if raw_voice_activity and not gated_voice_activity:
            return "below_asr_threshold"
        return "silence"

    @staticmethod
    def _compact_repeated_sentence(transcript: str) -> str:
        normalized = transcript.strip()
        if not normalized:
            return ""
        sentence_marks = "。！？!?"
        trailing = ""
        if normalized[-1] in sentence_marks:
            trailing = normalized[-1]
            normalized = normalized[:-1]
        parts = [
            part.strip()
            for part in normalized.replace("！", "。").replace("？", "。").replace("!", "。").replace("?", "。").split("。")
            if part.strip()
        ]
        if len(parts) > 1 and all(part == parts[0] for part in parts):
            return parts[0] + (trailing or "。")
        return transcript.strip()

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    def _read_float_config(self, subfunction_name: str, key: str, *, default: float) -> float:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return default
        value = config.driver.extra.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_int_config(self, subfunction_name: str, key: str, *, default: int) -> int:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return default
        value = config.driver.extra.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _merge_probe_details(*, probe: dict[str, object], elapsed_ms: float, status: str) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        merged["elapsed_ms"] = elapsed_ms
        merged["status"] = status
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged

    @staticmethod
    def _summarize_audio(details: dict[str, object], *, transcript: str) -> str:
        dbfs = details.get("dbfs")
        voice_activity = details.get("voice_activity")
        if transcript:
            return f"heard speech at {dbfs} dBFS: {transcript}"
        if voice_activity:
            return f"voice activity detected at {dbfs} dBFS"
        return f"no clear speech activity ({dbfs} dBFS)"
