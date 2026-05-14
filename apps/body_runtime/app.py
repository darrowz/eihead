"""Body runtime assembly for deployable configurations."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time

from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
from eibrain.body.health import DegradationManager, FallbackPolicy
from eibrain.body.state import BodyStateManager
from eibrain.body.ear_stream import EarStreamProcessor
from eibrain.body.ear_stream import ArecordStreamCapture
from eibrain.body.ear_stream import pcm_signal_stats
from eibrain.body.organs.ear.organ import EarOrgan
from eibrain.body.organs.eye.organ import EyeOrgan
from eibrain.body.organs.mouth.organ import MouthOrgan
from eibrain.body.organs.neck.organ import NeckOrgan
from eibrain.body.faster_whisper_recognizer import FasterWhisperRecognizer
from eibrain.body.neck_fusion import NeckFusionConfig, NeckFusionPolicy
from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer
from eibrain.body.vision_model_registry import select_profile as select_vision_model_profile
from eibrain.body.visual_follow_score import score_visual_follow
from eibrain.body.visual_follow_tuning import recommend_visual_follow_tuning
from eibrain.body.visual_target_lock import VisualTargetLockSelector
from eibrain.cognition.vision_events import VisionEventShaper
from eibrain.cognition.vision_scene_graph import build_vision_scene_graph
from eibrain.cognition.vision_voice_context import build_vision_voice_context
from eibrain.infra.config import EIBrainConfig, load_config
from eibrain.memory.visual_feedback import build_visual_feedback_record
from eibrain.memory.visual_memory import VisualMemoryPolicy
from eibrain.protocol.actions import Action, MoveHeadAction
from eibrain.protocol.outcomes import ActionExecuted
from eibrain.protocol.observations import AudioTranscriptFinal
from eibrain.voice.readiness import build_voice_chain_readiness
from apps.body_runtime.vision_soak import summarize_vision_soak
from apps.body_runtime.voice_chain_scenarios import run_voice_chain_scenarios


_DEFAULT_BARGE_IN_PROBE_WINDOW_S = 0.25
_MIN_BARGE_IN_PROBE_WINDOW_S = 0.05
_MAX_BARGE_IN_PROBE_WINDOW_S = 0.5


class BodyRuntimeApp:
    def __init__(self, *, config: EIBrainConfig | None = None) -> None:
        self.config = config or EIBrainConfig()
        self.organs = self._build_organs()
        self.degradation_manager = DegradationManager()
        self.body_state_manager = BodyStateManager(
            node_id=self.config.body.node_id,
            degradation_manager=self.degradation_manager,
        )
        self._recent_events: deque[dict[str, object]] = deque(maxlen=50)
        self.ear_processor: EarStreamProcessor | None = None
        self._neck_fusion = NeckFusionPolicy(self._build_neck_fusion_config())
        self._neck_fusion_last_action: dict[str, object] | None = None
        self._visual_target_lock = VisualTargetLockSelector()
        self._vision_event_shaper = VisionEventShaper(source="body_runtime.visual_tracking")
        self._visual_memory_policy = VisualMemoryPolicy()
        self._last_visual_target_error_x: float | None = None
        self._last_visual_tracking_sample: dict[str, object] | None = None
        self._last_visual_tracking_decision: dict[str, object] = {}
        self._visual_tracking_misses = 0
        self._visual_tracking_recentered_this_episode = False
        self._speech_busy_until = 0.0
        self.voice_dialogue_state: dict[str, object] = {
            "enabled": False,
            "running": False,
            "phase": "idle",
            "phase_started_at_ts": time.time(),
            "turn_count": 0,
            "last_transcript": "",
            "last_reply": "",
            "last_status": "idle",
            "last_error": "",
            "last_latency_s": {},
            "last_stage_latency_ms": {},
            "last_bottleneck_stage": "",
            "last_bottleneck_ms": None,
            "last_completed_turn": {},
            "current_phase_elapsed_s": 0.0,
            "updated_at_ts": None,
        }
        self.visual_tracking_state: dict[str, object] = {
            "running": False,
            "status": "idle",
            "source": "startup",
            "updated_at_ts": None,
            "frame_captured_at_ts": None,
            "detection_count": 0,
            "top_detection": None,
            "target": None,
            "last_outcome_status": None,
            "last_error": "",
            "miss_count": 0,
            "tracking_decision": {},
            "target_lock": {},
            "follow_score": {},
            "follow_tuning": {},
            "vision_events": [],
            "scene_graph": {},
            "voice_context": {},
            "memory_candidate": None,
            "training_feedback": None,
            "soak_summary": {},
            "model_profile": {},
            "tracking_target_center_x": None,
            "tracking_target_error_x": None,
            "tracking_suppressed_reason": "",
        }
        self._identity_registry_path = Path(".tmp-test-artifacts/identity_registry.json")
        self.identity_registry: dict[str, object] = self._load_identity_registry()
        now_ts = time.time()
        self.interaction_state: dict[str, object] = {
            "current_mode": "sleeping",
            "reason": "idle",
            "tracking_locked": False,
            "tracking_target_label": "",
            "tracking_target_score": 0.0,
            "tracking_target_x": None,
            "tracking_raw_target_x": None,
            "tracking_stable_count": 0,
            "tracking_miss_count": 0,
            "last_attention_at_ts": None,
            "last_voice_activity_at_ts": now_ts,
            "last_neck_action_at_ts": None,
            "updated_at_ts": now_ts,
        }

    @classmethod
    def from_config_path(cls, path) -> "BodyRuntimeApp":
        return cls(config=load_config(path))

    def _build_organs(self):
        organ_configs = self.config.body.organs
        organ_types = (("ear", EarOrgan), ("eye", EyeOrgan), ("mouth", MouthOrgan), ("neck", NeckOrgan))
        if not organ_configs:
            return [organ_cls() for _, organ_cls in organ_types]
        organs = []
        for organ_name, organ_cls in organ_types:
            organ_config = organ_configs.get(organ_name)
            if organ_config is None or not organ_config.enabled:
                continue
            organs.append(organ_cls(config=organ_config))
        return organs

    def _build_neck_fusion_config(self) -> NeckFusionConfig:
        neck_cfg = self.config.body.organs.get("neck")
        motor_cfg = neck_cfg.subfunctions.get("motor") if neck_cfg is not None else None
        extra = motor_cfg.driver.extra if motor_cfg is not None else {}
        config = NeckFusionConfig(consecutive_bias_required=1)
        int_fields = {
            "home_angle": "home_angle",
            "pan_min": "pan_min_angle",
            "pan_max": "pan_max_angle",
            "tracking_max_step": "max_step_degrees",
            "tracking_min_step": "min_step_degrees",
            "tracking_consecutive_bias_required": "consecutive_bias_required",
        }
        float_fields = {
            "tracking_deadband": "deadband",
            "tracking_hysteresis": "hysteresis",
            "tracking_step_gain": "pan_step_gain",
            "tracking_min_interval_s": "min_command_interval_s",
            "tracking_cooldown_s": "cooldown_s",
            "tracking_recenter_after_missing_s": "recenter_after_missing_s",
            "tracking_min_confidence": "min_confidence",
        }
        for source_key, field_name in int_fields.items():
            if source_key in extra:
                value = self._int_or_none(extra.get(source_key))
                if value is not None:
                    setattr(config, field_name, value)
        for source_key, field_name in float_fields.items():
            if source_key in extra:
                value = self._coerce_optional_float(extra.get(source_key))
                if value is not None:
                    setattr(config, field_name, value)
        return config

    def simulate_transcript(self, *, text: str, session_id: str, actor_id: str) -> AudioTranscriptFinal:
        return AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text=text,
            session_id=session_id,
            actor_id=actor_id,
        )

    def _build_ear_processor(self, *, capture, recognizer) -> EarStreamProcessor:
        return EarStreamProcessor(capture=capture, recognizer=recognizer)

    def _make_capture(self, capture_cfg):
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

    def _make_recognizer(self, asr_cfg):
        provider = str(asr_cfg.driver.extra.get("provider", "sherpa_onnx"))
        if provider == "faster_whisper":
            recognizer = FasterWhisperRecognizer(
                model_name=str(asr_cfg.driver.extra.get("model_name", "Systran/faster-whisper-tiny")),
                language=str(asr_cfg.driver.extra.get("language", "zh")),
                compute_type=str(asr_cfg.driver.extra.get("compute_type", "int8")),
                beam_size=int(asr_cfg.driver.extra.get("beam_size", 1)),
                vad_filter=bool(asr_cfg.driver.extra.get("vad_filter", False)),
                python_executable=str(asr_cfg.driver.extra.get("python_executable", "/usr/bin/python3")),
            )
            recognizer.prewarm()
            return recognizer
        recognizer = SherpaOnnxStreamingRecognizer(
            model_dir=str(asr_cfg.driver.extra.get("model_dir", "")),
            model_type=str(asr_cfg.driver.extra.get("model_type", "") or "") or None,
        )
        recognizer.prewarm()
        return recognizer

    def build_default_ear_processor(self) -> EarStreamProcessor:
        ear_cfg = self.config.body.organs.get("ear")
        if ear_cfg is None:
            raise RuntimeError("ear organ not configured")
        capture_cfg = ear_cfg.subfunctions.get("capture")
        asr_cfg = ear_cfg.subfunctions.get("asr")
        if capture_cfg is None or asr_cfg is None:
            raise RuntimeError("ear capture/asr configuration is incomplete")
        return self._build_ear_processor(
            capture=self._make_capture(capture_cfg),
            recognizer=self._make_recognizer(asr_cfg),
        )

    def transcribe_audio_window(
        self,
        *,
        chunk_count: int,
        session_id: str,
        actor_id: str,
    ) -> AudioTranscriptFinal:
        if self.is_speaking():
            return self._empty_transcript(
                session_id=session_id,
                actor_id=actor_id,
                status="speech_playback_active",
            )
        if self.ear_processor is None:
            try:
                self.ear_processor = self.build_default_ear_processor()
            except RuntimeError:
                self.ear_processor = None
        if self.ear_processor is not None:
            started = time.perf_counter()
            observation = self.ear_processor.transcribe_window(
                chunk_count=chunk_count,
                session_id=session_id,
                actor_id=actor_id,
            )
            observation = self._normalize_audio_observation(observation)
            self._record_ear_processor_event(
                observation=observation,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return observation
        ear = next((organ for organ in self.organs if organ.name == "ear"), None)
        if ear is not None and hasattr(ear, "heartbeat"):
            original_chunk_count = getattr(ear, "_chunk_count", None)
            if hasattr(ear, "_chunk_count"):
                ear._chunk_count = chunk_count
            if hasattr(ear, "_cached_heartbeat"):
                ear._cached_heartbeat = None
            try:
                heartbeat = ear.heartbeat()
            finally:
                if hasattr(ear, "_chunk_count") and original_chunk_count is not None:
                    ear._chunk_count = original_chunk_count
            asr_state = heartbeat.subfunctions.get("asr")
            capture_state = heartbeat.subfunctions.get("capture")
            details = asr_state.details if asr_state is not None else {}
            capture_details = capture_state.details if capture_state is not None else {}
            transcript = str(details.get("transcript", "") or "")
            observation = AudioTranscriptFinal(
                ts=time.time(),
                source="ear.asr",
                text=transcript,
                session_id=session_id,
                actor_id=actor_id,
            )
            self._recent_events.append(
                {
                    "kind": observation.kind,
                    "source": observation.source,
                    "status": "ok" if transcript else "degraded",
                    "session_id": session_id,
                    "recorded_at_ts": time.time(),
                    "details": {
                        "text": transcript,
                        "speech_window_summary": details.get("speech_window_summary", ""),
                        "asr_status": details.get("status"),
                        "asr_voice_activity": details.get("asr_voice_activity"),
                        "min_asr_dbfs": details.get("min_asr_dbfs"),
                        "recognizer_prewarmed": details.get("recognizer_prewarmed"),
                        "recognizer_prewarm_error": details.get("recognizer_prewarm_error"),
                        "dbfs": capture_details.get("dbfs"),
                        "rms_level": capture_details.get("rms_level"),
                        "peak_level": capture_details.get("peak_level"),
                        "payload_bytes": capture_details.get("payload_bytes"),
                        "capture_device": capture_details.get("capture_device"),
                        "sample_rate": capture_details.get("sample_rate"),
                        "channels": capture_details.get("channels"),
                        "chunk_count": capture_details.get("chunk_count"),
                        "voice_activity": capture_details.get("voice_activity"),
                        "streaming_vad": capture_details.get("streaming_vad"),
                        "vad_triggered": capture_details.get("vad_triggered"),
                        "vad_elapsed_ms": capture_details.get("vad_elapsed_ms"),
                        "captured_at_ts": capture_details.get("captured_at_ts"),
                        "asr_elapsed_ms": details.get("elapsed_ms"),
                        "asr_decode_elapsed_ms": details.get("elapsed_ms"),
                        "capture_elapsed_ms": capture_details.get("elapsed_ms"),
                        "capture_window_elapsed_ms": capture_details.get("elapsed_ms"),
                    },
                }
            )
            return observation
        self.ear_processor = self.build_default_ear_processor()
        return self.transcribe_audio_window(
            chunk_count=chunk_count,
            session_id=session_id,
            actor_id=actor_id,
        )

    def _normalize_audio_observation(self, observation: AudioTranscriptFinal) -> AudioTranscriptFinal:
        text = observation.text.strip()
        if not text:
            return observation
        asr_cfg = self.config.body.organs.get("ear")
        subfunction = asr_cfg.subfunctions.get("asr") if asr_cfg is not None else None
        replacements = subfunction.driver.extra.get("transcript_replacements", {}) if subfunction is not None else {}
        if isinstance(replacements, dict):
            for find_text, replace_text in replacements.items():
                if find_text:
                    text = text.replace(str(find_text), str(replace_text))
        elif isinstance(replacements, list):
            for replacement in replacements:
                if not isinstance(replacement, dict):
                    continue
                find_text = str(replacement.get("find", ""))
                replace_text = str(replacement.get("replace", ""))
                if find_text:
                    text = text.replace(find_text, replace_text)
        if text == observation.text:
            return observation
        return AudioTranscriptFinal(
            ts=observation.ts,
            source=observation.source,
            text=text.strip(),
            language=observation.language,
            session_id=observation.session_id,
            actor_id=observation.actor_id,
            target_id=observation.target_id,
        )

    def is_speaking(self) -> bool:
        return time.time() < self._speech_busy_until

    def probe_barge_in(self, *, session_id: str, actor_id: str) -> dict[str, object]:
        if self.ear_processor is None:
            try:
                self.ear_processor = self.build_default_ear_processor()
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                result = {
                    "detected": False,
                    "status": "not_wired",
                    "reason": str(exc),
                    "session_id": session_id,
                    "actor_id": actor_id,
                    "rms_level": 0.0,
                    "dbfs": -120.0,
                    "capture_elapsed_ms": 0.0,
                }
                self.record_runtime_event(
                    kind="voice_barge_in_probe",
                    source="body_runtime.voice_barge_in_probe",
                    status="not_wired",
                    session_id=session_id,
                    details=result,
                )
                return result

        capture = getattr(self.ear_processor, "capture", None)
        if capture is None:
            result = {
                "detected": False,
                "status": "not_wired",
                "reason": "ear_processor capture is not wired",
                "session_id": session_id,
                "actor_id": actor_id,
                "rms_level": 0.0,
                "dbfs": -120.0,
                "capture_elapsed_ms": 0.0,
            }
            self.record_runtime_event(
                kind="voice_barge_in_probe",
                source="body_runtime.voice_barge_in_probe",
                status="not_wired",
                session_id=session_id,
                details=result,
            )
            return result

        probe_window_s = float(
            getattr(capture, "barge_in_probe_duration_s", _DEFAULT_BARGE_IN_PROBE_WINDOW_S)
            or _DEFAULT_BARGE_IN_PROBE_WINDOW_S
        )
        probe_window_s = min(
            _MAX_BARGE_IN_PROBE_WINDOW_S,
            max(_MIN_BARGE_IN_PROBE_WINDOW_S, probe_window_s),
        )
        capture_reader = getattr(capture, "read_voice_window", None)
        capture_method = "read_voice_window"
        capture_arg: float | int = probe_window_s
        if not callable(capture_reader):
            capture_reader = getattr(capture, "read_window", None)
            capture_method = "read_window"
            capture_arg = probe_window_s
        if not callable(capture_reader):
            capture_reader = getattr(capture, "read_chunks", None)
            capture_method = "read_chunks"
            capture_arg = 1
        if not callable(capture_reader):
            result = {
                "detected": False,
                "status": "not_wired",
                "reason": "ear capture has no readable audio window",
                "session_id": session_id,
                "actor_id": actor_id,
                "rms_level": 0.0,
                "dbfs": -120.0,
                "capture_elapsed_ms": 0.0,
            }
            self.record_runtime_event(
                kind="voice_barge_in_probe",
                source="body_runtime.voice_barge_in_probe",
                status="not_wired",
                session_id=session_id,
                details=result,
            )
            return result

        started = time.perf_counter()
        try:
            chunks = list(capture_reader(capture_arg))
        except Exception as exc:  # pragma: no cover - hardware/runtime boundary
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            result = {
                "detected": False,
                "status": "capture_error",
                "reason": str(exc),
                "session_id": session_id,
                "actor_id": actor_id,
                "rms_level": 0.0,
                "dbfs": -120.0,
                "capture_elapsed_ms": elapsed_ms,
                "capture_method": capture_method,
                "probe_window_s": probe_window_s,
            }
            self.record_runtime_event(
                kind="voice_barge_in_probe",
                source="body_runtime.voice_barge_in_probe",
                status="capture_error",
                session_id=session_id,
                details=result,
            )
            return result
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        channels = int(getattr(capture, "channels", 1) or 1)
        stats = pcm_signal_stats(chunks, channels=channels)
        rms_level = float(stats.get("rms_level", 0.0) or 0.0)
        dbfs = float(stats.get("dbfs", -120.0) or -120.0)
        voice_activity = bool(stats.get("voice_activity", False))
        rms_threshold = max(0.015, float(getattr(capture, "vad_rms_threshold", 0.015) or 0.015))
        dbfs_threshold = float(getattr(capture, "barge_in_min_dbfs", -42.0) or -42.0)
        detected = bool(chunks) and voice_activity and rms_level >= rms_threshold and dbfs >= dbfs_threshold
        status = "detected" if detected else ("no_audio" if not chunks else "clear")
        reason = (
            "voice_activity_above_threshold"
            if detected
            else ("no_audio_captured" if not chunks else "below_threshold")
        )
        result = {
            "detected": detected,
            "status": status,
            "reason": reason,
            "session_id": session_id,
            "actor_id": actor_id,
            "rms_level": rms_level,
            "dbfs": dbfs,
            "voice_activity": voice_activity,
            "rms_threshold": rms_threshold,
            "dbfs_threshold": dbfs_threshold,
            "capture_elapsed_ms": elapsed_ms,
            "capture_method": capture_method,
            "probe_window_s": probe_window_s,
            "chunk_count": len(chunks),
            "payload_bytes": sum(len(chunk) for chunk in chunks),
            "sample_rate": getattr(capture, "sample_rate", None),
            "channels": channels,
        }
        self.record_runtime_event(
            kind="voice_barge_in_probe",
            source="body_runtime.voice_barge_in_probe",
            status=status,
            session_id=session_id,
            details=result,
        )
        return result

    def _empty_transcript(self, *, session_id: str, actor_id: str, status: str) -> AudioTranscriptFinal:
        observation = AudioTranscriptFinal(
            ts=time.time(),
            source="ear.asr",
            text="",
            session_id=session_id,
            actor_id=actor_id,
        )
        self._recent_events.append(
            {
                "kind": observation.kind,
                "source": observation.source,
                "status": status,
                "session_id": session_id,
                "recorded_at_ts": time.time(),
                "details": {"text": "", "speech_window_summary": status},
            }
        )
        return observation

    def _record_ear_processor_event(self, *, observation: AudioTranscriptFinal, elapsed_ms: float) -> None:
        capture = getattr(self.ear_processor, "capture", None)
        recognizer = getattr(self.ear_processor, "recognizer", None)
        chunks = list(getattr(capture, "last_chunks", []) or [])
        stats = pcm_signal_stats(chunks, channels=int(getattr(capture, "channels", 1) or 1)) if chunks else {}
        text = observation.text.strip()
        self._recent_events.append(
            {
                "kind": observation.kind,
                "source": observation.source,
                "status": "ok" if text else "degraded",
                "session_id": observation.session_id,
                "recorded_at_ts": time.time(),
                "details": {
                    "text": text,
                    "speech_window_summary": "transcribed speech" if text else "no vad speech trigger",
                    "asr_status": "transcribed" if text else "silence",
                    "asr_voice_activity": bool(text),
                    "recognizer_prewarmed": bool(getattr(recognizer, "prewarmed", False)),
                    "recognizer_prewarm_error": getattr(recognizer, "prewarm_error", ""),
                    "dbfs": stats.get("dbfs"),
                    "rms_level": stats.get("rms_level"),
                    "peak_level": stats.get("peak_level"),
                    "payload_bytes": sum(len(chunk) for chunk in chunks),
                    "capture_device": getattr(capture, "device", None),
                    "sample_rate": getattr(capture, "sample_rate", None),
                    "channels": getattr(capture, "channels", None),
                    "chunk_count": len(chunks),
                    "voice_activity": stats.get("voice_activity", False),
                    "streaming_vad": getattr(capture, "streaming_vad", False),
                    "vad_triggered": getattr(capture, "last_vad_triggered", None),
                    "vad_elapsed_ms": getattr(capture, "last_vad_elapsed_ms", None),
                    "vad_backend": getattr(capture, "last_vad_backend", None),
                    "vad_threshold": getattr(capture, "last_vad_threshold", None),
                    "vad_noise_floor": getattr(capture, "last_vad_noise_floor", None),
                    "vad_error": getattr(capture, "last_vad_error", ""),
                    "capture_elapsed_ms": getattr(self.ear_processor, "last_capture_elapsed_ms", None),
                    "capture_chunk_count": getattr(capture, "last_capture_chunk_count", None),
                    "captured_at_ts": time.time(),
                    "asr_elapsed_ms": getattr(self.ear_processor, "last_transcribe_elapsed_ms", elapsed_ms),
                    "asr_decode_elapsed_ms": getattr(self.ear_processor, "last_decode_elapsed_ms", None),
                },
            }
        )

    def update_voice_dialogue_state(self, **updates: object) -> None:
        phase = updates.get("phase")
        if phase is not None and phase != self.voice_dialogue_state.get("phase"):
            updates.setdefault("phase_started_at_ts", time.time())
        self.voice_dialogue_state.update(updates)
        phase_started_at_ts = self.voice_dialogue_state.get("phase_started_at_ts")
        if isinstance(phase_started_at_ts, (int, float)):
            self.voice_dialogue_state["current_phase_elapsed_s"] = round(time.time() - float(phase_started_at_ts), 2)
        self.voice_dialogue_state["updated_at_ts"] = time.time()
        if updates.get("last_transcript") or updates.get("last_reply") or updates.get("running"):
            self._note_voice_activity()
        self._refresh_interaction_mode()

    def voice_status(self) -> dict[str, object]:
        return self.voice_realtime()

    def voice_realtime(self) -> dict[str, object]:
        dialogue = self._voice_dialogue_payload()
        realtime_session = self._voice_realtime_session_payload(dialogue)
        round_info = self._voice_round_payload(dialogue, realtime_session)
        scheduler_state = self._voice_scheduler_state_payload(dialogue)
        cognition = self._voice_realtime_cognition_payloads(dialogue, realtime_session, scheduler_state)
        interruption = self._voice_interruption_payload(dialogue, realtime_session)
        cancellation_chain = self._voice_cancellation_chain_payload(dialogue, realtime_session)
        recent_events = self._recent_voice_events()
        ear = self._voice_organ_payload("ear", recent_events=recent_events)
        mouth = self._voice_organ_payload("mouth", recent_events=recent_events)
        latency = self._voice_latency_payload(dialogue)
        self._merge_voice_realtime_latency(latency, realtime_session)
        bottleneck = self._voice_bottleneck_payload(dialogue, latency=latency)
        last_turn = self._voice_last_turn(dialogue)
        voice_chain_readiness = self._voice_chain_readiness_payload(dialogue)
        realtime_audio = self._first_mapping_from(dialogue, keys=("realtime_audio", "realtimeAudio")) or {
            "enabled": False,
            "running": False,
        }
        status, wired, not_wired = self._voice_overall_status(
            ear=ear,
            mouth=mouth,
            dialogue=dialogue,
            scheduler=scheduler_state,
            interruption=interruption,
            latency=latency,
            last_turn=last_turn,
        )
        readiness_message = self._voice_readiness_message(
            status=status,
            ear=ear,
            mouth=mouth,
            dialogue=dialogue,
            scheduler=scheduler_state,
            interruption=interruption,
        )
        return {
            "schema": "eihead.monitor.voice_realtime.v1",
            "status": status,
            "wired": wired,
            "source": "body_runtime.voice_realtime",
            "channel": "voice.realtime",
            "aliases": ["audio.realtime"],
            "captured_at_ts": time.time(),
            "ear": ear,
            "mouth": mouth,
            "dialogue": dialogue,
            "voice_dialogue": dialogue,
            "realtime_audio": realtime_audio,
            "realtime_session": realtime_session,
            "round": round_info,
            "current_round_id": round_info["current_round_id"],
            "current_cancellation_token": round_info["current_cancellation_token"],
            "scheduler_state": scheduler_state,
            "lanes": cognition["lanes"],
            "fast_think": cognition["fast_think"],
            "slow_reasoner": cognition["slow_reasoner"],
            "arbiter": cognition["arbiter"],
            "speech_action_plan": cognition["speech_action_plan"],
            "proactive_activity": cognition["proactive_activity"],
            "interruption": interruption,
            "last_interrupt": interruption["last_interrupt"],
            "interrupt_count": interruption["interrupt_count"],
            "interrupted_round_count": interruption["interrupted_round_count"],
            "interrupt_active": interruption["active"],
            "cancellation_chain": cancellation_chain,
            "latency": latency,
            "last_stage_latency_ms": dict(latency.get("stage_latency_ms", {}) or {}),
            "last_latency_s": dict(latency.get("stage_latency_s", {}) or {}),
            "bottleneck": bottleneck,
            "last_bottleneck_stage": (bottleneck or {}).get("stage"),
            "last_bottleneck_ms": (bottleneck or {}).get("latency_ms"),
            "last_turn": last_turn,
            "voice_chain_readiness": voice_chain_readiness,
            "not_wired": not_wired,
            "readiness_message": readiness_message,
            "recent_events": recent_events,
        }

    def request_voice_interrupt(self, reason: str = "user_barge_in") -> dict[str, object]:
        requested_at_ts = time.time()
        was_busy = self.is_speaking()
        action = StopSpeechAction(
            ts=requested_at_ts,
            source="body_runtime.voice_interrupt",
            session_id="voice-runtime",
            actor_id="user",
            reason=reason,
            details={"reason": reason},
        )
        mouth = next((organ for organ in self.organs if getattr(organ, "name", None) == "mouth"), None)
        outcomes = []
        error = ""
        if mouth is not None:
            try:
                outcomes = self.dispatch_actions([action])
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                error = str(exc)
        outcome_payloads = [self._json_ready(outcome) for outcome in outcomes]
        outcome_statuses = [str(getattr(outcome, "status", "") or "") for outcome in outcomes]
        ok_statuses = {"ok", "healthy", "completed", "stopped"}
        status = "ok" if mouth is not None and outcomes and all(s in ok_statuses for s in outcome_statuses) else "degraded"
        if error:
            status = "degraded"
        busy_cleared = status == "ok"
        if busy_cleared:
            self._speech_busy_until = 0.0
        summary: dict[str, object] = {
            "status": status,
            "reason": reason,
            "requested_at_ts": requested_at_ts,
            "busy_cleared": busy_cleared,
            "busy_retained": was_busy and not busy_cleared,
            "was_busy": was_busy,
            "mouth_available": mouth is not None,
            "outcome_count": len(outcomes),
            "outcomes": outcome_payloads,
        }
        if error:
            summary["error"] = error
        self.update_voice_dialogue_state(
            phase="interrupted",
            last_status="interrupted" if status == "ok" else "interrupt_degraded",
            interrupt_active=True,
            interrupt=dict(summary),
            last_interrupt=dict(summary),
        )
        self.record_runtime_event(
            kind="voice_interrupt_requested",
            source="body_runtime.voice_interrupt",
            status=status,
            session_id="voice-runtime",
            details=summary,
        )
        return summary

    def plan_visual_tracking_action(
        self,
        *,
        target_name: str,
        target_x: float | None,
        score: float = 1.0,
        session_id: str,
        actor_id: str,
        now_ts: float | None = None,
    ) -> MoveHeadAction | None:
        current_angle = self._neck_home_angle()
        if isinstance(self._neck_fusion_last_action, dict):
            try:
                current_angle = int(self._neck_fusion_last_action.get("target_angle") or current_angle)
            except (TypeError, ValueError):
                current_angle = self._neck_home_angle()
        recommendation = self._neck_fusion.recommend(
            target_x=target_x,
            score=float(score),
            current_angle=current_angle,
            last_action=self._neck_fusion_last_action,
            now_ts=time.time() if now_ts is None else float(now_ts),
        )
        self._neck_fusion_last_action = dict(recommendation.last_action)
        target_error_x = None if target_x is None else round(float(target_x) - 0.5, 4)
        self._last_visual_tracking_decision = {
            "action": recommendation.action,
            "reason": recommendation.reason,
            "target_angle": recommendation.target_angle,
            "current_angle": current_angle,
            "target_x": None if target_x is None else round(float(target_x), 4),
            "target_error_x": target_error_x,
            "score": round(float(score), 4),
            "deadband": self._neck_fusion.config.deadband,
            "hysteresis": self._neck_fusion.config.hysteresis,
            "min_command_interval_s": self._neck_fusion.config.min_command_interval_s,
            "cooldown_s": self._neck_fusion.config.cooldown_s,
            "max_step_degrees": self._neck_fusion.config.max_step_degrees,
            "bias_direction": recommendation.last_action.get("bias_direction"),
            "bias_count": recommendation.last_action.get("bias_count"),
            "acted_at_ts": recommendation.last_action.get("acted_at_ts"),
        }
        if recommendation.action == "hold":
            return None
        return MoveHeadAction(
            ts=1.0,
            source="eye.tracking",
            session_id=session_id,
            actor_id=actor_id,
            target_name="recenter" if recommendation.action == "recenter" else target_name,
            target_x=target_x,
            target_angle=recommendation.target_angle,
        )

    def track_visual_target_once(
        self,
        *,
        preferred_labels: tuple[str, ...] = ("face", "person"),
        recenter_after_misses: int = 3,
        session_id: str = "tracking-session",
        actor_id: str = "vision-runtime",
        source: str = "tracking-loop",
    ):
        target, eye_details = self._select_visual_tracking_target(preferred_labels=preferred_labels)
        self._update_visual_tracking_state(
            running=True,
            source=source,
            frame_captured_at_ts=eye_details.get("frame_captured_at_ts"),
            detection_count=eye_details.get("detection_count", 0),
            top_detection=eye_details.get("top_detection"),
        )
        if target is None:
            self._visual_tracking_misses += 1
            self._note_visual_miss()
            diagnostics = self._build_visual_loop_diagnostics(
                eye_details=eye_details,
                detections=[],
                selected_target=None,
                tracking_target=None,
                outcome=None,
                session_id=session_id,
                actor_id=actor_id,
                status="waiting_for_target",
                suppressed_reason="target_missing",
            )
            self._update_visual_tracking_state(
                status="waiting_for_target",
                target=None,
                miss_count=self._visual_tracking_misses,
                last_outcome_status=None,
                tracking_decision={
                    "action": "hold",
                    "reason": "target_missing",
                    "target_angle": self._neck_home_angle(),
                    "current_angle": self._neck_home_angle(),
                },
                tracking_target_center_x=None,
                tracking_target_error_x=None,
                tracking_suppressed_reason="target_missing",
                **diagnostics,
            )
            if self._visual_tracking_recentered_this_episode:
                return None
            if self._visual_tracking_misses < recenter_after_misses:
                return None
            action = MoveHeadAction(
                ts=1.0,
                source="eye.tracking",
                session_id=session_id,
                actor_id=actor_id,
                target_name="recenter",
                target_angle=self._neck_home_angle(),
            )
            self._visual_tracking_recentered_this_episode = True
            outcomes = self.dispatch_actions([action])
            outcome = outcomes[0] if outcomes else self._empty_action_outcome(action, reason="recenter_dispatch_empty")
            self._update_visual_tracking_state(
                status="recentering",
                target={"label": "recenter", "target_angle": self._neck_home_angle()},
                miss_count=self._visual_tracking_misses,
                last_outcome_status=getattr(outcome, "status", None),
                tracking_decision={
                    "action": "recenter",
                    "reason": "recenter_after_miss",
                    "target_angle": self._neck_home_angle(),
                    "current_angle": self._neck_home_angle(),
                },
                tracking_target_center_x=None,
                tracking_target_error_x=None,
                tracking_suppressed_reason="",
                **self._build_visual_loop_diagnostics(
                    eye_details=eye_details,
                    detections=[],
                    selected_target=None,
                    tracking_target=None,
                    outcome=outcome,
                    session_id=session_id,
                    actor_id=actor_id,
                    status="recentering",
                    suppressed_reason="",
                ),
            )
            self._refresh_interaction_mode(force_reason="recenter_after_miss")
            return outcome
        self._visual_tracking_misses = 0
        self._visual_tracking_recentered_this_episode = False
        tracking_target = self._prepare_tracking_target(target)
        if tracking_target is None:
            diagnostics = self._build_visual_loop_diagnostics(
                eye_details=eye_details,
                detections=self._detections_from_eye_details(eye_details),
                selected_target=target,
                tracking_target=None,
                outcome=None,
                session_id=session_id,
                actor_id=actor_id,
                status="holding_target",
                suppressed_reason="target_preparation_suppressed",
            )
            self._update_visual_tracking_state(
                status="holding_target",
                target=target,
                miss_count=0,
                last_outcome_status=None,
                tracking_decision={
                    "action": "hold",
                    "reason": "target_preparation_suppressed",
                },
                tracking_target_center_x=target.get("target_x"),
                tracking_target_error_x=self._target_error_x(target.get("target_x")),
                tracking_suppressed_reason="target_preparation_suppressed",
                **diagnostics,
            )
            return None
        action = self.plan_visual_tracking_action(
            target_name=str(tracking_target.get("label", "target")),
            target_x=float(tracking_target["target_x"]),
            score=self._coerce_float(tracking_target.get("score"), default=0.0),
            session_id=session_id,
            actor_id=actor_id,
        )
        if action is None:
            self._note_visual_target_locked(tracking_target, neck_action=False)
            diagnostics = self._build_visual_loop_diagnostics(
                eye_details=eye_details,
                detections=self._detections_from_eye_details(eye_details),
                selected_target=target,
                tracking_target=tracking_target,
                outcome=None,
                session_id=session_id,
                actor_id=actor_id,
                status="holding_target",
                suppressed_reason=str(self._last_visual_tracking_decision.get("reason", "")),
            )
            self._update_visual_tracking_state(
                status="holding_target",
                target=tracking_target,
                miss_count=0,
                last_outcome_status=None,
                tracking_decision=dict(self._last_visual_tracking_decision),
                tracking_target_center_x=tracking_target.get("target_x"),
                tracking_target_error_x=self._target_error_x(tracking_target.get("target_x")),
                tracking_suppressed_reason=str(self._last_visual_tracking_decision.get("reason", "")),
                **diagnostics,
            )
            return None
        outcomes = self.dispatch_actions([action])
        outcome = outcomes[0] if outcomes else None
        self._note_visual_target_locked(tracking_target)
        diagnostics = self._build_visual_loop_diagnostics(
            eye_details=eye_details,
            detections=self._detections_from_eye_details(eye_details),
            selected_target=target,
            tracking_target=tracking_target,
            outcome=outcome,
            session_id=session_id,
            actor_id=actor_id,
            status="tracking",
            suppressed_reason="",
        )
        self._update_visual_tracking_state(
            status="tracking",
            target=tracking_target,
            miss_count=0,
            last_outcome_status=getattr(outcome, "status", None),
            tracking_decision=dict(self._last_visual_tracking_decision),
            tracking_target_center_x=tracking_target.get("target_x"),
            tracking_target_error_x=self._target_error_x(tracking_target.get("target_x")),
            tracking_suppressed_reason="",
            **diagnostics,
        )
        return outcome

    def pause_visual_tracking(self, *, reason: str) -> None:
        self._visual_tracking_misses = 0
        self._visual_tracking_recentered_this_episode = False
        self._update_visual_tracking_state(
            running=False,
            status="idle",
            source="engagement_gate",
            target=None,
            miss_count=0,
            detection_count=0,
            top_detection=None,
            last_outcome_status=None,
            last_error="",
            tracking_decision={},
            target_lock={},
            follow_score={},
            follow_tuning={},
            vision_events=[],
            scene_graph={},
            voice_context={},
            memory_candidate=None,
            training_feedback=None,
            soak_summary={},
            tracking_target_center_x=None,
            tracking_target_error_x=None,
            tracking_suppressed_reason=reason,
        )
        self.interaction_state.update(
            {
                "tracking_locked": False,
                "tracking_target_label": "",
                "tracking_target_score": 0.0,
                "tracking_target_x": None,
                "tracking_raw_target_x": None,
                "tracking_stable_count": 0,
                "tracking_miss_count": 0,
            }
        )
        neck = self._get_neck_organ()
        if neck is not None and hasattr(neck, "clear_neck_control"):
            neck.clear_neck_control(reason=reason)
        self._refresh_interaction_mode(force_reason=reason or "visual_pause")

    def _get_neck_organ(self):
        return next((organ for organ in self.organs if organ.name == "neck"), None)

    def snapshot(self) -> dict[str, object]:
        organ_states = [
            self._snapshot_organ(organ)
            for organ in self.organs
        ]
        degradation = self.degradation_manager.evaluate(organ_states)
        fallback_policy = FallbackPolicy.from_capabilities(
            degradation.capabilities,
            degradation_mode=degradation.degradation_mode,
        )
        runtime_sections = {
            "voice_dialogue": dict(self.voice_dialogue_state),
            "visual_tracking": dict(self.visual_tracking_state),
            "interaction_state": dict(self.interaction_state),
            "identity_registry": dict(self.identity_registry),
        }
        body_state = self.body_state_manager.snapshot(
            organ_states,
            recent_events=list(self._recent_events),
            runtime=runtime_sections,
        )
        return {
            "node_id": self.config.body.node_id,
            "organ_count": len(organ_states),
            "degradation_mode": degradation.degradation_mode,
            "capabilities": degradation.capabilities.to_dict(),
            "fallback_policy": fallback_policy.to_dict(),
            "body_state": body_state,
            "organs": {state.organ: state.to_dict() for state in organ_states},
            "recent_event_count": len(self._recent_events),
            "voice_dialogue": dict(self.voice_dialogue_state),
            "visual_tracking": dict(self.visual_tracking_state),
            "interaction_state": dict(self.interaction_state),
            "identity_registry": dict(self.identity_registry),
        }

    def status(self) -> dict[str, object]:
        return {
            "ok": True,
            "status": "ok",
            "runtime": "body_runtime",
            "node_id": self.config.body.node_id,
            "overall_status": "online",
            "recent_event_count": len(self._recent_events),
            "voice_dialogue": dict(self.voice_dialogue_state),
            "visual_tracking": dict(self.visual_tracking_state),
            "interaction_state": dict(self.interaction_state),
        }

    def capabilities(self) -> dict[str, object]:
        capabilities = {}
        for organ in self.organs:
            name = str(getattr(organ, "name", "") or organ.__class__.__name__)
            capabilities[name] = {
                "status": "wired",
                "kind": name,
                "class": organ.__class__.__name__,
            }
        return {
            "schema": "body_runtime.capabilities.v1",
            "node_id": self.config.body.node_id,
            "summary": {
                "online": len(capabilities),
                "degraded": 0,
                "offline": 0,
                "total": len(capabilities),
            },
            "capabilities": capabilities,
        }

    def register_current_identity(
        self,
        *,
        display_name: str = "Darrow",
        actor_id: str = "darrow",
    ) -> dict[str, object]:
        name = (display_name or "").strip() or "Darrow"
        actor = (actor_id or "").strip() or name.lower()
        target = self._current_identity_target()
        if target is None:
            if self.identity_registry.get("registered"):
                return {"ok": True, "status": "already_registered", "identity": dict(self.identity_registry)}
            result = {
                "ok": False,
                "status": "no_visual_target",
                "display_name": name,
                "actor_id": actor,
            }
            self.record_runtime_event(
                kind="identity_registration",
                source="eye.identity",
                status="degraded",
                details=result,
            )
            return result

        profile = {
            "registered": True,
            "actor_id": actor,
            "display_name": name,
            "registered_at_ts": time.time(),
            "source": "visual_tracking",
            "target": dict(target),
        }
        self.identity_registry = profile
        self._save_identity_registry(profile)
        self.interaction_state.update(
            {
                "recognized_actor_id": actor,
                "recognized_display_name": name,
                "updated_at_ts": time.time(),
            }
        )
        self.record_runtime_event(
            kind="identity_registered",
            source="eye.identity",
            status="ok",
            details=profile,
        )
        return {"ok": True, "status": "registered", "identity": dict(profile)}

    @staticmethod
    def _empty_action_outcome(action: Action, *, reason: str) -> ActionExecuted:
        return ActionExecuted(
            ts=getattr(action, "ts", 0.0),
            source="body_runtime.dispatch",
            status="skipped",
            session_id=str(getattr(action, "session_id", "") or ""),
            actor_id=str(getattr(action, "actor_id", "") or ""),
            target_id=str(getattr(action, "target_id", "") or ""),
            action_kind=str(getattr(action, "kind", "") or ""),
            details={"reason": reason},
        )

    def dispatch_actions(self, actions: list[Action]) -> list:
        outcomes = []
        for action in actions:
            for organ in self.organs:
                if organ.supports_action(action):
                    if isinstance(action, PlaySpeechAction):
                        self._speech_busy_until = time.time() + 120.0
                        self._note_voice_activity()
                        self._refresh_interaction_mode(force_mode="responding", force_reason="speaking")
                    try:
                        outcome = organ.handle_action(action)
                    finally:
                        if isinstance(action, PlaySpeechAction):
                            self._speech_busy_until = time.time() + 0.75
                    if outcome is not None:
                        outcomes.append(outcome)
                        details = dict(getattr(outcome, "details", {}) or {})
                        action_kind = str(getattr(outcome, "action_kind", "") or getattr(action, "kind", "") or "")
                        if action_kind and "action_kind" not in details:
                            details["action_kind"] = action_kind
                        self._recent_events.append(
                            {
                                "kind": outcome.kind,
                                "source": outcome.source,
                                "status": outcome.status,
                                "session_id": outcome.session_id,
                                "action_kind": action_kind,
                                "recorded_at_ts": time.time(),
                                "details": details,
                            }
                        )
                    break
        self._refresh_interaction_mode()
        return outcomes

    def record_runtime_event(
        self,
        *,
        kind: str,
        source: str,
        status: str,
        session_id: str = "runtime",
        details: dict[str, object] | None = None,
    ) -> None:
        self._recent_events.append(
            {
                "kind": kind,
                "source": source,
                "status": status,
                "session_id": session_id,
                "recorded_at_ts": time.time(),
                "details": dict(details or {}),
            }
        )
        if source == "eye.tracking" and status == "error":
            self._update_visual_tracking_state(
                running=True,
                status="error",
                last_error=str((details or {}).get("error", "") or ""),
            )

    def _snapshot_organ(self, organ):
        if organ.name == "ear" and self.voice_dialogue_state.get("running"):
            if hasattr(organ, "passive_heartbeat"):
                return organ.passive_heartbeat()
        if organ.name == "eye" and self.voice_dialogue_state.get("running"):
            if hasattr(organ, "passive_heartbeat"):
                return organ.passive_heartbeat()
        return organ.heartbeat()

    def recent_events(self) -> list[dict[str, object]]:
        return list(self._recent_events)

    def recent_actions(self) -> list[dict[str, object]]:
        actions: list[dict[str, object]] = []
        for event in self._recent_events:
            kind = str(event.get("kind", "") or "")
            source = str(event.get("source", "") or "")
            action_kind = str(event.get("action_kind", "") or "")
            details = event.get("details")
            detail_action_kind = ""
            if isinstance(details, dict):
                detail_action_kind = str(details.get("action_kind", "") or "")
            if not (action_kind or detail_action_kind or "action" in kind or source.startswith(("mouth", "neck"))):
                continue
            payload = self._json_ready(event)
            if isinstance(payload, dict):
                normalized = dict(payload)
                normalized.setdefault("action_kind", action_kind or detail_action_kind or kind)
                actions.append(normalized)
        return actions[-20:]

    def latest_visual_frame_path(self) -> str | None:
        for organ in self.organs:
            frame_path = getattr(organ, "latest_frame_path", None)
            if isinstance(frame_path, str) and frame_path:
                return frame_path
        return None

    def _current_identity_target(self) -> dict[str, object] | None:
        target = self.visual_tracking_state.get("target")
        if isinstance(target, dict) and target.get("label") and target.get("bbox"):
            return dict(target)
        target, _eye_details = self._select_visual_tracking_target(preferred_labels=("face", "person"))
        if target is not None:
            return dict(target)
        return None

    def _load_identity_registry(self) -> dict[str, object]:
        default = {
            "registered": False,
            "actor_id": "",
            "display_name": "",
            "registered_at_ts": None,
            "source": "",
            "target": None,
        }
        try:
            if self._identity_registry_path.exists():
                payload = json.loads(self._identity_registry_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return {**default, **payload}
        except (OSError, json.JSONDecodeError):
            pass
        return default

    def _save_identity_registry(self, profile: dict[str, object]) -> None:
        try:
            self._identity_registry_path.parent.mkdir(parents=True, exist_ok=True)
            self._identity_registry_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self.record_runtime_event(
                kind="identity_registry_persist",
                source="eye.identity",
                status="degraded",
                details={"error": str(exc)},
            )

    def _select_visual_tracking_target(
        self,
        *,
        preferred_labels: tuple[str, ...],
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        eye = next((organ for organ in self.organs if organ.name == "eye"), None)
        if eye is None:
            return None, {}
        heartbeat = eye.heartbeat()
        detection_state = heartbeat.subfunctions.get("detection")
        if detection_state is None:
            return None, {}
        detections = detection_state.details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        eye_details = {
            "frame_captured_at_ts": detection_state.details.get("frame_captured_at_ts"),
            "frame_path": detection_state.details.get("frame_path"),
            "fps": detection_state.details.get("fps") or detection_state.details.get("vision_fps"),
            "target_fps": detection_state.details.get("target_fps") or detection_state.details.get("vision_target_fps"),
            "detection_count": len(detections),
            "top_detection": detection_state.details.get("top_detection"),
            "detections": detections,
        }
        ranked: list[tuple[int, float, dict[str, object]]] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox", {})
            if not isinstance(bbox, dict):
                continue
            try:
                x_min = float(bbox.get("x_min", 0.0))
                x_max = float(bbox.get("x_max", 0.0))
                score = float(detection.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            label = str(detection.get("label", "target"))
            priority = preferred_labels.index(label) if label in preferred_labels else len(preferred_labels)
            ranked.append(
                (
                    priority,
                    -score,
                    {
                        "label": label,
                        "score": score,
                        "target_x": max(0.0, min(1.0, (x_min + x_max) / 2.0)),
                        "bbox": bbox,
                    },
                )
            )
        if ranked and all(item[0] >= len(preferred_labels) for item in ranked):
            return None, eye_details
        if not ranked:
            return None, eye_details
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[0][2], eye_details

    def _neck_home_angle(self) -> int:
        neck = next((organ for organ in self.organs if organ.name == "neck"), None)
        if neck is None:
            return 90
        motor = getattr(neck.config, "subfunctions", {}).get("motor") if getattr(neck, "config", None) else None
        extra = motor.driver.extra if motor is not None else {}
        try:
            return int(extra.get("home_angle", 90))
        except (TypeError, ValueError):
            return 90

    def _update_visual_tracking_state(self, **updates: object) -> None:
        self.visual_tracking_state.update(updates)
        self.visual_tracking_state["updated_at_ts"] = time.time()
        if "last_error" not in updates and self.visual_tracking_state.get("status") != "error":
            self.visual_tracking_state["last_error"] = ""

    def _build_visual_loop_diagnostics(
        self,
        *,
        eye_details: dict[str, object],
        detections: list[dict[str, object]],
        selected_target: dict[str, object] | None,
        tracking_target: dict[str, object] | None,
        outcome: object | None,
        session_id: str,
        actor_id: str,
        status: str,
        suppressed_reason: str,
    ) -> dict[str, object]:
        now_ts = time.time()
        target = tracking_target or selected_target
        current_error = self._target_error_x(target.get("target_x")) if isinstance(target, dict) else None
        previous_error = self._last_visual_target_error_x
        decision = dict(self._last_visual_tracking_decision)
        command_delta = self._decision_command_delta(decision)
        frame_ts = eye_details.get("frame_captured_at_ts")
        target_age_s = (
            max(0.0, now_ts - float(frame_ts))
            if isinstance(frame_ts, (int, float))
            else None
        )
        outcome_status = str(getattr(outcome, "status", "") or "")
        loop_elapsed_s = self._loop_elapsed_s()
        follow_score = score_visual_follow(
            before_error=previous_error if previous_error is not None else current_error,
            after_error=current_error,
            command_angle_delta=command_delta,
            target_age_s=target_age_s,
            action_elapsed_s=loop_elapsed_s,
            settle_time_s=loop_elapsed_s,
            suppressed=bool(suppressed_reason and suppressed_reason not in {"", "tracking", "none"}),
            suppressed_reason=suppressed_reason or None,
            held=decision.get("action") == "hold" or tracking_target is None,
        ).to_dict()
        self._last_visual_target_error_x = current_error

        lock_result = self._visual_target_lock.select(
            self._lock_input_detections(detections, target=target),
            now_ts=now_ts,
        ).to_dict()
        scene_graph = build_vision_scene_graph(
            detections=detections,
            tracks=[lock_result["target"]] if isinstance(lock_result.get("target"), dict) else None,
            frame_metadata={
                "frameId": eye_details.get("frame_id") or eye_details.get("frame_captured_at_ts") or "",
                "observedAt": frame_ts or now_ts,
                "imageUrl": eye_details.get("frame_path") or "",
            },
        )
        vision_events = self._vision_event_shaper.shape(
            scene_delta={"scene": scene_graph},
            target_delta={"current": lock_result.get("target"), "locked": [lock_result.get("target")] if lock_result.get("is_locked") else []},
            tracking_delta={
                "current": {
                    "state": status,
                    "target": target or {},
                    "follow_state": follow_score.get("reason"),
                },
                "follow_state": follow_score.get("reason"),
            },
            timestamp_ms=now_ts * 1000.0,
            freshness_ms=(target_age_s or 0.0) * 1000.0,
            diagnostics={"status": status, "outcome_status": outcome_status},
        )
        voice_context = build_vision_voice_context(
            visual_state={
                "target": target or {},
                "tracking_decision": decision,
                "status": status,
                "frame_captured_at_ts": frame_ts,
            },
            scene=scene_graph,
            events=vision_events,
            now_ts=now_ts,
        )
        memory_candidate = self._visual_memory_policy.evaluate(
            event=(vision_events[0] if vision_events else {"event_type": status}),
            target_lock=lock_result,
            follow_score=follow_score,
            context={"session_id": session_id, "actor_id": actor_id},
        )
        training_feedback = build_visual_feedback_record(
            feedback_type="target_lost" if status == "waiting_for_target" else "follow_result",
            subject=target or lock_result.get("target") or {},
            outcome={
                "success": follow_score.get("success"),
                "status": outcome_status or status,
                "reason": follow_score.get("reason"),
            },
            follow_score=follow_score,
            round_id=session_id,
            session_id=session_id,
            timestamp_ms=int(now_ts * 1000),
        )
        tuning = recommend_visual_follow_tuning(
            {
                "filtered_error": decision.get("filtered_error", current_error),
                "stable_error_count": decision.get("stable_error_count", decision.get("bias_count", 0)),
                "suppress_reason": suppressed_reason or decision.get("reason"),
                "action_interval_s": loop_elapsed_s,
                "fps": self._coerce_optional_float(eye_details.get("fps")),
                "target_freshness_s": target_age_s,
                "pan_proof_dx": decision.get("target_error_x"),
                "pan_min": self._neck_fusion.config.pan_min_angle,
                "pan_max": self._neck_fusion.config.pan_max_angle,
                "current_angle": decision.get("current_angle", self._neck_home_angle()),
            }
        ).to_dict()
        model_selection = select_vision_model_profile(
            device_capabilities={"hailo8"},
            required_capabilities={"detection", "tracking"},
            target_fps=self._coerce_float(eye_details.get("target_fps"), default=10.0),
        )
        model_profile = dict(model_selection.diagnostics)
        if model_selection.profile is not None:
            model_profile["profile_id"] = model_selection.profile.id
            model_profile["model_id"] = model_selection.profile.model_id
            model_profile["backend"] = model_selection.profile.backend
            model_profile["device"] = model_selection.profile.device
        model_profile["ok"] = model_selection.ok
        model_profile["reason"] = model_selection.reason
        soak_summary = summarize_vision_soak(
            [
                {
                    "fps": eye_details.get("fps"),
                    "target_fps": eye_details.get("target_fps"),
                    "frame_age_ms": None if target_age_s is None else target_age_s * 1000.0,
                    "loop_elapsed_ms": None if loop_elapsed_s is None else loop_elapsed_s * 1000.0,
                    "service_state": status,
                }
            ],
            min_service_ok_ratio=0.0,
        )
        if memory_candidate is not None:
            self._recent_events.append(
                {
                    "kind": "visual_memory_candidate",
                    "source": "eibrain.vision",
                    "status": "candidate",
                    "session_id": session_id,
                    "recorded_at_ts": now_ts,
                    "details": {
                        "memory_trace": memory_candidate.get("memory_trace", {}),
                        "event_type": memory_candidate.get("event_type", ""),
                        "dedupe_key": memory_candidate.get("dedupe_key", ""),
                    },
                }
            )
        return {
            "target_lock": lock_result,
            "follow_score": follow_score,
            "follow_tuning": tuning,
            "vision_events": vision_events,
            "scene_graph": scene_graph,
            "voice_context": voice_context,
            "memory_candidate": memory_candidate,
            "training_feedback": training_feedback,
            "soak_summary": soak_summary,
            "model_profile": model_profile,
        }

    def _detections_from_eye_details(self, eye_details: dict[str, object]) -> list[dict[str, object]]:
        detections = eye_details.get("detections", [])
        return [dict(item) for item in detections if isinstance(item, dict)] if isinstance(detections, list) else []

    @staticmethod
    def _decision_command_delta(decision: dict[str, object]) -> float | None:
        try:
            current = decision.get("current_angle")
            target = decision.get("target_angle")
            if current is None or target is None:
                return None
            return abs(float(target) - float(current))
        except (TypeError, ValueError):
            return None

    def _loop_elapsed_s(self) -> float | None:
        acted_at = self._last_visual_tracking_decision.get("acted_at_ts")
        if isinstance(acted_at, (int, float)):
            return max(0.0, time.time() - float(acted_at))
        return None

    @staticmethod
    def _lock_input_detections(
        detections: list[dict[str, object]],
        *,
        target: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        payload = [dict(item) for item in detections]
        if isinstance(target, dict):
            payload.append(dict(target))
        return payload

    def _prepare_tracking_target(self, target: dict[str, object]) -> dict[str, object] | None:
        score = self._coerce_float(target.get("score"), default=0.0)
        if score < 0.3:
            self._note_visual_miss(reason="low_confidence_target")
            return None
        raw_target_x = self._coerce_float(target.get("target_x"), default=0.5)
        had_tracking_target = (
            isinstance(self.interaction_state.get("tracking_target_x"), (int, float))
            and isinstance(self.interaction_state.get("tracking_raw_target_x"), (int, float))
        )
        previous_x = self._coerce_float(self.interaction_state.get("tracking_target_x"), default=raw_target_x)
        previous_raw_x = self._coerce_float(self.interaction_state.get("tracking_raw_target_x"), default=raw_target_x)
        previous_locked = bool(self.interaction_state.get("tracking_locked", False))
        stable_count = int(self.interaction_state.get("tracking_stable_count", 0))
        stable_count = stable_count + 1 if abs(raw_target_x - previous_raw_x) <= 0.08 else 1
        alpha = 0.35 if stable_count > 1 else 0.65
        if previous_locked:
            smoothed_target_x = previous_x + ((raw_target_x - previous_x) * alpha)
        else:
            smoothed_target_x = raw_target_x
        current_ts = time.time()
        smoothed_target_x = max(0.0, min(1.0, round(smoothed_target_x, 4)))
        last_neck_action_at_ts = self.interaction_state.get("last_neck_action_at_ts")
        since_last_action_s = (
            current_ts - float(last_neck_action_at_ts)
            if isinstance(last_neck_action_at_ts, (int, float))
            else None
        )
        command_delta = abs(smoothed_target_x - previous_x)
        if (
            had_tracking_target
            and previous_locked
            and since_last_action_s is not None
            and since_last_action_s < 0.45
            and command_delta < 0.04
        ):
            self.interaction_state.update(
                {
                    "tracking_locked": True,
                    "tracking_target_label": str(target.get("label", "target")),
                    "tracking_target_score": score,
                    "tracking_target_x": round(smoothed_target_x, 4),
                    "tracking_raw_target_x": round(raw_target_x, 4),
                    "tracking_stable_count": stable_count,
                    "tracking_miss_count": 0,
                    "updated_at_ts": current_ts,
                }
            )
            self._refresh_interaction_mode(force_mode="attention", force_reason="tracking_hold")
            return None
        if (
            had_tracking_target
            and previous_locked
            and command_delta < 0.02
        ):
            self.interaction_state.update(
                {
                    "tracking_locked": True,
                    "tracking_target_label": str(target.get("label", "target")),
                    "tracking_target_score": score,
                    "tracking_target_x": round(smoothed_target_x, 4),
                    "tracking_raw_target_x": round(raw_target_x, 4),
                    "tracking_stable_count": stable_count,
                    "tracking_miss_count": 0,
                    "updated_at_ts": current_ts,
                }
            )
            self._refresh_interaction_mode(force_mode="attention", force_reason="tracking_hold")
            return None
        return {
            **target,
            "label": str(target.get("label", "target")),
            "score": score,
            "target_x": smoothed_target_x,
            "raw_target_x": round(raw_target_x, 4),
            "tracking_target_center_x": smoothed_target_x,
            "tracking_target_error_x": self._target_error_x(smoothed_target_x),
            "tracking_stable_count": stable_count,
        }

    @staticmethod
    def _target_error_x(value: object) -> float | None:
        try:
            return round(float(value) - 0.5, 4)
        except (TypeError, ValueError):
            return None

    def _note_voice_activity(self) -> None:
        now_ts = time.time()
        self.interaction_state["last_voice_activity_at_ts"] = now_ts
        self.interaction_state["updated_at_ts"] = now_ts

    def _note_visual_target_locked(self, target: dict[str, object], *, neck_action: bool = True) -> None:
        now_ts = time.time()
        state_update: dict[str, object] = {
            "tracking_locked": True,
            "tracking_target_label": str(target.get("label", "target")),
            "tracking_target_score": self._coerce_float(target.get("score"), default=0.0),
            "tracking_target_x": self._coerce_float(target.get("target_x"), default=0.5),
            "tracking_raw_target_x": self._coerce_float(target.get("raw_target_x"), default=0.5),
            "tracking_stable_count": int(target.get("tracking_stable_count", 1)),
            "tracking_miss_count": 0,
            "last_attention_at_ts": now_ts,
            "updated_at_ts": now_ts,
        }
        if neck_action:
            state_update["last_neck_action_at_ts"] = now_ts
        self.interaction_state.update(state_update)
        self._refresh_interaction_mode(force_mode="attention", force_reason="visual_target_locked")

    def _note_visual_miss(self, *, reason: str = "visual_target_missing") -> None:
        miss_count = int(self.interaction_state.get("tracking_miss_count", 0)) + 1
        stable_count = int(self.interaction_state.get("tracking_stable_count", 0))
        self.interaction_state.update(
            {
                "tracking_miss_count": miss_count,
                "tracking_stable_count": max(0, stable_count - 1),
                "updated_at_ts": time.time(),
            }
        )
        if miss_count >= 3:
            self.interaction_state["tracking_locked"] = False
        self._refresh_interaction_mode(force_reason=reason)

    def _voice_organ_payload(self, organ_name: str, *, recent_events: list[dict[str, object]]) -> dict[str, object]:
        organ = next((item for item in self.organs if getattr(item, "name", None) == organ_name), None)
        latest_event = self._latest_voice_event(organ_name, recent_events)
        if organ is None:
            return {
                "status": "degraded" if latest_event else "not_wired",
                "state": "degraded" if latest_event else "not_wired",
                "not_wired": latest_event is None,
                "readiness_message": f"{organ_name} organ missing",
                "latest_event": latest_event,
            }
        try:
            if self.voice_dialogue_state.get("running") and hasattr(organ, "passive_heartbeat"):
                heartbeat = organ.passive_heartbeat()
            else:
                heartbeat = organ.heartbeat()
            raw = self._json_ready(heartbeat)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return {
                "status": "degraded",
                "state": "degraded",
                "health": "degraded",
                "readiness_message": f"{organ_name} heartbeat failed: {exc}",
                "latest_event": latest_event,
            }
        if not isinstance(raw, dict):
            raw = {"health": "unknown", "raw": raw}
        return self._voice_component_from_heartbeat(organ_name, raw, latest_event=latest_event)

    def _voice_component_from_heartbeat(
        self,
        organ_name: str,
        heartbeat: dict[str, object],
        *,
        latest_event: dict[str, object] | None,
    ) -> dict[str, object]:
        subfunctions = heartbeat.get("subfunctions")
        subfunctions = subfunctions if isinstance(subfunctions, dict) else {}
        if organ_name == "ear":
            capture = self._subfunction_payload(subfunctions, "capture")
            asr = self._subfunction_payload(subfunctions, "asr")
            provider = self._first_text(
                self._details_value(asr, "provider"),
                self._details_value(asr, "backend"),
                self._details_value(capture, "provider"),
            )
            status, state, readiness = self._voice_component_status(
                organ_name,
                heartbeat=heartbeat,
                subfunctions=[capture, asr],
                latest_event=latest_event,
                wired=bool(provider),
            )
            payload = {
                "status": status,
                "state": state,
                "health": self._text_or_none(heartbeat.get("health")),
                "provider": provider,
                "readiness_message": readiness,
                "latest_event": latest_event,
                "subfunctions": subfunctions,
                "capture": capture,
                "asr": asr,
            }
            if state == "not_wired":
                payload["not_wired"] = True
            return payload
        playback = self._subfunction_payload(subfunctions, "tts_playback")
        plan = self._subfunction_payload(subfunctions, "tts_plan")
        backend = self._first_text(
            self._details_value(playback, "backend"),
            self._details_value(playback, "provider"),
            self._details_value(plan, "backend"),
            self._details_value(plan, "provider"),
        )
        status, state, readiness = self._voice_component_status(
            organ_name,
            heartbeat=heartbeat,
            subfunctions=[playback, plan],
            latest_event=latest_event,
            wired=bool(backend and backend != "noop"),
        )
        payload = {
            "status": status,
            "state": state,
            "health": self._text_or_none(heartbeat.get("health")),
            "backend": backend,
            "model": self._first_text(self._details_value(playback, "model"), self._details_value(plan, "model")),
            "voice_id": self._first_text(
                self._details_value(playback, "voice_id"),
                self._details_value(plan, "voice_id"),
            ),
            "text_preview": self._first_text(self._details_value(playback, "text_preview")),
            "readiness_message": readiness,
            "latest_event": latest_event,
            "subfunctions": subfunctions,
            "tts_playback": playback,
            "tts_plan": plan,
        }
        if state == "not_wired":
            payload["not_wired"] = True
        return payload

    def _voice_component_status(
        self,
        organ_name: str,
        *,
        heartbeat: dict[str, object],
        subfunctions: list[dict[str, object] | None],
        latest_event: dict[str, object] | None,
        wired: bool,
    ) -> tuple[str, str, str]:
        health = self._normalized_text(heartbeat.get("health"))
        driver_names = {
            self._normalized_text(self._details_value(subfunction, "driver"))
            for subfunction in subfunctions
            if subfunction is not None
        }
        sub_statuses = {
            self._normalized_text(subfunction.get("health"))
            for subfunction in subfunctions
            if subfunction is not None
        }
        if driver_names and driver_names == {"noop"}:
            return "not_wired", "not_wired", f"{organ_name} uses noop drivers"
        if health in {"unavailable", "disabled", "missing", "not_wired"}:
            return "not_wired", "not_wired", f"{organ_name} heartbeat unavailable"
        if "degraded" in sub_statuses or health in {"degraded", "error", "failed", "unhealthy"}:
            return "degraded", "degraded", f"{organ_name} heartbeat degraded"
        if wired:
            return "ok", "wired", f"{organ_name} wired"
        if latest_event is not None:
            return "degraded", "degraded", f"{organ_name} has recent events but heartbeat is incomplete"
        return "unknown", "unknown", f"{organ_name} heartbeat present but no realtime data"

    def _voice_dialogue_payload(self) -> dict[str, object]:
        dialogue = dict(self.voice_dialogue_state)
        has_turn = bool(dialogue.get("last_completed_turn") or dialogue.get("last_transcript") or dialogue.get("last_reply"))
        has_latency = bool(dialogue.get("last_stage_latency_ms") or dialogue.get("last_latency_s"))
        running = bool(dialogue.get("running"))
        enabled = bool(dialogue.get("enabled"))
        if running:
            status = self._first_text(dialogue.get("phase"), "running")
            state = "wired"
            readiness = "voice dialogue running"
        elif self._normalized_text(dialogue.get("last_status")) in {"stopped", "disabled", "offline"}:
            status = self._first_text(dialogue.get("last_status"), "stopped")
            state = "not_wired"
            readiness = "voice dialogue stopped; historical turn data is retained"
        elif has_turn or has_latency:
            status = self._first_text(dialogue.get("last_status"), "completed")
            state = "degraded" if not enabled else "wired"
            readiness = "voice dialogue has recent turn data" if enabled else "voice dialogue has history but is disabled"
        elif enabled:
            status = "waiting_for_voice"
            state = "degraded"
            readiness = "voice dialogue enabled but waiting for data"
        else:
            status = "unknown"
            state = "unknown"
            readiness = "voice dialogue has no running data"
        dialogue.update(
            {
                "status": status,
                "state": state,
                "readiness_message": readiness,
            }
        )
        return dialogue

    def _voice_chain_readiness_payload(self, dialogue: dict[str, object]) -> dict[str, object]:
        explicit = self._first_mapping_from(dialogue, keys=("voice_chain_readiness", "voiceChainReadiness"))
        benchmark = self._first_mapping_from(dialogue, keys=("voice_chain_benchmark", "voiceChainBenchmark"))
        scenario_targets = run_voice_chain_scenarios()
        return build_voice_chain_readiness(
            explicit=explicit,
            benchmark=benchmark,
            scenario_targets=scenario_targets,
        )

    def _voice_latency_payload(self, dialogue: dict[str, object]) -> dict[str, object]:
        stage_latency_ms = self._float_mapping(dialogue.get("last_stage_latency_ms"))
        stage_latency_s = self._float_mapping(dialogue.get("last_latency_s"))
        for name, seconds in stage_latency_s.items():
            stage_latency_ms.setdefault(name, round(seconds * 1000.0, 3))
        total_ms = None
        explicit_total = stage_latency_ms.get("total")
        if explicit_total is not None:
            total_ms = explicit_total
        if stage_latency_ms:
            total_ms = round(
                sum(
                    value
                    for key, value in stage_latency_ms.items()
                    if key not in {"total", "overhead"}
                ),
                3,
            ) if total_ms is None else total_ms
        return {
            "total_ms": total_ms,
            "stage_latency_ms": stage_latency_ms,
            "stage_latency_s": stage_latency_s,
        }

    def _merge_voice_realtime_latency(
        self,
        latency: dict[str, object],
        realtime_session: dict[str, object] | None,
    ) -> None:
        if not isinstance(realtime_session, dict):
            return
        session_latency = realtime_session.get("latency_ms")
        if not isinstance(session_latency, dict):
            return
        stage_latency_ms = latency.setdefault("stage_latency_ms", {})
        if not isinstance(stage_latency_ms, dict):
            return
        for key, value in session_latency.items():
            number = self._coerce_optional_float(value)
            if number is not None:
                stage_latency_ms.setdefault(str(key), number)
        first_speech_ms = self._coerce_optional_float(stage_latency_ms.get("first_speech"))
        if first_speech_ms is None:
            return
        latency["first_speech_ms"] = first_speech_ms
        latency["first_speech_within_2s"] = first_speech_ms <= 2000.0

    def _voice_bottleneck_payload(
        self,
        dialogue: dict[str, object],
        *,
        latency: dict[str, object],
    ) -> dict[str, object] | None:
        stage = self._first_text(dialogue.get("last_bottleneck_stage"))
        latency_ms = self._coerce_optional_float(dialogue.get("last_bottleneck_ms"))
        stage_latency = latency.get("stage_latency_ms")
        if (not stage or latency_ms is None) and isinstance(stage_latency, dict) and stage_latency:
            candidates = {
                key: value
                for key, value in stage_latency.items()
                if key not in {"total", "overhead"}
            }
            computed_stage, computed_ms = max((candidates or stage_latency).items(), key=lambda item: float(item[1]))
            stage = stage or str(computed_stage)
            latency_ms = latency_ms if latency_ms is not None else float(computed_ms)
        if not stage and latency_ms is None:
            return None
        return {
            "stage": stage or None,
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _voice_last_turn(dialogue: dict[str, object]) -> dict[str, object] | None:
        completed = dialogue.get("last_completed_turn")
        if isinstance(completed, dict) and completed:
            return dict(completed)
        transcript = str(dialogue.get("last_transcript", "") or "").strip()
        reply = str(dialogue.get("last_reply", "") or "").strip()
        if transcript or reply:
            return {"transcript": transcript, "reply": reply}
        return None

    def _voice_realtime_session_payload(self, dialogue: dict[str, object]) -> dict[str, object] | None:
        raw = self._first_present(dialogue, "realtime_session", "latest_realtime_session")
        if raw is None:
            return None
        if hasattr(raw, "snapshot") and callable(raw.snapshot):
            raw = raw.snapshot()
        ready = self._json_ready(raw)
        return dict(ready) if isinstance(ready, dict) and ready else None

    def _voice_round_payload(
        self,
        dialogue: dict[str, object],
        realtime_session: dict[str, object] | None,
    ) -> dict[str, object]:
        round_id = self._first_present(dialogue, "current_round_id", "round_id")
        if round_id in (None, ""):
            round_id = self._first_present(realtime_session, "current_round_id", "round_id", "roundId")
        cancellation_token = self._first_present(dialogue, "current_cancellation_token", "cancellation_token")
        if cancellation_token in (None, ""):
            cancellation_token = self._first_present(
                realtime_session,
                "current_cancellation_token",
                "cancellation_token",
                "cancellationToken",
            )
        phase = self._first_text(
            dialogue.get("phase"),
            self._first_present(realtime_session, "phase"),
        )
        last_status = self._first_text(
            dialogue.get("last_status"),
            dialogue.get("status"),
            self._first_present(realtime_session, "last_status", "status"),
        )
        normalized_status = self._normalized_text(last_status)
        interrupted = (
            normalized_status in {"interrupted", "interrupt", "cancelled", "canceled"}
            or self._truthy(dialogue.get("interrupted"))
            or self._truthy(self._first_present(realtime_session, "interrupted"))
        )
        complete = self._truthy(self._first_present(realtime_session, "complete")) or normalized_status in {
            "completed",
            "complete",
            "done",
            "finished",
        }
        has_round = round_id not in (None, "")
        lifecycle = (
            "interrupted"
            if interrupted
            else "completed"
            if complete
            else "active"
            if has_round
            else "unknown"
        )
        return {
            "current_round_id": self._json_ready(round_id) if has_round else None,
            "current_cancellation_token": (
                self._json_ready(cancellation_token) if cancellation_token not in (None, "") else None
            ),
            "has_cancellation_token": cancellation_token not in (None, ""),
            "phase": phase,
            "last_status": last_status,
            "active": lifecycle == "active",
            "complete": bool(complete),
            "interrupted": bool(interrupted),
            "lifecycle": lifecycle,
            "state": lifecycle if has_round else "unknown",
        }

    def _voice_scheduler_state_payload(self, dialogue: dict[str, object]) -> dict[str, object]:
        raw = self._first_present(dialogue, "scheduler_state", "scheduler")
        details = self._json_ready(raw) if raw is not None else None
        payload: dict[str, object] = {}
        if isinstance(details, dict):
            payload.update(details)
            state = self._first_text(payload.get("state"), payload.get("status"), payload.get("phase"))
        else:
            state = self._first_text(details)
            if details is not None:
                payload["value"] = details
        normalized = self._normalized_text(state)
        stale = self._truthy(payload.get("stale")) or normalized == "stale"
        not_wired = self._truthy(payload.get("not_wired")) or normalized in {
            "not_wired",
            "missing",
            "unavailable",
            "disabled",
            "offline",
        }
        component_state = self._voice_scheduler_component_state(
            state=state,
            stale=stale,
            not_wired=not_wired,
        )
        payload["state"] = state or ("not_wired" if not_wired else "unknown")
        payload["component_state"] = component_state
        payload["wired"] = component_state == "wired"
        payload["not_wired"] = component_state == "not_wired"
        payload["stale"] = bool(stale)
        return payload

    @staticmethod
    def _voice_scheduler_component_state(*, state: str, stale: bool, not_wired: bool) -> str:
        normalized = BodyRuntimeApp._normalized_text(state)
        if not_wired:
            return "not_wired"
        if stale or normalized in {"stale", "blocked", "error", "failed", "unhealthy"}:
            return "degraded"
        if normalized in {"ok", "healthy", "ready", "running", "active", "scheduled", "idle"}:
            return "wired"
        return "unknown"

    def _voice_realtime_cognition_payloads(
        self,
        dialogue: dict[str, object],
        realtime_session: dict[str, object] | None,
        scheduler_state: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        lanes_source = self._first_mapping_from(
            scheduler_state,
            dialogue,
            realtime_session,
            keys=("lanes", "lane_states", "scheduler_lanes"),
        )
        fast_think = self._voice_lane_payload(
            self._first_mapping_from(
                lanes_source,
                scheduler_state,
                dialogue,
                realtime_session,
                keys=("fast_think", "fastThink", "fast_lane", "fast"),
            ),
            default_state="unknown",
        )
        slow_reasoner = self._voice_lane_payload(
            self._first_mapping_from(
                lanes_source,
                scheduler_state,
                dialogue,
                realtime_session,
                keys=("slow_reasoner", "slowReasoner", "slow_reasoning", "slowReasoning", "slow_lane", "slow", "slow_thinking"),
            ),
            default_state="unknown",
        )
        arbiter = self._voice_lane_payload(
            self._first_mapping_from(
                lanes_source,
                scheduler_state,
                dialogue,
                realtime_session,
                keys=("arbiter", "response_arbiter"),
            ),
            default_state="unknown",
        )
        lanes = self._voice_lanes_payload(fast_think=fast_think, slow_reasoner=slow_reasoner, arbiter=arbiter)
        speech_action_plan = self._voice_speech_action_plan_payload(
            self._first_mapping_from(
                scheduler_state,
                dialogue,
                realtime_session,
                keys=("speech_action_plan", "speechActionPlan", "speech_plan"),
            )
        )
        proactive_activity = self._voice_proactive_activity_payload(
            self._first_mapping_from(
                scheduler_state,
                dialogue,
                realtime_session,
                keys=("proactive_activity", "proactiveActivity", "activity_proposal", "activity"),
            )
        )
        return {
            "lanes": lanes,
            "fast_think": fast_think,
            "slow_reasoner": slow_reasoner,
            "arbiter": arbiter,
            "speech_action_plan": speech_action_plan,
            "proactive_activity": proactive_activity,
        }

    def _voice_lane_payload(self, raw: dict[str, object] | None, *, default_state: str) -> dict[str, object]:
        if raw is None:
            return {
                "state": default_state,
                "status": default_state,
                "component_state": default_state,
                "wired": False,
                "not_wired": default_state == "not_wired",
                "summary": default_state,
            }
        payload = dict(raw)
        state = self._first_text(payload.get("state"), payload.get("status"), payload.get("phase"), default_state)
        normalized = self._normalized_text(state)
        not_wired = normalized in {"not_wired", "missing", "unavailable", "disabled", "offline"}
        stale = normalized in {"stale", "blocked", "error", "failed", "unhealthy"}
        if not_wired:
            component_state = "not_wired"
        elif stale:
            component_state = "degraded"
        elif normalized in {"unknown", ""}:
            component_state = "unknown"
        else:
            component_state = "wired"
        latency_ms = self._coerce_optional_float(
            self._first_present(payload, "latency_ms", "latencyMs", "elapsed_ms", "elapsedMs")
        )
        summary = state
        if latency_ms is not None:
            summary = f"{state} ({latency_ms}ms)"
        payload.update(
            {
                "state": state,
                "status": state,
                "component_state": component_state,
                "wired": component_state == "wired",
                "not_wired": component_state == "not_wired",
                "summary": summary,
            }
        )
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        return payload

    @staticmethod
    def _voice_lanes_payload(
        *,
        fast_think: dict[str, object],
        slow_reasoner: dict[str, object],
        arbiter: dict[str, object],
    ) -> dict[str, object]:
        states = [
            str(component.get("component_state") or "unknown")
            for component in (fast_think, slow_reasoner, arbiter)
        ]
        if any(state == "wired" for state in states):
            component_state = "wired"
        elif any(state == "degraded" for state in states):
            component_state = "degraded"
        elif any(state == "not_wired" for state in states):
            component_state = "not_wired"
        else:
            component_state = "unknown"
        return {
            "fast_think": fast_think,
            "slow_reasoner": slow_reasoner,
            "arbiter": arbiter,
            "component_state": component_state,
            "wired": component_state == "wired",
            "not_wired": component_state == "not_wired",
            "summary": " / ".join(
                f"{name}={component.get('state', 'unknown')}"
                for name, component in (
                    ("fast_think", fast_think),
                    ("slow_reasoner", slow_reasoner),
                    ("arbiter", arbiter),
                )
            ),
        }

    def _voice_speech_action_plan_payload(self, raw: dict[str, object] | None) -> dict[str, object]:
        if raw is None:
            return self._voice_missing_realtime_component("not_wired")
        payload = dict(raw)
        plan_id = self._first_text(payload.get("planId"), payload.get("plan_id"), payload.get("id"), "unknown")
        speech_segments = self._first_list(payload, "speechSegments", "speech_segments", "speech")
        action_segments = self._first_list(payload, "actionSegments", "action_segments", "actions", "action_plan")
        state = self._first_text(payload.get("state"), payload.get("status"), "ready")
        payload.update(
            {
                "plan_id": plan_id,
                "state": state,
                "status": state,
                "component_state": "wired",
                "wired": True,
                "not_wired": False,
                "speech_count": len(speech_segments),
                "action_count": len(action_segments),
                "summary": (
                    f"{plan_id}: {len(speech_segments)} speech, "
                    f"{len(action_segments)} {self._plural('action', len(action_segments))}"
                ),
            }
        )
        return payload

    def _voice_proactive_activity_payload(self, raw: dict[str, object] | None) -> dict[str, object]:
        if raw is None:
            return self._voice_missing_realtime_component("not_wired")
        payload = dict(raw)
        proposal_id = self._first_text(payload.get("proposalId"), payload.get("proposal_id"), payload.get("id"), "unknown")
        channel = self._first_text(payload.get("channel"), "unknown")
        should_emit = self._truthy(self._first_present(payload, "shouldEmit", "should_emit"))
        state = self._first_text(payload.get("state"), payload.get("status"), "proposed")
        payload.update(
            {
                "proposal_id": proposal_id,
                "channel": channel,
                "should_emit": should_emit,
                "state": state,
                "status": state,
                "component_state": "wired",
                "wired": True,
                "not_wired": False,
                "summary": f"{proposal_id}: {channel} / {'emit' if should_emit else 'hold'}",
            }
        )
        return payload

    @staticmethod
    def _voice_missing_realtime_component(state: str) -> dict[str, object]:
        return {
            "state": state,
            "status": state,
            "component_state": state,
            "wired": False,
            "not_wired": state == "not_wired",
            "summary": state,
        }

    @staticmethod
    def _plural(word: str, count: int) -> str:
        return word if count == 1 else f"{word}s"

    @staticmethod
    def _first_mapping_from(*sources: dict[str, object] | None, keys: tuple[str, ...]) -> dict[str, object] | None:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if isinstance(value, dict):
                    return dict(value)
        return None

    @staticmethod
    def _first_list(mapping: dict[str, object], *keys: str) -> list[object]:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, list):
                return list(value)
        return []

    def _voice_interruption_payload(
        self,
        dialogue: dict[str, object],
        realtime_session: dict[str, object] | None,
    ) -> dict[str, object]:
        interrupt_count = self._int_or_none(self._first_present(dialogue, "interrupt_count", "interruption_count", "interrupts"))
        interrupted_round_count = self._int_or_none(dialogue.get("interrupted_round_count"))
        last_interrupt_raw = self._first_present(dialogue, "last_interrupt", "interruption", "interrupt")
        last_interrupt = self._json_ready(last_interrupt_raw) if last_interrupt_raw is not None else None
        last_status = self._normalized_text(self._first_text(dialogue.get("last_status"), dialogue.get("status")))
        interrupt_active_raw = dialogue.get("interrupt_active")
        interrupt_active = self._truthy(interrupt_active_raw)
        stale = self._truthy(self._mapping_value(last_interrupt, "stale")) or self._truthy(dialogue.get("interrupt_stale"))
        session_interrupted = self._truthy(self._first_present(realtime_session, "interrupted"))
        interrupted = (
            interrupt_active
            or self._truthy(dialogue.get("interrupted"))
            or session_interrupted
            or last_status in {"interrupted", "interrupt", "cancelled", "canceled"}
        )
        if last_interrupt is None and session_interrupted:
            last_interrupt = {
                "round_id": self._first_present(realtime_session, "round_id", "roundId"),
                "cancellation_token": self._first_present(
                    realtime_session,
                    "cancellation_token",
                    "cancellationToken",
                ),
                "reason": self._first_present(realtime_session, "interrupt_reason"),
            }
        has_history = bool(
            last_interrupt is not None
            or (interrupt_count is not None and interrupt_count > 0)
            or (interrupted_round_count is not None and interrupted_round_count > 0)
        )
        has_interrupt_signal = bool(
            interrupt_active_raw is not None
            or interrupt_count is not None
            or interrupted_round_count is not None
            or last_interrupt is not None
            or dialogue.get("interrupted") is not None
            or self._first_present(realtime_session, "interrupted") is not None
        )
        no_interrupts_seen = (
            has_interrupt_signal
            and interrupt_active is False
            and not interrupted
            and last_interrupt is None
            and (interrupt_count == 0 or interrupt_count is None)
            and (interrupted_round_count == 0 or interrupted_round_count is None)
        )
        state = (
            "stale"
            if stale
            else "interrupted"
            if interrupted
            else "history"
            if has_history
            else "clear"
            if no_interrupts_seen
            else "unknown"
        )
        component_state = "degraded" if stale or interrupted else "wired" if state == "clear" else "unknown"
        return {
            "state": state,
            "status": state,
            "component_state": component_state,
            "active": bool(interrupt_active),
            "interrupted": bool(interrupted),
            "stale": bool(stale),
            "clear": state == "clear",
            "has_history": bool(has_history),
            "interrupt_count": interrupt_count,
            "interrupted_round_count": interrupted_round_count,
            "last_interrupt": last_interrupt,
        }

    def _voice_cancellation_chain_payload(
        self,
        dialogue: dict[str, object],
        realtime_session: dict[str, object] | None,
    ) -> list[object]:
        raw = self._first_present(dialogue, "cancellation_chain")
        if raw is None:
            raw = self._first_present(realtime_session, "cancellation_chain")
        ready = self._json_ready(raw)
        return list(ready) if isinstance(ready, list) else []

    def _voice_overall_status(
        self,
        *,
        ear: dict[str, object],
        mouth: dict[str, object],
        dialogue: dict[str, object],
        latency: dict[str, object],
        last_turn: dict[str, object] | None,
        scheduler: dict[str, object] | None = None,
        interruption: dict[str, object] | None = None,
    ) -> tuple[str, bool, bool]:
        states = [
            self._normalized_text(component.get("component_state") or component.get("state"))
            for component in (ear, mouth, dialogue, scheduler, interruption)
            if isinstance(component, dict)
        ]
        has_signal = bool(any(state and state != "unknown" for state in states) or last_turn or latency.get("stage_latency_ms"))
        if not has_signal:
            return "not_wired", False, True
        if any(state == "wired" for state in states):
            if any(state in {"degraded", "not_wired"} for state in states):
                return "degraded", False, False
            return "wired", True, False
        if any(state == "degraded" for state in states):
            return "degraded", False, False
        if any(state == "not_wired" for state in states):
            return "not_wired", False, True
        return "unknown", False, False

    def _voice_readiness_message(
        self,
        *,
        status: str,
        ear: dict[str, object],
        mouth: dict[str, object],
        dialogue: dict[str, object],
        scheduler: dict[str, object] | None = None,
        interruption: dict[str, object] | None = None,
    ) -> str:
        if status == "wired":
            return "voice loop wired"
        messages = []
        for name, component in (
            ("ear", ear),
            ("mouth", mouth),
            ("dialogue", dialogue),
            ("scheduler", scheduler),
            ("interruption", interruption),
        ):
            if not isinstance(component, dict):
                continue
            message = self._first_text(component.get("readiness_message"), component.get("status"))
            if message and message not in messages:
                messages.append(f"{name}: {message}")
        joined = "; ".join(messages)
        if status == "not_wired":
            return f"voice realtime not wired: {joined}" if joined else "voice realtime not wired"
        if status == "degraded":
            return f"voice realtime degraded: {joined}" if joined else "voice realtime degraded"
        return f"voice realtime state unknown: {joined}" if joined else "voice realtime state unknown"

    def _recent_voice_events(self) -> list[dict[str, object]]:
        events = []
        for event in self._recent_events:
            source = str(event.get("source", "") or "")
            kind = str(event.get("kind", "") or "")
            if source.startswith(("ear", "mouth", "body_runtime.voice")) or "speech" in kind or "transcript" in kind:
                events.append(self._json_ready(event))
        return events[-10:]

    @staticmethod
    def _latest_voice_event(organ_name: str, events: list[dict[str, object]]) -> dict[str, object] | None:
        prefix = "ear" if organ_name == "ear" else "mouth"
        for event in reversed(events):
            source = str(event.get("source", "") or "")
            kind = str(event.get("kind", "") or "")
            if source.startswith(prefix) or (organ_name == "mouth" and "speech" in kind):
                return dict(event)
        return None

    @staticmethod
    def _subfunction_payload(subfunctions: dict[str, object], name: str) -> dict[str, object] | None:
        payload = subfunctions.get(name)
        return dict(payload) if isinstance(payload, dict) else None

    @staticmethod
    def _details_value(subfunction: dict[str, object] | None, key: str) -> object:
        if not isinstance(subfunction, dict):
            return None
        details = subfunction.get("details")
        if isinstance(details, dict):
            return details.get(key)
        return None

    @staticmethod
    def _float_mapping(value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, float] = {}
        for key, item in value.items():
            number = BodyRuntimeApp._coerce_optional_float(item)
            if number is not None:
                result[str(key)] = number
        return result

    @staticmethod
    def _json_ready(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): BodyRuntimeApp._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [BodyRuntimeApp._json_ready(item) for item in value]
        if is_dataclass(value):
            return BodyRuntimeApp._json_ready(asdict(value))
        if hasattr(value, "to_dict") and callable(value.to_dict):
            payload = value.to_dict()
            if isinstance(payload, dict):
                return BodyRuntimeApp._json_ready(payload)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _first_present(mapping: dict[str, object] | None, *keys: str) -> object:
        if not isinstance(mapping, dict):
            return None
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _mapping_value(mapping: object, key: str) -> object:
        if isinstance(mapping, dict):
            return mapping.get(key)
        return None

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _truthy(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        text = BodyRuntimeApp._normalized_text(value)
        return text in {"1", "true", "yes", "y", "on", "active", "interrupted", "cancelled", "canceled"}

    def _refresh_interaction_mode(
        self,
        *,
        force_mode: str | None = None,
        force_reason: str | None = None,
    ) -> None:
        now_ts = time.time()
        if force_mode is not None:
            mode = force_mode
        elif self.is_speaking():
            mode = "responding"
        else:
            last_attention_at_ts = self.interaction_state.get("last_attention_at_ts")
            attention_recent = isinstance(last_attention_at_ts, (int, float)) and now_ts - float(last_attention_at_ts) < 5.0
            if self.interaction_state.get("tracking_locked") and attention_recent:
                mode = "attention"
            elif self.voice_dialogue_state.get("running"):
                mode = "listening"
            else:
                last_voice_at_ts = self.interaction_state.get("last_voice_activity_at_ts")
                heard_recently = isinstance(last_voice_at_ts, (int, float)) and now_ts - float(last_voice_at_ts) < 8.0
                mode = "listening" if heard_recently else "sleeping"
        self.interaction_state["current_mode"] = mode
        self.interaction_state["reason"] = force_reason or self._interaction_reason(mode)
        self.interaction_state["updated_at_ts"] = now_ts

    def _interaction_reason(self, mode: str) -> str:
        if mode == "responding":
            return "speaking"
        if mode == "attention":
            return "visual_target_locked"
        if mode == "listening":
            return "voice_runtime_active"
        return "idle"

    @staticmethod
    def _coerce_float(value: object, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _text_or_none(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _first_text(*values: object) -> str:
        for value in values:
            text = BodyRuntimeApp._text_or_none(value)
            if text:
                return text
        return ""

    @staticmethod
    def _normalized_text(value: object) -> str:
        text = BodyRuntimeApp._text_or_none(value)
        return text.lower() if text is not None else ""
