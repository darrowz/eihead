"""Streaming ear capture helpers."""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass, field
import math
import subprocess
import time

from eibrain.protocol.observations import AudioTranscriptFinal
from eibrain.body.vad_policy import VadEndpointPolicy, VadFrame


@dataclass(slots=True)
class ArecordStreamCapture:
    device: str
    sample_rate: int
    channels: int
    retry_count: int = 2
    retry_delay_s: float = 1.0
    lock_path: str = "/tmp/eibrain-arecord.lock"
    lock_timeout_s: float = 8.0
    streaming_vad: bool = False
    vad_frame_ms: int = 80
    vad_rms_threshold: float = 0.028
    vad_min_voice_ms: int = 160
    vad_end_silence_ms: int = 360
    vad_pre_roll_ms: int = 240
    vad_min_capture_ms: int = 0
    transcribe_vad_miss: bool = False
    vad_miss_rms_threshold: float = 0.0
    vad_endpoint_policy: bool = False
    vad_backend: str = "rms"
    vad_noise_ratio: float = 1.18
    vad_silero_threshold: float = 0.5
    last_returncode: int | None = None
    last_stderr: str = ""
    last_stdout_bytes: int = 0
    last_vad_backend: str = ""
    last_command: list[str] = field(default_factory=list)
    last_vad_triggered: bool = False
    last_vad_frame_count: int = 0
    last_vad_voice_frame_count: int = 0
    last_vad_elapsed_ms: float = 0.0
    last_vad_reason: str = ""
    last_chunks: list[bytes] = field(default_factory=list)

    def build_command(self) -> list[str]:
        return [
            "arecord",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-t",
            "raw",
        ]

    def read_chunks(self, chunk_count: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        command = self.build_command() + ["-d", str(max(1, chunk_count))]
        payload = self._run_arecord(command)
        if chunk_count <= 1:
            return [payload]
        return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)][:chunk_count]

    def read_window(self, duration_s: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        if self.streaming_vad:
            return self.read_voice_window(duration_s, chunk_bytes=chunk_bytes)
        command = self.build_command() + ["-d", str(max(1, duration_s))]
        payload = self._run_arecord(command)
        if not payload:
            self.last_chunks = []
            return []
        self.last_chunks = [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)]
        return list(self.last_chunks)

    def read_voice_window(self, max_duration_s: int, *, chunk_bytes: int = 4096) -> list[bytes]:
        command = self.build_command()
        frame_bytes = self._frame_bytes()
        max_frames = max(1, math.ceil(max_duration_s * 1000 / max(1, self.vad_frame_ms)))
        pre_roll_frames = max(1, math.ceil(self.vad_pre_roll_ms / max(1, self.vad_frame_ms)))
        min_voice_frames = max(1, math.ceil(self.vad_min_voice_ms / max(1, self.vad_frame_ms)))
        end_silence_frames = max(1, math.ceil(self.vad_end_silence_ms / max(1, self.vad_frame_ms)))
        min_capture_frames = max(0, math.ceil(self.vad_min_capture_ms / max(1, self.vad_frame_ms)))
        endpoint_policy = self._endpoint_policy(max_duration_s=max_duration_s) if self.vad_endpoint_policy else None
        all_frames: list[bytes] = []
        captured_frames: list[bytes] = []
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        pending_voice: list[bytes] = []
        triggered = False
        voice_frames = 0
        consecutive_voice_frames = 0
        silence_after_voice = 0
        started = time.perf_counter()
        self.last_command = list(command)
        self.last_returncode = None
        self.last_stderr = ""
        self.last_stdout_bytes = 0
        self.last_vad_backend = self.vad_backend
        try:
            with self._capture_lock():
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    if process.stdout is None:
                        return []
                    for _ in range(max_frames):
                        frame = process.stdout.read(frame_bytes)
                        if not frame:
                            break
                        all_frames.append(frame)
                        stats = pcm_signal_stats([frame], channels=self.channels)
                        decision = (
                            endpoint_policy.observe(VadFrame(rms_level=float(stats["rms_level"])))
                            if endpoint_policy is not None
                            else None
                        )
                        is_voice = bool(decision.is_voice) if decision is not None else bool(stats["rms_level"] >= self.vad_rms_threshold)
                        if decision is not None:
                            self.last_vad_reason = decision.reason
                            if not triggered and decision.should_force_decode:
                                break
                        if not triggered:
                            if not is_voice:
                                pre_roll.append(frame)
                                pending_voice.clear()
                                consecutive_voice_frames = 0
                                continue
                            pending_voice.append(frame)
                            consecutive_voice_frames += 1
                            if decision is not None:
                                should_start = decision.should_start
                            else:
                                should_start = consecutive_voice_frames >= min_voice_frames
                            if not should_start:
                                continue
                            triggered = True
                            captured_frames.extend(pre_roll)
                            captured_frames.extend(pending_voice)
                            voice_frames = consecutive_voice_frames
                            pending_voice.clear()
                            silence_after_voice = 0
                            continue
                        captured_frames.append(frame)
                        if is_voice:
                            voice_frames += 1
                            silence_after_voice = 0
                        else:
                            silence_after_voice += 1
                        if (
                            decision.should_stop or decision.should_force_decode
                            if decision is not None
                            else (
                                len(captured_frames) >= min_capture_frames
                                and voice_frames >= min_voice_frames
                                and silence_after_voice >= end_silence_frames
                            )
                        ):
                            break
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    stderr = process.stderr.read() if process.stderr is not None else b""
                    self.last_returncode = process.returncode
                    self.last_stderr = stderr.decode("utf-8", errors="replace").strip()
        except TimeoutError as exc:
            self.last_returncode = None
            self.last_stderr = str(exc)
            self.last_stdout_bytes = 0
            self.last_chunks = []
            return []
        payload = b"".join(captured_frames if triggered else all_frames)
        self.last_stdout_bytes = len(payload)
        self.last_vad_triggered = triggered
        self.last_vad_frame_count = len(all_frames)
        self.last_vad_voice_frame_count = voice_frames
        self.last_vad_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        if not payload:
            self.last_chunks = []
            return []
        self.last_chunks = [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)]
        return list(self.last_chunks)

    def _endpoint_policy(self, *, max_duration_s: int) -> VadEndpointPolicy:
        return VadEndpointPolicy(
            rms_threshold=self.vad_rms_threshold,
            frame_ms=self.vad_frame_ms,
            min_voice_ms=self.vad_min_voice_ms,
            end_silence_ms=self.vad_end_silence_ms,
            min_capture_ms=self.vad_min_capture_ms,
            max_capture_ms=max(1, int(max_duration_s * 1000)),
            fallback_rms_threshold=self.vad_miss_rms_threshold,
        )

    def _run_arecord(self, command: list[str]) -> bytes:
        self.last_command = list(command)
        attempts = max(1, self.retry_count + 1)
        payload = b""
        try:
            with self._capture_lock():
                for attempt in range(attempts):
                    completed = subprocess.run(command, capture_output=True, check=False)
                    payload = completed.stdout or b""
                    self._record_result(completed=completed, payload=payload)
                    if completed.returncode == 0 and payload:
                        return payload
                    if attempt + 1 < attempts:
                        time.sleep(max(0.0, self.retry_delay_s))
                return payload
        except TimeoutError as exc:
            self.last_returncode = None
            self.last_stderr = str(exc)
            self.last_stdout_bytes = 0
            return b""

    def _record_result(self, *, completed: subprocess.CompletedProcess[bytes], payload: bytes) -> None:
        self.last_returncode = completed.returncode
        stderr = completed.stderr or b""
        self.last_stderr = stderr.decode("utf-8", errors="replace").strip()
        self.last_stdout_bytes = len(payload)

    def _frame_bytes(self) -> int:
        bytes_per_sample = 2
        frame_bytes = int(self.sample_rate * self.channels * bytes_per_sample * max(1, self.vad_frame_ms) / 1000)
        alignment = max(1, self.channels * bytes_per_sample)
        return max(alignment, frame_bytes - (frame_bytes % alignment))

    def _capture_lock(self):
        class _NoopLock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-Linux developer machines
            return _NoopLock()

        capture = self

        class _FileLock:
            def __init__(self) -> None:
                self._handle = None

            def __enter__(self):
                started = time.monotonic()
                self._handle = open(capture.lock_path, "a+", encoding="utf-8")
                while True:
                    try:
                        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return self
                    except BlockingIOError:
                        if time.monotonic() - started >= capture.lock_timeout_s:
                            raise TimeoutError(f"timed out waiting for audio capture lock: {capture.lock_path}")
                        time.sleep(0.1)

            def __exit__(self, exc_type, exc, traceback):
                if self._handle is not None:
                    try:
                        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._handle.close()
                return False

        return _FileLock()


@dataclass(slots=True)
class EarStreamProcessor:
    capture: object
    recognizer: object
    last_capture_elapsed_ms: float = 0.0
    last_decode_elapsed_ms: float = 0.0
    last_transcribe_elapsed_ms: float = 0.0

    def transcribe_window(
        self,
        *,
        chunk_count: int,
        session_id: str,
        actor_id: str,
    ) -> AudioTranscriptFinal:
        started = time.perf_counter()
        capture_started = time.perf_counter()
        if hasattr(self.capture, "read_window"):
            chunks = list(self.capture.read_window(chunk_count))
        else:
            chunks = list(self.capture.read_chunks(chunk_count))
        self.last_capture_elapsed_ms = round((time.perf_counter() - capture_started) * 1000, 2)
        vad_missed = bool(getattr(self.capture, "streaming_vad", False)) and not bool(
            getattr(self.capture, "last_vad_triggered", True)
        )
        should_transcribe = bool(chunks)
        if vad_missed:
            stats = pcm_signal_stats(chunks, channels=int(getattr(self.capture, "channels", 1) or 1))
            fallback_enabled = bool(getattr(self.capture, "transcribe_vad_miss", False))
            fallback_threshold = float(getattr(self.capture, "vad_miss_rms_threshold", 0.0) or 0.0)
            should_transcribe = fallback_enabled and float(stats.get("rms_level", 0.0) or 0.0) >= fallback_threshold
        if should_transcribe:
            decode_started = time.perf_counter()
            text = self.recognizer.transcribe(
                chunks,
                sample_rate=self.capture.sample_rate,
                channels=self.capture.channels,
            )
            self.last_decode_elapsed_ms = round((time.perf_counter() - decode_started) * 1000, 2)
        else:
            text = ""
            self.last_decode_elapsed_ms = 0.0
        self.last_transcribe_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text=text,
            session_id=session_id,
            actor_id=actor_id,
        )


def pcm_signal_stats(pcm_chunks: list[bytes], *, channels: int) -> dict[str, float | int | bool]:
    samples = array("h")
    for chunk in pcm_chunks:
        chunk_bytes = chunk[: len(chunk) - (len(chunk) % 2)]
        if chunk_bytes:
            samples.frombytes(chunk_bytes)
    if channels > 1 and samples:
        mono = array("h")
        for index in range(0, len(samples), channels):
            frame = samples[index : index + channels]
            if frame:
                mono.append(int(sum(frame) / len(frame)))
        samples = mono
    if not samples:
        return {
            "sample_count": 0,
            "peak_level": 0.0,
            "rms_level": 0.0,
            "dbfs": -120.0,
            "voice_activity": False,
        }
    peak = max(abs(sample) for sample in samples) / 32768.0
    rms = math.sqrt(sum(float(sample) * float(sample) for sample in samples) / len(samples)) / 32768.0
    dbfs = 20.0 * math.log10(max(rms, 1e-6))
    return {
        "sample_count": len(samples),
        "peak_level": round(peak, 6),
        "rms_level": round(rms, 6),
        "dbfs": round(dbfs, 2),
        "voice_activity": rms >= 0.015,
    }
