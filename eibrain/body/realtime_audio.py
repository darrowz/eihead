"""Realtime audio primitives for JoyInside-like wake detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Iterable

from eibrain.body.ear_stream import pcm_signal_stats
from eibrain.protocol.observations import AudioTranscriptFinal


@dataclass(frozen=True, slots=True)
class RingBufferSnapshot:
    chunks: list[bytes]
    duration_ms: int
    sample_rate: int
    channels: int
    chunk_count: int
    sequence: int
    started_at_s: float | None
    ended_at_s: float | None


@dataclass(frozen=True, slots=True)
class _BufferedChunk:
    payload: bytes
    duration_ms: int
    captured_at_s: float
    sequence: int


class PcmRingBuffer:
    """Thread-safe PCM ring buffer capped by audio duration."""

    def __init__(self, *, max_duration_ms: int, sample_rate: int, channels: int) -> None:
        self.max_duration_ms = max(1, int(max_duration_ms))
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._chunks: deque[_BufferedChunk] = deque()
        self._duration_ms = 0
        self._sequence = 0
        self._dropped_oldest_chunks = 0
        self._dropped_oldest_duration_ms = 0
        self._lock = threading.RLock()

    def append(self, payload: bytes, *, duration_ms: int, captured_at_s: float | None = None) -> None:
        if not payload:
            return
        duration = max(1, int(duration_ms))
        captured_at = time.time() if captured_at_s is None else float(captured_at_s)
        with self._lock:
            self._sequence += 1
            self._chunks.append(
                _BufferedChunk(
                    payload=bytes(payload),
                    duration_ms=duration,
                    captured_at_s=captured_at,
                    sequence=self._sequence,
                )
            )
            self._duration_ms += duration
            self._trim_locked()

    def snapshot(self, *, duration_ms: int | None = None) -> RingBufferSnapshot:
        with self._lock:
            requested_ms = self.max_duration_ms if duration_ms is None else max(1, int(duration_ms))
            selected: deque[_BufferedChunk] = deque()
            total_ms = 0
            for chunk in reversed(self._chunks):
                if total_ms >= requested_ms:
                    break
                selected.appendleft(chunk)
                total_ms += chunk.duration_ms
            chunks = list(selected)
            return RingBufferSnapshot(
                chunks=[chunk.payload for chunk in chunks],
                duration_ms=sum(chunk.duration_ms for chunk in chunks),
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_count=len(chunks),
                sequence=self._sequence,
                started_at_s=chunks[0].captured_at_s if chunks else None,
                ended_at_s=chunks[-1].captured_at_s if chunks else None,
            )

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "max_buffer_ms": self.max_duration_ms,
                "buffer_ms": self._duration_ms,
                "chunk_count": len(self._chunks),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sequence": self._sequence,
                "dropped_oldest_chunks": self._dropped_oldest_chunks,
                "dropped_oldest_duration_ms": self._dropped_oldest_duration_ms,
            }

    def _trim_locked(self) -> None:
        while self._duration_ms > self.max_duration_ms and self._chunks:
            old = self._chunks.popleft()
            self._duration_ms -= old.duration_ms
            self._dropped_oldest_chunks += 1
            self._dropped_oldest_duration_ms += old.duration_ms


class RealtimeAudioCaptureWorker:
    """Continuously pulls PCM chunks from a source into a ring buffer."""

    def __init__(
        self,
        *,
        ring_buffer: PcmRingBuffer,
        chunk_source: object,
        chunk_duration_ms: int,
        idle_sleep_s: float = 0.01,
    ) -> None:
        self.ring_buffer = ring_buffer
        self.chunk_source = chunk_source
        self.chunk_duration_ms = max(1, int(chunk_duration_ms))
        self.idle_sleep_s = max(0.0, float(idle_sleep_s))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error = ""
        self._read_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        starter = getattr(self.chunk_source, "start", None)
        if callable(starter):
            starter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="realtime-audio-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        stopper = getattr(self.chunk_source, "stop", None)
        if callable(stopper):
            stopper()

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "read_count": self._read_count,
            "last_error": self._last_error,
            "chunk_duration_ms": self.chunk_duration_ms,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._read_chunk()
            except Exception as exc:  # pragma: no cover - hardware/runtime boundary
                self._last_error = str(exc)
                time.sleep(max(0.05, self.idle_sleep_s))
                continue
            if not payload:
                time.sleep(self.idle_sleep_s)
                continue
            self._read_count += 1
            self._last_error = ""
            self.ring_buffer.append(payload, duration_ms=self.chunk_duration_ms)

    def _read_chunk(self) -> bytes:
        source = self.chunk_source
        for name in ("read_chunk", "read"):
            reader = getattr(source, name, None)
            if callable(reader):
                return bytes(reader() or b"")
        if callable(source):
            return bytes(source() or b"")
        raise TypeError("chunk_source must expose read_chunk(), read(), or be callable")


class ArecordRawChunkSource:
    """Reads fixed-size raw PCM frames from arecord."""

    def __init__(
        self,
        *,
        device: str,
        sample_rate: int,
        channels: int,
        frame_ms: int = 120,
        command: str = "arecord",
    ) -> None:
        self.device = device
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frame_ms = max(20, int(frame_ms))
        self.command = command
        self._process: subprocess.Popen[bytes] | None = None
        self._last_error = ""
        self._read_count = 0

    @property
    def frame_bytes(self) -> int:
        bytes_per_sample = 2
        alignment = max(1, self.channels * bytes_per_sample)
        size = int(self.sample_rate * self.channels * bytes_per_sample * self.frame_ms / 1000)
        return max(alignment, size - (size % alignment))

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [
                self.command,
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
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        if process.stderr is not None:
            self._last_error = process.stderr.read().decode("utf-8", errors="replace").strip()

    def read_chunk(self) -> bytes:
        self.start()
        process = self._process
        if process is None or process.stdout is None:
            return b""
        payload = process.stdout.read(self.frame_bytes)
        if payload:
            self._read_count += 1
            self._last_error = ""
            return payload
        if process.poll() is not None:
            if process.stderr is not None:
                self._last_error = process.stderr.read().decode("utf-8", errors="replace").strip()
            self._process = None
        return b""

    def snapshot(self) -> dict[str, object]:
        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_ms": self.frame_ms,
            "frame_bytes": self.frame_bytes,
            "running": self._process is not None and self._process.poll() is None,
            "read_count": self._read_count,
            "last_error": self._last_error,
        }


class RealtimeWakeDetector:
    """Quasi-streaming wake detector that decodes recent ring-buffer audio."""

    def __init__(
        self,
        *,
        ring_buffer: PcmRingBuffer,
        recognizer: object,
        wake_words: Iterable[str],
        transcript_replacements: dict[str, str] | list[dict[str, str]] | None = None,
        session_id: str = "voice-dialogue-loop",
        actor_id: str = "darrow",
        target_id: str | None = None,
        language: str = "zh",
        lookback_ms: int = 2400,
        min_buffer_ms: int = 480,
        min_rms_level: float = 0.0,
        poll_interval_s: float = 0.25,
    ) -> None:
        self.ring_buffer = ring_buffer
        self.recognizer = recognizer
        self.wake_words = tuple(str(word) for word in wake_words if str(word))
        self.transcript_replacements = transcript_replacements or {}
        self.session_id = session_id
        self.actor_id = actor_id
        self.target_id = target_id
        self.language = language
        self.lookback_ms = max(1, int(lookback_ms))
        self.min_buffer_ms = max(1, int(min_buffer_ms))
        self.min_rms_level = max(0.0, float(min_rms_level))
        self.poll_interval_s = max(0.01, float(poll_interval_s))
        self._queue: Queue[AudioTranscriptFinal] = Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sequence = 0
        self._poll_count = 0
        self._decode_count = 0
        self._emitted_count = 0
        self._last_error = ""
        self._last_text = ""
        self._last_audio_stats: dict[str, float | int | bool] = {}
        self._last_detected_at_ts: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="realtime-wake-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def poll_once(self) -> AudioTranscriptFinal | None:
        self._poll_count += 1
        snapshot = self.ring_buffer.snapshot(duration_ms=self.lookback_ms)
        if snapshot.sequence == self._last_sequence or snapshot.duration_ms < self.min_buffer_ms:
            return None
        self._last_sequence = snapshot.sequence
        self._last_audio_stats = pcm_signal_stats(snapshot.chunks, channels=snapshot.channels)
        if self.min_rms_level > 0.0 and float(self._last_audio_stats.get("rms_level", 0.0) or 0.0) < self.min_rms_level:
            return None
        try:
            text = self._normalize_text(self._decode_snapshot(snapshot))
        except Exception as exc:  # pragma: no cover - recognizer boundary
            self._last_error = str(exc)
            return None
        self._decode_count += 1
        self._last_error = ""
        self._last_text = text
        if not text or not self._contains_wake_word(text):
            return None
        observation = AudioTranscriptFinal(
            ts=time.time(),
            source="ear.realtime_wake",
            text=text,
            language=self.language,
            session_id=self.session_id,
            actor_id=self.actor_id,
            target_id=self.target_id,
        )
        self._queue.put(observation)
        self._emitted_count += 1
        self._last_detected_at_ts = observation.ts
        return observation

    def next_transcript(self, *, timeout_s: float = 0.0) -> AudioTranscriptFinal | None:
        try:
            return self._queue.get(timeout=max(0.0, float(timeout_s)))
        except Empty:
            return None

    def snapshot(self) -> dict[str, object]:
        stats = self.ring_buffer.stats()
        return {
            "enabled": True,
            "running": self._thread is not None and self._thread.is_alive(),
            "buffer_ms": stats.get("buffer_ms", 0),
            "max_buffer_ms": stats.get("max_buffer_ms", 0),
            "chunk_count": stats.get("chunk_count", 0),
            "sample_rate": stats.get("sample_rate", 0),
            "channels": stats.get("channels", 0),
            "wake_detector": {
                "poll_count": self._poll_count,
                "decode_count": self._decode_count,
                "emitted_count": self._emitted_count,
                "lookback_ms": self.lookback_ms,
                "min_buffer_ms": self.min_buffer_ms,
                "min_rms_level": self.min_rms_level,
                "last_audio_stats": dict(self._last_audio_stats),
                "last_text": self._last_text,
                "last_error": self._last_error,
                "last_detected_at_ts": self._last_detected_at_ts,
            },
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.poll_interval_s)

    def _decode_snapshot(self, snapshot: RingBufferSnapshot) -> str:
        recognizer = self.recognizer
        for name in ("transcribe_audio_chunks", "transcribe_chunks", "transcribe"):
            method = getattr(recognizer, name, None)
            if not callable(method):
                continue
            try:
                value = method(snapshot.chunks, sample_rate=snapshot.sample_rate, channels=snapshot.channels)
            except TypeError:
                value = method(snapshot.chunks)
            return self._text_from_result(value)
        if callable(recognizer):
            return self._text_from_result(recognizer(snapshot.chunks))
        raise TypeError("recognizer must expose transcribe_audio_chunks(), transcribe_chunks(), transcribe(), or be callable")

    def _normalize_text(self, value: str) -> str:
        text = str(value or "").strip()
        replacements = self.transcript_replacements
        if isinstance(replacements, dict):
            for find_text, replace_text in replacements.items():
                if find_text:
                    text = text.replace(str(find_text), str(replace_text))
        else:
            for item in replacements:
                if not isinstance(item, dict):
                    continue
                find_text = str(item.get("find", ""))
                replace_text = str(item.get("replace", ""))
                if find_text:
                    text = text.replace(find_text, replace_text)
        return text.strip()

    def _contains_wake_word(self, text: str) -> bool:
        return any(word in text for word in self.wake_words)

    @staticmethod
    def _text_from_result(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return str(value.get("text") or value.get("transcript") or "")
        return str(getattr(value, "text", "") or getattr(value, "transcript", "") or "")


class RealtimeWakeAudioPipeline:
    """Composes capture, ring buffer, and wake detector behind one source API."""

    def __init__(
        self,
        *,
        ring_buffer: PcmRingBuffer,
        wake_detector: RealtimeWakeDetector,
        capture_worker: RealtimeAudioCaptureWorker | None = None,
    ) -> None:
        self.ring_buffer = ring_buffer
        self.wake_detector = wake_detector
        self.capture_worker = capture_worker

    def start(self) -> None:
        if self.capture_worker is not None:
            self.capture_worker.start()
        self.wake_detector.start()

    def stop(self) -> None:
        self.wake_detector.stop()
        if self.capture_worker is not None:
            self.capture_worker.stop()

    def pause(self) -> None:
        self.stop()

    def resume(self) -> None:
        self.start()

    def next_transcript(self, *, timeout_s: float = 0.0) -> AudioTranscriptFinal | None:
        return self.wake_detector.next_transcript(timeout_s=timeout_s)

    def snapshot(self) -> dict[str, object]:
        payload = self.wake_detector.snapshot()
        if self.capture_worker is not None:
            payload["capture"] = self.capture_worker.snapshot()
        return payload
