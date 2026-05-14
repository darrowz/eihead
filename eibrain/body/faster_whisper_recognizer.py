"""Reusable faster-whisper recognizer for live audio windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import wave

from eibrain.body.runtime_linux import resolve_faster_whisper_model_path
from eibrain.body.runtime_linux import transcribe_pcm_with_faster_whisper_subprocess


@dataclass(slots=True)
class FasterWhisperRecognizer:
    model_name: str
    language: str = "zh"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = False
    device: str = "cpu"
    python_executable: str = "/usr/bin/python3"
    _model: object | None = field(default=None, init=False)

    def prewarm(self) -> bool:
        self._get_model()
        return True

    def transcribe(self, pcm_chunks: list[bytes], *, sample_rate: int, channels: int) -> str:
        if not pcm_chunks:
            return ""
        if self._model is None and not self._can_import_faster_whisper():
            result = transcribe_pcm_with_faster_whisper_subprocess(
                pcm_bytes=b"".join(pcm_chunks),
                model_name=self.model_name,
                sample_rate=sample_rate,
                channels=channels,
                language=self.language,
                compute_type=self.compute_type,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                python_executable=self.python_executable,
            )
            details = result.get("details", {})
            if result.get("status") == "ok" and isinstance(details, dict):
                return str(details.get("text", "") or "").strip()
            if isinstance(details, dict):
                raise RuntimeError(str(details.get("stderr") or details.get("reason") or "faster_whisper_failed"))
            raise RuntimeError("faster_whisper_failed")
        wav_path = self._write_wav(pcm_chunks=pcm_chunks, sample_rate=sample_rate, channels=channels)
        try:
            segments, _info = self._get_model().transcribe(
                str(wav_path),
                language=self.language or None,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        finally:
            wav_path.unlink(missing_ok=True)

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # pragma: no cover - host dependency

            self._model = WhisperModel(
                resolve_faster_whisper_model_path(self.model_name),
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    @staticmethod
    def _can_import_faster_whisper() -> bool:
        try:
            import faster_whisper  # noqa: F401  # pragma: no cover - optional host dependency
        except Exception:
            return False
        return True

    @staticmethod
    def _write_wav(*, pcm_chunks: list[bytes], sample_rate: int, channels: int) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(pcm_chunks))
        return wav_path
