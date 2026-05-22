"""Face embedding providers for visual identity matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from .matcher import FaceEvidence


ImageLoader = Callable[[str], Any]
SessionFactory = Callable[[str], Any]


class OnnxFaceEmbeddingProvider:
    """Extract L2-normalized face embeddings from crop images with ONNX Runtime."""

    provider_id = "onnx_face_embedding"

    def __init__(
        self,
        *,
        model_path: str | Path,
        input_size: tuple[int, int] = (112, 112),
        provider_id: str | None = None,
        session_factory: SessionFactory | None = None,
        image_loader: ImageLoader | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.provider_id = provider_id or self.provider_id
        self._session_factory = session_factory
        self._image_loader = image_loader
        self._session: Any | None = None

    def is_available(self) -> bool:
        if not self.model_path.exists():
            return False
        if self._session_factory is not None:
            return True
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return False
        return True

    def embed(self, evidence: FaceEvidence) -> Sequence[float] | None:
        if not self.is_available() or not evidence.crop_path:
            return None
        crop_path = Path(evidence.crop_path)
        if not crop_path.exists():
            return None
        image = self._load_image(str(crop_path))
        if image is None:
            return None
        tensor = _preprocess_image(image, input_size=self.input_size)
        if tensor is None:
            return None
        session = self._load_session()
        input_name = _session_input_name(session)
        try:
            outputs = session.run(None, {input_name: tensor})
        except Exception:
            return None
        vector = _first_output_vector(outputs)
        return _l2_normalize(vector) if vector else None

    def _load_session(self) -> Any:
        if self._session is not None:
            return self._session
        if self._session_factory is not None:
            self._session = self._session_factory(str(self.model_path))
            return self._session
        import onnxruntime

        self._session = onnxruntime.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        return self._session

    def _load_image(self, crop_path: str) -> Any | None:
        if self._image_loader is not None:
            return self._image_loader(crop_path)
        try:
            import cv2
        except Exception:
            return None
        image = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _preprocess_image(image: Any, *, input_size: tuple[int, int]) -> Any | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    try:
        resized = cv2.resize(image, input_size)
        tensor = resized.astype(np.float32)
        tensor = (tensor - 127.5) / 127.5
        tensor = np.transpose(tensor, (2, 0, 1))
        return tensor.reshape((1, 3, input_size[1], input_size[0])).astype(np.float32)
    except Exception:
        return None


def _session_input_name(session: Any) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("ONNX face embedding model has no inputs")
    return str(inputs[0].name)


def _first_output_vector(outputs: Any) -> tuple[float, ...]:
    if not outputs:
        return ()
    output = outputs[0]
    try:
        array = output.reshape(-1)
    except Exception:
        try:
            array = output[0]
        except Exception:
            return ()
    return tuple(float(value) for value in array)


def _l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    norm = sum(float(value) * float(value) for value in vector) ** 0.5
    if norm == 0.0:
        return ()
    return tuple(float(value) / norm for value in vector)


__all__ = ["OnnxFaceEmbeddingProvider"]
