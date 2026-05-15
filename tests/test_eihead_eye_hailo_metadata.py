from __future__ import annotations

import importlib.util

import pytest


def test_parse_hailo_detections_reads_metadata_without_importing_hailo_module() -> None:
    metadata = _load_metadata_module()
    hailo = FakeHailoModule(
        detections=[
            FakeDetection(
                label="person",
                class_id=0,
                confidence=0.91,
                bbox=FakeBBox(0.1, 0.2, 0.3, 0.4),
            )
        ]
    )

    parsed = metadata.parse_hailo_detections(
        buffer=object(),
        hailo_module=hailo,
        model_id="personface",
        score_threshold=0.5,
    )

    assert parsed["detections"] == [
        {
            "label": "person",
            "score": 0.91,
            "confidence": 0.91,
            "bbox": {
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.3,
                "y_max": 0.4,
            },
            "class_id": 0,
            "source": "hailo",
            "model_id": "personface",
        }
    ]
    assert parsed["parse_error_count"] == 0
    assert parsed["errors"] == []
    assert hailo.buffer_calls == [True]


def test_parse_hailo_detections_filters_low_scores() -> None:
    metadata = _load_metadata_module()
    hailo = FakeHailoModule(
        detections=[
            FakeDetection(
                label="person",
                class_id=0,
                confidence=0.49,
                bbox=FakeBBox(0.1, 0.2, 0.3, 0.4),
            ),
            FakeDetection(
                label="cat",
                class_id=15,
                confidence=0.88,
                bbox=FakeBBox(0.5, 0.6, 0.7, 0.8),
            ),
        ]
    )

    parsed = metadata.parse_hailo_detections(
        buffer=object(),
        hailo_module=hailo,
        model_id="personface",
        score_threshold=0.5,
    )

    assert parsed["detections"] == [
        {
            "label": "cat",
            "score": 0.88,
            "confidence": 0.88,
            "bbox": {
                "x_min": 0.5,
                "y_min": 0.6,
                "x_max": 0.7,
                "y_max": 0.8,
            },
            "class_id": 15,
            "source": "hailo",
            "model_id": "personface",
        }
    ]


def test_parse_hailo_detections_falls_back_to_labels_and_reads_track_id() -> None:
    metadata = _load_metadata_module()
    hailo = FakeHailoModule(
        detections=[
            FakeDetection(
                label="",
                class_id=1,
                confidence=0.95,
                bbox=FakeBBox(0.11, 0.22, 0.33, 0.44),
                unique_ids=[FakeUniqueId(12)],
            )
        ]
    )

    parsed = metadata.parse_hailo_detections(
        buffer=object(),
        hailo_module=hailo,
        model_id="tracker",
        score_threshold=0.5,
        labels=["person", "bicycle", "car"],
    )

    assert parsed["detections"][0]["label"] == "bicycle"
    assert parsed["detections"][0]["track_id"] == 12


def test_parse_hailo_detections_skips_bad_detection_and_reports_error_by_default() -> None:
    metadata = _load_metadata_module()
    hailo = FakeHailoModule(
        detections=[
            FakeDetection(
                label="person",
                class_id=0,
                confidence=0.9,
                bbox=FakeBBox(0.1, 0.2, 0.3, 0.4),
            ),
            FakeDetection(
                label="broken",
                class_id=99,
                confidence=0.7,
                bbox=BrokenBBox(),
            ),
        ]
    )

    parsed = metadata.parse_hailo_detections(
        buffer=object(),
        hailo_module=hailo,
        model_id="personface",
        score_threshold=0.5,
    )

    assert len(parsed["detections"]) == 1
    assert parsed["detections"][0]["label"] == "person"
    assert parsed["parse_error_count"] == 1
    assert len(parsed["errors"]) == 1
    assert parsed["errors"][0]["index"] == 1
    assert parsed["errors"][0]["exception"] == "AttributeError"


def test_parse_hailo_detections_raises_custom_error_in_strict_mode() -> None:
    metadata = _load_metadata_module()
    hailo = FakeHailoModule(
        detections=[
            FakeDetection(
                label="broken",
                class_id=99,
                confidence=0.7,
                bbox=BrokenBBox(),
            )
        ]
    )

    with pytest.raises(metadata.HailoMetadataParseError, match="failed to parse detection"):
        metadata.parse_hailo_detections(
            buffer=object(),
            hailo_module=hailo,
            model_id="personface",
            score_threshold=0.5,
            strict=True,
        )


def test_parse_hailo_detections_reports_roi_read_failure_by_default() -> None:
    metadata = _load_metadata_module()
    hailo = BrokenHailoModule()

    parsed = metadata.parse_hailo_detections(
        buffer=object(),
        hailo_module=hailo,
        model_id="personface",
        score_threshold=0.5,
    )

    assert parsed["detections"] == []
    assert parsed["parse_error_count"] == 1
    assert parsed["errors"][0]["index"] is None
    assert parsed["errors"][0]["exception"] == "RuntimeError"


def test_parse_hailo_detections_raises_custom_error_when_roi_read_fails_in_strict_mode() -> None:
    metadata = _load_metadata_module()

    with pytest.raises(metadata.HailoMetadataParseError, match="failed to read Hailo ROI metadata"):
        metadata.parse_hailo_detections(
            buffer=object(),
            hailo_module=BrokenHailoModule(),
            model_id="personface",
            score_threshold=0.5,
            strict=True,
        )


def test_parse_hailo_detections_uses_unknown_label_when_class_id_has_no_mapping() -> None:
    metadata = _load_metadata_module()
    hailo = FakeHailoModule(
        detections=[
            FakeDetection(
                label="",
                class_id=8,
                confidence=0.77,
                bbox=FakeBBox(0.2, 0.2, 0.4, 0.5),
            )
        ]
    )

    parsed = metadata.parse_hailo_detections(
        buffer=object(),
        hailo_module=hailo,
        model_id="personface",
        score_threshold=0.5,
        labels=["person", "bicycle"],
    )

    assert parsed["detections"][0]["label"] == "class_8"


def _load_metadata_module():
    spec = importlib.util.spec_from_file_location(
        "eihead_eye_hailo_metadata_under_test",
        "eihead/eye/hailo_metadata.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHailoModule:
    HAILO_DETECTION = "HAILO_DETECTION"

    def __init__(self, *, detections: list[object]) -> None:
        self._detections = detections
        self.buffer_calls: list[bool] = []

    def get_roi_from_buffer(self, buffer: object) -> "FakeROI":
        self.buffer_calls.append(buffer is not None)
        return FakeROI(self._detections)


class BrokenHailoModule:
    HAILO_DETECTION = "HAILO_DETECTION"

    def get_roi_from_buffer(self, _buffer: object) -> "FakeROI":
        raise RuntimeError("roi unavailable")


class FakeROI:
    def __init__(self, detections: list[object]) -> None:
        self._detections = detections

    def get_objects_typed(self, kind: object) -> list[object]:
        if kind == "HAILO_DETECTION":
            return list(self._detections)
        return []


class FakeDetection:
    def __init__(
        self,
        *,
        label: str,
        class_id: int,
        confidence: float,
        bbox: object,
        unique_ids: list[object] | None = None,
    ) -> None:
        self._label = label
        self._class_id = class_id
        self._confidence = confidence
        self._bbox = bbox
        self._unique_ids = unique_ids or []

    def get_label(self) -> str:
        return self._label

    def get_class_id(self) -> int:
        return self._class_id

    def get_confidence(self) -> float:
        return self._confidence

    def get_bbox(self) -> object:
        return self._bbox

    def get_objects_typed(self, kind: object) -> list[object]:
        if kind == "HAILO_UNIQUE_ID":
            return list(self._unique_ids)
        return []


class FakeBBox:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        self._xmin = xmin
        self._ymin = ymin
        self._xmax = xmax
        self._ymax = ymax

    def xmin(self) -> float:
        return self._xmin

    def ymin(self) -> float:
        return self._ymin

    def xmax(self) -> float:
        return self._xmax

    def ymax(self) -> float:
        return self._ymax


class BrokenBBox:
    def xmin(self) -> float:
        raise AttributeError("xmin missing")


class FakeUniqueId:
    def __init__(self, value: int) -> None:
        self._value = value

    def get_id(self) -> int:
        return self._value
