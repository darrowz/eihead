from __future__ import annotations

import sys

from eihead.eye import adapters

GStreamerHailoDetector = adapters.GStreamerHailoDetector
GStreamerHailoFrameSource = adapters.GStreamerHailoFrameSource
GStreamerHailoRealtimeAdapter = adapters.GStreamerHailoRealtimeAdapter
GStreamerHailoRealtimeConfig = adapters.GStreamerHailoRealtimeConfig
RealtimeVisionFrame = adapters.RealtimeVisionFrame
normalize_hailo_detection = adapters.normalize_hailo_detection


def test_adapter_classes_are_exported_from_eihead_eye_package() -> None:
    from eihead.eye import GStreamerHailoRealtimeAdapter as ExportedAdapter
    from eihead.eye import GStreamerHailoRealtimeConfig as ExportedConfig
    from eihead.eye import GStreamerAppSinkFrameReader as ExportedReader
    from eihead.eye import parse_hailo_detections as exported_parser

    assert ExportedAdapter is GStreamerHailoRealtimeAdapter
    assert ExportedConfig is GStreamerHailoRealtimeConfig
    assert ExportedReader.__name__ == "GStreamerAppSinkFrameReader"
    assert callable(exported_parser)


def test_gstreamer_hailo_config_exposes_realtime_pipeline_fields() -> None:
    config = GStreamerHailoRealtimeConfig(
        camera_device="/dev/video2",
        hailo_device="/dev/hailo1",
        width=1280,
        height=720,
        framerate=60,
        hef_path="/opt/models/face.hef",
    )

    fields = config.pipeline_fields()
    pipeline = config.build_pipeline_description()

    assert config.mode == "realtime_stream"
    assert fields["camera_device"] == "/dev/video2"
    assert fields["hailo_device"] == "/dev/hailo1"
    assert fields["caps"] == "video/x-raw,width=1280,height=720,framerate=60/1"
    assert fields["source"].startswith("v4l2src")
    assert fields["scale"] == "videoscale"
    assert fields["inference_caps"] == "video/x-raw,format=RGB,width=640,height=640"
    assert fields["inference"].startswith("hailonet")
    assert "device=/dev/hailo1" not in fields["inference"]
    assert "/opt/models/face.hef" in fields["inference"]
    assert "/dev/video2" in pipeline
    assert "device=/dev/hailo1" not in pipeline
    assert "filesrc" not in pipeline
    assert "compat_static_frame" not in pipeline


def test_gstreamer_hailo_pipeline_uses_device_id_only_when_configured() -> None:
    config = GStreamerHailoRealtimeConfig(
        hailo_device="/dev/hailo0",
        hailo_device_id="0000:01:00.0",
        hef_path="/opt/models/personface.hef",
    )

    fields = config.pipeline_fields()
    pipeline = config.build_pipeline_description()

    assert "device-id=0000:01:00.0" in fields["inference"]
    assert "device=/dev/hailo0" not in fields["inference"]
    assert "device-id=0000:01:00.0" in pipeline
    assert "videoscale" in pipeline
    assert "video/x-raw,format=RGB,width=640,height=640" in pipeline


def test_device_paths_are_configured_without_touching_hardware() -> None:
    default_config = GStreamerHailoRealtimeConfig()
    custom_config = GStreamerHailoRealtimeConfig(camera_device="/dev/video9", hailo_device="/dev/hailo9")

    assert default_config.device_paths == ("/dev/video0", "/dev/hailo0")
    assert custom_config.device_paths == ("/dev/video9", "/dev/hailo9")


def test_missing_hardware_reports_not_wired_instead_of_fake_ok() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: False,
        gst_available=lambda: True,
    )

    status = adapter.status()

    assert status.status == "not_wired"
    assert status.backend == "gstreamer_hailo"
    assert status.placeholder is True
    assert status.not_wired is True
    assert status.frame_count == 0
    assert status.detection_count == 0
    assert "missing realtime devices" in status.message
    assert "/dev/video0" in status.message
    assert "/dev/hailo0" in status.message


def test_missing_gstreamer_backend_reports_not_wired_without_importing_gst() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: False,
    )

    status = adapter.status()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "GStreamer backend is not installed" in status.message


def test_default_gstreamer_probe_requires_importable_gst_even_when_gi_spec_exists(monkeypatch) -> None:
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "gi", None)

    assert adapters._default_gst_available() is False


def test_default_gstreamer_probe_requires_hailofilter_element(monkeypatch) -> None:
    class FakeGI:
        @staticmethod
        def require_version(*_args) -> None:
            return None

    class FakeElementFactory:
        @staticmethod
        def find(name: str) -> object | None:
            if name == "hailofilter":
                return None
            return object()

    class FakeGst:
        ElementFactory = FakeElementFactory

        @staticmethod
        def init(_argv) -> None:
            return None

    def fake_import_module(name: str):
        if name == "gi":
            return FakeGI()
        if name == "gi.repository.Gst":
            return FakeGst()
        raise AssertionError(name)

    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(adapters.importlib, "import_module", fake_import_module)

    assert adapters._default_gst_available() is False


def test_ready_adapter_without_frame_reports_waiting_not_ok() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: None,
        detection_reader=lambda _frame: [],
        clock=lambda: 100.0,
    )

    status = adapter.poll()

    assert status.status == "waiting_for_frame"
    assert status.not_wired is False
    assert status.placeholder is False
    assert status.frame_count == 0
    assert "no realtime frame available" in status.message


def test_frame_reader_exception_reports_degraded_without_raising() -> None:
    def failing_frame_reader():
        raise RuntimeError("camera appsink failed")

    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=failing_frame_reader,
        detection_reader=lambda _frame: [],
    )

    status = adapter.poll()

    assert status.status == "degraded"
    assert status.not_wired is False
    assert status.degraded is True
    assert status.stream_ready is False
    assert status.status_reason == "frame_reader_failed"
    assert "frame reader failed" in status.degraded_reason
    assert "frame reader failed" in status.message
    assert "RuntimeError" in status.message


def test_detection_reader_exception_reports_degraded_without_raising() -> None:
    def failing_detection_reader(_frame):
        raise RuntimeError("hailo parser failed")

    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: {"frame_id": "cam-1", "timestamp": 42.0},
        detection_reader=failing_detection_reader,
    )

    status = adapter.poll()

    assert status.status == "degraded"
    assert status.not_wired is False
    assert status.degraded is True
    assert status.stream_ready is False
    assert status.status_reason == "detection_reader_failed"
    assert "detection reader failed" in status.degraded_reason
    assert "detection reader failed" in status.message
    assert "RuntimeError" in status.message


def test_ready_hardware_without_frame_reader_reports_not_wired() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        detection_reader=lambda _frame: [],
    )

    status = adapter.status()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "frame reader is not wired" in status.message


def test_ready_hardware_without_detection_reader_reports_not_wired() -> None:
    adapter = GStreamerHailoRealtimeAdapter(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: {"frame_id": "cam-1", "timestamp": 42.0},
    )

    status = adapter.poll()

    assert status.status == "not_wired"
    assert status.not_wired is True
    assert "detection reader is not wired" in status.message


def test_frame_source_primary_path_is_realtime_not_static_image() -> None:
    source = GStreamerHailoFrameSource(
        GStreamerHailoRealtimeConfig(camera_device="/dev/video3", hailo_device="/dev/hailo3"),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        frame_reader=lambda: {"frame_id": "cam-1", "timestamp": 42.0, "payload": b"raw", "width": 800, "height": 600},
    )

    frame = source.next_frame()

    assert frame is not None
    assert frame.mode == "realtime_stream"
    assert frame.source == "gstreamer_hailo"
    assert frame.frame_id == "cam-1"
    assert frame.width == 800
    assert frame.height == 600
    assert frame.payload == b"raw"
    assert frame.metadata["camera_device"] == "/dev/video3"
    assert not hasattr(source.config, "static_image_path")


def test_detector_normalizes_hailo_results_for_realtime_status_consumers() -> None:
    frame = RealtimeVisionFrame(
        frame_id="cam-2",
        timestamp=50.0,
        width=200,
        height=400,
        source="gstreamer_hailo",
    )
    detector = GStreamerHailoDetector(
        GStreamerHailoRealtimeConfig(model_id="face-yolo"),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        detection_reader=lambda _frame: [
            {
                "label": "face",
                "score": "0.925",
                "box": {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220},
                "class_id": "7",
                "track_id": 12,
                "attributes": {"source_tensor": "hailo0"},
            }
        ],
    )

    payload = detector.detect(frame)[0]

    assert payload["label"] == "face"
    assert payload["score"] == 0.925
    assert payload["bbox"] == {"x_min": 0.05, "y_min": 0.05, "x_max": 0.55, "y_max": 0.55}
    assert payload["confidence"] == 0.925
    assert payload["class_id"] == 7
    assert payload["track_id"] == 12
    assert payload["source"] == "gstreamer_hailo"
    assert payload["model_id"] == "face-yolo"
    assert payload["attributes"] == {"source_tensor": "hailo0"}
    assert payload["ts"] == 50.0
    assert payload["raw"] == {
        "label": "face",
        "score": "0.925",
        "box": {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220},
        "class_id": "7",
        "track_id": 12,
        "attributes": {"source_tensor": "hailo0"},
    }


def test_detector_detects_empty_input_without_exception() -> None:
    frame = RealtimeVisionFrame(
        frame_id="cam-empty",
        timestamp=0.0,
        width=640,
        height=480,
        source="gstreamer_hailo",
    )
    detector = GStreamerHailoDetector(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        detection_reader=lambda _frame: [],
    )

    detections = detector.detect(frame)

    assert detections == []


def test_detector_normalizes_illegal_inputs_as_empty_detections() -> None:
    frame = RealtimeVisionFrame(
        frame_id="cam-3",
        timestamp=80.0,
        width=320,
        height=240,
        source="gstreamer_hailo",
    )
    detector = GStreamerHailoDetector(
        GStreamerHailoRealtimeConfig(),
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        detection_reader=lambda _frame: [None, 123, "bad-detection", {"score": "naive"}],
    )

    detections = detector.detect(frame)

    assert len(detections) == 4
    for detection in detections:
        assert detection["label"] == "unknown"
        assert detection["score"] == 0.0
        assert detection["ts"] == 80.0
        assert detection["raw"] is not None
        assert "bbox" in detection


def test_detector_normalizes_bbox_pixels_to_unit_space() -> None:
    frame = RealtimeVisionFrame(
        frame_id="cam-4",
        timestamp=100.0,
        width=400,
        height=300,
        source="gstreamer_hailo",
    )
    detection = normalize_hailo_detection(
        {"label": "face", "score": 0.8, "bbox": [40, 30, 200, 150]},
        frame=frame,
        config=GStreamerHailoRealtimeConfig(),
    )

    assert detection["bbox"] == {"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5}


def test_native_gstreamer_adapter_wires_reader_and_hailo_metadata_parser() -> None:
    config = GStreamerHailoRealtimeConfig(
        camera_device="/dev/video0",
        hailo_device="/dev/hailo0",
        hef_path="/opt/models/personface.hef",
        postprocess_so_path="/opt/hailo/libpost.so",
        postprocess_config_path="/opt/hailo/personface.json",
        postprocess_function="filter",
        model_id="personface",
        score_threshold=0.5,
        labels=("person", "face"),
    )
    native_reader = _FakeNativeFrameReader()
    created_kwargs: dict[str, object] = {}

    def frame_reader_factory(**kwargs):
        created_kwargs.update(kwargs)
        return native_reader

    adapter = GStreamerHailoRealtimeAdapter.from_native_gstreamer(
        config,
        device_exists=lambda _path: True,
        gst_available=lambda: True,
        clock=lambda: 100.25,
        frame_reader_factory=frame_reader_factory,
        hailo_module_loader=lambda: _FakeHailoModule(),
    )

    status = adapter.poll()
    payload = status.to_dict()

    assert native_reader.started == 1
    assert created_kwargs["hailofilter_config_path"] == "/opt/hailo/personface.json"
    assert status.status == "ok"
    assert payload["devices"] == {"camera": "/dev/video0", "hailo": "/dev/hailo0"}
    assert payload["pipeline"]["inference"].startswith("hailonet")
    assert "device=/dev/hailo0" not in payload["pipeline"]["inference"]
    assert "/opt/models/personface.hef" in payload["pipeline"]["inference"]
    assert "/opt/hailo/personface.json" in payload["pipeline"]["postprocess"]
    assert payload["readiness_message"] == "realtime adapter is wired"
    assert payload["parse_error_count"] == 0
    assert payload["parse_errors"] == []
    assert payload["detections"][0]["label"] == "face"
    assert payload["detections"][0]["score"] == 0.88
    assert payload["stream_ready"] is True
    assert payload["stale"] is False
    assert payload["degraded"] is False
    assert payload["detection_boxes"] == [{"x_min": 0.2, "y_min": 0.1, "x_max": 0.6, "y_max": 0.9}]
    assert payload["detection_scores"] == [0.88]


class _FakeNativeFrameReader:
    def __init__(self) -> None:
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def read_frame(self) -> RealtimeVisionFrame:
        return RealtimeVisionFrame(
            frame_id="native-1",
            timestamp=100.0,
            width=640,
            height=480,
            source="gstreamer_hailo",
            payload=_FakeSample(),
        )


class _FakeSample:
    def get_buffer(self) -> object:
        return object()


class _FakeHailoModule:
    HAILO_DETECTION = "HAILO_DETECTION"

    def get_roi_from_buffer(self, _buffer: object) -> "_FakeROI":
        return _FakeROI()


class _FakeROI:
    def get_objects_typed(self, kind: object) -> list[object]:
        if kind != "HAILO_DETECTION":
            return []
        return [_FakeHailoDetection()]


class _FakeHailoDetection:
    def get_label(self) -> str:
        return ""

    def get_class_id(self) -> int:
        return 1

    def get_confidence(self) -> float:
        return 0.88

    def get_bbox(self) -> "_FakeHailoBBox":
        return _FakeHailoBBox()

    def get_objects_typed(self, _kind: object) -> list[object]:
        return []


class _FakeHailoBBox:
    def xmin(self) -> float:
        return 0.2

    def ymin(self) -> float:
        return 0.1

    def xmax(self) -> float:
        return 0.6

    def ymax(self) -> float:
        return 0.9
