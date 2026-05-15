from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Generator

from eihead.eye import gstreamer
from eihead.eye.realtime import RealtimeVisionFrame


def test_reader_init_does_not_import_gstreamer_modules(monkeypatch) -> None:
    def should_not_be_called() -> tuple[object, object, object, object]:
        raise AssertionError("GStreamer modules should be imported lazily on start()")

    monkeypatch.setattr(gstreamer, "_load_gstreamer_modules", should_not_be_called)
    reader = gstreamer.GStreamerAppSinkFrameReader()
    assert reader.mode == "realtime_stream"


def test_pipeline_text_uses_realtime_v4l2_source_and_hailo_blocks() -> None:
    reader = gstreamer.GStreamerAppSinkFrameReader(camera_device="/dev/video0")
    pipeline_text = reader.pipeline_text()

    assert "v4l2src device=/dev/video0" in pipeline_text
    assert "hailonet" in pipeline_text
    assert "hailofilter" in pipeline_text
    assert "appsink" in pipeline_text
    assert "filesrc" not in pipeline_text
    assert "compat_static_frame" not in pipeline_text


def test_start_starts_and_stop_sets_pipeline_null(monkeypatch) -> None:
    sample = _fake_sample(frame_id="cam-1", pts=1_500_000_000, width=640, height=480)
    fake = _FakeGstreamerEnvironment(sample=sample)
    with _patch_loader(monkeypatch, fake):
        reader = gstreamer.GStreamerAppSinkFrameReader(appsink_name="unit_sink")
        reader.start()
        assert reader._pipeline is fake.pipeline
        assert reader._appsink is fake.appsink
        assert fake.pipeline.states[-1] == "PLAYING"

        reader.stop()
        assert reader._pipeline is None
        assert reader._appsink is None
        assert fake.pipeline.states[-1] == "NULL"


def test_read_frame_returns_realtime_vision_frame(monkeypatch) -> None:
    sample = _fake_sample(frame_id="cam-1", pts=2_500_000_000, width=1920, height=1080)
    fake = _FakeGstreamerEnvironment(sample=sample)
    with _patch_loader(monkeypatch, fake):
        reader = gstreamer.GStreamerAppSinkFrameReader(
            appsink_name="unit_sink",
            backend="unit-backend",
            clock=lambda: 1234.5,
        )
        reader.start()
        frame = reader.read_frame()

    assert isinstance(frame, RealtimeVisionFrame)
    assert frame.frame_id == "cam-1"
    assert frame.timestamp == 1234.5
    assert frame.width == 1920
    assert frame.height == 1080
    assert frame.source == "gstreamer_hailo"
    assert frame.mode == "realtime_stream"
    assert frame.metadata["backend"] == "unit-backend"
    assert frame.metadata["pipeline"]
    assert frame.metadata["frame_index"] == 1
    assert frame.metadata["camera_device"] == "/dev/video0"
    assert frame.metadata["gst_buffer_timestamp_s"] == 2.5
    assert fake.appsink.emitted == [("try-pull-sample", (5_000_000_000,))]


def test_read_frame_returns_none_when_no_sample(monkeypatch) -> None:
    fake = _FakeGstreamerEnvironment(sample=None)
    with _patch_loader(monkeypatch, fake):
        reader = gstreamer.GStreamerAppSinkFrameReader(appsink_name="unit_sink", sample_timeout_s=0.25)
        reader.start()
        assert reader.read_frame() is None
        assert fake.appsink.emitted == [("try-pull-sample", (250_000_000,))]


class _FakeGstreamerEnvironment:
    def __init__(self, sample: object | None) -> None:
        self.gi = SimpleNamespace(
            require_version=lambda *_args, **_kwargs: None,
        )
        self.glib = SimpleNamespace()
        self.hailo = SimpleNamespace()

        class _State:
            PLAYING = "PLAYING"
            NULL = "NULL"

        class _StateChangeReturn:
            FAILURE = "FAILURE"

        class _AppSink:
            def __init__(self, payload: object | None) -> None:
                self.payload = payload
                self.emitted: list[tuple[str, tuple[object, ...]]] = []

            def emit(self, name: str, *args: object) -> object | None:
                self.emitted.append((name, args))
                if name == "try-pull-sample":
                    return self.payload
                return None

        class _Pipeline:
            def __init__(self, appsink: _AppSink) -> None:
                self.appsink = appsink
                self.states: list[str] = []
                self.description: str | None = None

            def get_by_name(self, name: str):
                if name == "unit_sink":
                    return self.appsink
                return None

            def set_state(self, state: str) -> str:
                self.states.append(state)
                return "ok"

        class _Gst:
            State = _State
            StateChangeReturn = _StateChangeReturn

            def __init__(self, pipeline: _Pipeline) -> None:
                self.init_called = False
                self.pipeline = pipeline

            def init(self, _argv: object) -> None:
                self.init_called = True

            def parse_launch(self, description: str) -> _Pipeline:
                self.pipeline.description = description
                return self.pipeline

        self.appsink = _AppSink(sample)
        self.pipeline = _Pipeline(self.appsink)
        self.gst = _Gst(self.pipeline)

    def load(self) -> tuple[SimpleNamespace, object, object, object]:
        return self.gi, self.gst, self.glib, self.hailo


@contextmanager
def _patch_loader(monkeypatch, fake: _FakeGstreamerEnvironment) -> Generator[None, None, None]:
    original = gstreamer._load_gstreamer_modules
    monkeypatch.setattr(gstreamer, "_load_gstreamer_modules", fake.load)
    try:
        yield None
    finally:
        monkeypatch.setattr(gstreamer, "_load_gstreamer_modules", original)


def _fake_sample(*, frame_id: str, pts: float, width: int, height: int):
    class _Buffer:
        def __init__(self, pts_value: float) -> None:
            self.pts = pts_value
            self.offset = 123

    class _Structure:
        def __init__(self, width_value: int, height_value: int) -> None:
            self.width_value = width_value
            self.height_value = height_value

        def get_int(self, key: str):
            if key == "width":
                return True, self.width_value
            if key == "height":
                return True, self.height_value
            return False, 0

        def get_value(self, key: str):
            if key == "width":
                return self.width_value
            if key == "height":
                return self.height_value
            return None

    class _Caps:
        def __init__(self, width_value: int, height_value: int) -> None:
            self._structure = _Structure(width_value, height_value)

        def get_structure(self, _index: int) -> _Structure:
            return self._structure

    class _Sample:
        def __init__(self, pts_value: float) -> None:
            self._buffer = _Buffer(pts_value)
            self._caps = _Caps(width, height)

        def get_buffer(self) -> _Buffer:
            return self._buffer

        def get_caps(self) -> _Caps:
            return self._caps

        def get_frame_id(self) -> str:
            return frame_id

    return _Sample(pts)
