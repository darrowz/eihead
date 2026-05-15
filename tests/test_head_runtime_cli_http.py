from __future__ import annotations

import io
import json

from eihead.runtime import cli
from eihead.runtime.app import HeadRuntimeApp


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-cli-http-test",
            "organ_count": 3,
            "organs": {
                "ear": {"status": "mock"},
                "eye": {"status": "mock"},
                "neck": {"status": "mock"},
            },
        }


def make_fake_head_runtime(config_path: str) -> HeadRuntimeApp:
    return HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path=config_path)


def test_help_lists_http_and_monitor_subcommands() -> None:
    help_text = cli.build_parser().format_help()

    assert "http" in help_text
    assert "Start the eihead HTTP API" in help_text
    assert "monitor" in help_text
    assert "Start the eihead native monitoring Web" in help_text


def test_http_defaults_use_injected_server_without_blocking() -> None:
    stdout = io.StringIO()
    called: dict[str, object] = {}

    def fake_http_server(*, app: HeadRuntimeApp, host: str, port: int) -> dict[str, object]:
        called["app"] = app
        called["host"] = host
        called["port"] = port
        return {
            "command": "http",
            "runtime": "eihead",
            "status": "serving",
            "host": host,
            "port": port,
            "config_path": app.config_path,
        }

    exit_code = cli.main(
        ["--config", "config/test.yaml", "http"],
        app_factory=make_fake_head_runtime,
        http_server=fake_http_server,
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["command"] == "http"
    assert payload["status"] == "serving"
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 18081
    assert payload["config_path"] == "config/test.yaml"
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 18081
    assert isinstance(called["app"], HeadRuntimeApp)


def test_http_custom_host_and_port_parse_for_service_entrypoint() -> None:
    called: dict[str, object] = {}

    def fake_http_server(*, app: HeadRuntimeApp, host: str, port: int) -> None:
        called["config_path"] = app.config_path
        called["host"] = host
        called["port"] = port
        return None

    payload = cli.dispatch(
        cli.build_parser().parse_args(
            ["--config", "config/honjia.yaml", "http", "--host", "0.0.0.0", "--port", "18081"]
        ),
        app_factory=make_fake_head_runtime,
        http_server=fake_http_server,
    )

    assert payload == {
        "command": "http",
        "runtime": "eihead",
        "status": "stopped",
        "host": "0.0.0.0",
        "port": 18081,
    }
    assert called == {
        "config_path": "config/honjia.yaml",
        "host": "0.0.0.0",
        "port": 18081,
    }


def test_monitor_defaults_use_injected_server_without_blocking() -> None:
    stdout = io.StringIO()
    called: dict[str, object] = {}

    def fake_monitor_server(*, app: HeadRuntimeApp, host: str, port: int) -> dict[str, object]:
        called["app"] = app
        called["host"] = host
        called["port"] = port
        return {
            "command": "monitor",
            "runtime": "eihead",
            "status": "serving",
            "host": host,
            "port": port,
            "config_path": app.config_path,
        }

    exit_code = cli.main(
        ["--config", "config/test.yaml", "monitor"],
        app_factory=make_fake_head_runtime,
        monitor_server=fake_monitor_server,
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["command"] == "monitor"
    assert payload["status"] == "serving"
    assert payload["host"] == "0.0.0.0"
    assert payload["port"] == 18080
    assert payload["config_path"] == "config/test.yaml"
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 18080
    assert isinstance(called["app"], HeadRuntimeApp)


def test_monitor_custom_host_and_port_parse_for_service_entrypoint() -> None:
    called: dict[str, object] = {}

    def fake_monitor_server(*, app: HeadRuntimeApp, host: str, port: int) -> None:
        called["config_path"] = app.config_path
        called["host"] = host
        called["port"] = port
        return None

    payload = cli.dispatch(
        cli.build_parser().parse_args(
            ["--config", "config/honjia.yaml", "monitor", "--host", "0.0.0.0", "--port", "18080"]
        ),
        app_factory=make_fake_head_runtime,
        monitor_server=fake_monitor_server,
    )

    assert payload == {
        "command": "monitor",
        "runtime": "eihead",
        "status": "stopped",
        "host": "0.0.0.0",
        "port": 18080,
    }
    assert called == {
        "config_path": "config/honjia.yaml",
        "host": "0.0.0.0",
        "port": 18080,
    }


def test_http_uses_lazy_loader_when_no_runner_is_injected(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_http_server(*, app: HeadRuntimeApp, host: str, port: int) -> dict[str, object]:
        called["config_path"] = app.config_path
        return {
            "command": "http",
            "runtime": "eihead",
            "status": "serving",
            "host": host,
            "port": port,
        }

    def fake_loader() -> cli.HttpServerRunner:
        called["lazy_loader"] = True
        return fake_http_server

    monkeypatch.setattr(cli, "_load_http_server_runner", fake_loader)

    payload = cli.dispatch(
        cli.build_parser().parse_args(["http"]),
        app_factory=make_fake_head_runtime,
    )

    assert payload["command"] == "http"
    assert payload["status"] == "serving"
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 18081
    assert called == {
        "lazy_loader": True,
        "config_path": "config/eibrain.yaml",
    }


def test_monitor_uses_lazy_loader_when_no_runner_is_injected(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_monitor_server(*, app: HeadRuntimeApp, host: str, port: int) -> dict[str, object]:
        called["config_path"] = app.config_path
        return {
            "command": "monitor",
            "runtime": "eihead",
            "status": "serving",
            "host": host,
            "port": port,
        }

    def fake_loader() -> cli.MonitorServerRunner:
        called["lazy_loader"] = True
        return fake_monitor_server

    monkeypatch.setattr(cli, "_load_monitor_server_runner", fake_loader)

    payload = cli.dispatch(
        cli.build_parser().parse_args(["monitor"]),
        app_factory=make_fake_head_runtime,
    )

    assert payload["command"] == "monitor"
    assert payload["status"] == "serving"
    assert payload["host"] == "0.0.0.0"
    assert payload["port"] == 18080
    assert called == {
        "lazy_loader": True,
        "config_path": "config/eibrain.yaml",
    }


def test_legacy_status_still_uses_injected_runtime() -> None:
    stdout = io.StringIO()

    exit_code = cli.main(
        ["--config", "config/test.yaml", "status"],
        app_factory=make_fake_head_runtime,
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["command"] == "status"
    assert payload["runtime"] == "eihead"
    assert payload["body_runtime"]["node_id"] == "honjia-cli-http-test"
