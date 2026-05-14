"""CLI for the eihead runtime scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Mapping, Sequence, TextIO

from .app import DEFAULT_CONFIG_PATH, HeadRuntimeApp
from .legacy_body import run_body_hardware_verifier

HeadRuntimeFactory = Callable[[str], HeadRuntimeApp]
ServerRunner = Callable[..., Mapping[str, Any] | None]
HttpServerRunner = ServerRunner
MonitorServerRunner = ServerRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the eihead compatibility runtime")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Start the head runtime compatibility wrapper")
    subparsers.add_parser("status", help="Print a head runtime status snapshot")
    subparsers.add_parser("verify", help="Verify the wrapper without touching real hardware")
    http_parser = subparsers.add_parser("http", help="Start the eihead HTTP API")
    http_parser.add_argument("--host", default="127.0.0.1")
    http_parser.add_argument("--port", default=18081, type=int)
    monitor_parser = subparsers.add_parser("monitor", help="Start the eihead native monitoring Web")
    monitor_parser.add_argument("--host", default="0.0.0.0")
    monitor_parser.add_argument("--port", default=18080, type=int)
    parser.set_defaults(command="status")
    return parser


def dispatch(
    args: argparse.Namespace,
    *,
    app_factory: HeadRuntimeFactory | None = None,
    http_server: HttpServerRunner | None = None,
    monitor_server: MonitorServerRunner | None = None,
) -> dict[str, Any]:
    factory = app_factory or HeadRuntimeApp.from_config_path
    app = factory(str(args.config))
    if args.command == "http":
        return _run_http_command(app, host=str(args.host), port=int(args.port), http_server=http_server)
    if args.command == "monitor":
        return _run_monitor_command(app, host=str(args.host), port=int(args.port), monitor_server=monitor_server)
    if args.command == "serve":
        return app.serve()
    if args.command == "verify":
        return app.verify()
    return app.status()


def main(
    argv: Sequence[str] | None = None,
    *,
    app_factory: HeadRuntimeFactory | None = None,
    http_server: HttpServerRunner | None = None,
    monitor_server: MonitorServerRunner | None = None,
    stdout: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = dispatch(
        args,
        app_factory=app_factory,
        http_server=http_server,
        monitor_server=monitor_server,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=stdout or sys.stdout)
    return 0


def verify_hardware_main() -> None:
    _run_body_hardware_verifier()


def _run_body_hardware_verifier() -> None:
    run_body_hardware_verifier()


def _run_http_command(
    app: HeadRuntimeApp,
    *,
    host: str,
    port: int,
    http_server: HttpServerRunner | None = None,
) -> dict[str, Any]:
    runner = http_server or _load_http_server_runner()
    result = runner(app=app, host=host, port=port)
    if result is None:
        return {
            "command": "http",
            "runtime": "eihead",
            "status": "stopped",
            "host": host,
            "port": port,
        }
    if not isinstance(result, Mapping):
        raise TypeError("eihead HTTP server runner must return a mapping or None")
    return dict(result)


def _load_http_server_runner() -> HttpServerRunner:
    """Load the HTTP API lazily so CLI import stays usable before A-line lands."""

    try:
        from . import http_api
    except ImportError as exc:
        raise RuntimeError(
            "eihead HTTP API is not available yet; expected eihead.runtime.http_api "
            "to expose run_http_api(app=..., host=..., port=...)."
        ) from exc

    for name in ("run_http_api", "serve_http_api", "serve", "run"):
        runner = getattr(http_api, name, None)
        if callable(runner):
            return runner
    raise RuntimeError(
        "eihead.runtime.http_api must expose one of: run_http_api, serve_http_api, serve, run."
    )


def _run_monitor_command(
    app: HeadRuntimeApp,
    *,
    host: str,
    port: int,
    monitor_server: MonitorServerRunner | None = None,
) -> dict[str, Any]:
    runner = monitor_server or _load_monitor_server_runner()
    result = runner(app=app, host=host, port=port)
    if result is None:
        return {
            "command": "monitor",
            "runtime": "eihead",
            "status": "stopped",
            "host": host,
            "port": port,
        }
    if not isinstance(result, Mapping):
        raise TypeError("eihead monitor server runner must return a mapping or None")
    return dict(result)


def _load_monitor_server_runner() -> MonitorServerRunner:
    try:
        from eihead.monitoring import web
    except ImportError as exc:
        raise RuntimeError(
            "eihead native monitor is not available yet; expected eihead.monitoring.web "
            "to expose serve(app=..., host=..., port=...)."
        ) from exc

    for name in ("serve_monitor", "serve", "run"):
        runner = getattr(web, name, None)
        if callable(runner):
            return runner
    raise RuntimeError("eihead.monitoring.web must expose one of: serve_monitor, serve, run.")
