"""CLI entrypoint for body runtime."""

from __future__ import annotations

import argparse
import json

from .app import BodyRuntimeApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Start eibrain body runtime")
    parser.add_argument("--config", default="config/eibrain.yaml")
    args = parser.parse_args()

    runtime = BodyRuntimeApp.from_config_path(args.config)
    print(json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
