from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from eihead.eivoice_runtime.cloud_providers import CloudProviderConfig


SCHEMA = "eibrain.voice_provider_smoke.v1"
PROVIDER_CHOICES = ("minimax-tts", "dashscope-asr", "all")
ALL_PROVIDERS = ("minimax-tts", "dashscope-asr")


def build_readiness(provider: str, *, dry_run: bool = True, live: bool = False) -> dict[str, Any]:
    provider_name, env_provider, required_fields, defaulted_fields = _provider_spec(provider)
    config = CloudProviderConfig.from_env(env_provider)
    missing_fields = _missing_fields(config, required_fields)
    defaults_used = _missing_fields(config, defaulted_fields)
    diagnostics = dict(config.diagnostics())
    diagnostics.update(
        {
            "readiness_mode": "dry_run" if dry_run else "live_requested",
            "network_called": False,
            "defaults_used": defaults_used,
        }
    )
    if live:
        diagnostics["live_note"] = "live provider calls are intentionally not implemented in this smoke tool yet"

    return {
        "provider": provider_name,
        "configured": not missing_fields,
        "missing_fields": missing_fields,
        "base_url_present": bool(config.base_url),
        "model_present": bool(config.model),
        "voice_id_present": bool(config.voice_id),
        "defaults_used": defaults_used,
        "live_supported": False,
        "dry_run": bool(dry_run),
        "diagnostics": diagnostics,
    }


def build_report(provider: str, *, dry_run: bool = True, live: bool = False) -> dict[str, Any]:
    providers = ALL_PROVIDERS if provider == "all" else (provider,)
    readiness = [build_readiness(item, dry_run=dry_run, live=live) for item in providers]
    configured_providers = [str(item["provider"]) for item in readiness if item["configured"] is True]
    missing_providers = [str(item["provider"]) for item in readiness if item["configured"] is not True]
    return {
        "schema": SCHEMA,
        "dry_run": bool(dry_run),
        "live": bool(live),
        "providers": readiness,
        "configured": all(item["configured"] for item in readiness),
        "readiness": {
            "status": "healthy" if len(configured_providers) == len(readiness) else "degraded",
            "configured_provider_count": len(configured_providers),
            "configured_providers": configured_providers,
            "missing_providers": missing_providers,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dry_run = bool(args.dry_run or not args.live)
    report = build_report(args.provider, dry_run=dry_run, live=bool(args.live))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run readiness smoke check for cloud voice providers.")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default="all")
    parser.add_argument("--dry-run", action="store_true", help="Check env/config readiness without external network calls.")
    parser.add_argument("--live", action="store_true", help="Reserved for future live provider checks; no network calls are made today.")
    return parser.parse_args(argv)


def _provider_spec(provider: str) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    normalized = str(provider).strip().lower()
    if normalized == "minimax-tts":
        return "minimax-tts", "minimax", ("api_key",), ("model", "voice_id")
    if normalized == "dashscope-asr":
        return "dashscope-asr", "dashscope", ("api_key",), ("model",)
    raise ValueError(f"unsupported provider: {provider}")


def _missing_fields(config: CloudProviderConfig, required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if not getattr(config, field)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
