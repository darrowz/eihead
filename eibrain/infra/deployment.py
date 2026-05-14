"""Deployment bootstrap helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eibrain.infra.config import EIBrainConfig


@dataclass(slots=True)
class DeploymentLayout:
    root_dir: Path
    body_runtime_dir: Path
    cognitive_runtime_dir: Path
    sherpa_model_dir: Path


def _resolve_sherpa_model_dir(config: EIBrainConfig, root_dir: Path) -> Path:
    asr_config = config.body.organs.get("ear")
    if asr_config is None:
        return root_dir / "models" / "asr" / "sherpa-onnx-streaming"
    asr_driver = asr_config.subfunctions.get("asr")
    if asr_driver is None:
        return root_dir / "models" / "asr" / "sherpa-onnx-streaming"
    model_dir = str(asr_driver.driver.extra.get("model_dir", "")).strip()
    if model_dir:
        return Path(model_dir)
    return root_dir / "models" / "asr" / "sherpa-onnx-streaming"


def bootstrap_default_deployment(config: EIBrainConfig) -> DeploymentLayout:
    root_dir = Path(config.deployment.root_dir)
    body_runtime_dir = Path(config.deployment.body_runtime_dir or config.deployment.root_dir)
    cognitive_runtime_dir = Path(config.deployment.cognitive_runtime_dir or config.deployment.root_dir)
    sherpa_model_dir = _resolve_sherpa_model_dir(config, root_dir)

    for directory in (root_dir, body_runtime_dir, cognitive_runtime_dir, sherpa_model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (sherpa_model_dir / "README.md").write_text(
        "\n".join(
            [
                "# sherpa-onnx streaming model directory",
                "",
                "Populate this directory with the local streaming model assets expected by eibrain:",
                "",
                "- tokens.txt",
                "- encoder.onnx",
                "- decoder.onnx",
                "- joiner.onnx",
                "",
                "Replace the empty marker files once the real model bundle is ready.",
            ]
        ),
        encoding="utf-8",
    )
    for filename in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
        target = sherpa_model_dir / filename
        if not target.exists():
            target.write_text("", encoding="utf-8")

    return DeploymentLayout(
        root_dir=root_dir,
        body_runtime_dir=body_runtime_dir,
        cognitive_runtime_dir=cognitive_runtime_dir,
        sherpa_model_dir=sherpa_model_dir,
    )
