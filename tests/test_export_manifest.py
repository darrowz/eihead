import json
from pathlib import Path


def test_export_manifest_marks_eihead_as_native_boundary() -> None:
    manifest = json.loads(Path("EXPORT_MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["standalone_repo"]["name"] == "eihead"
    assert manifest["standalone_repo"]["runtime_path"] == "/opt/eihead/current"
    assert manifest["code_completion"]["full_detachment_claim_allowed"] is False
    assert manifest["cutover_readiness"]["hardware_verified"] is False
    assert manifest["cutover_readiness"]["legacy_body_runtime_detached"] is True
