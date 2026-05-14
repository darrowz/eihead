import json
from pathlib import Path


def test_export_manifest_marks_eihead_as_transitional_split() -> None:
    manifest = json.loads(Path("EXPORT_MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["standalone_repo"]["name"] == "eihead"
    assert manifest["standalone_repo"]["runtime_path"] == "/opt/eihead/current"
    assert manifest["source"]["repository"] == "eibrain"
    assert manifest["cutover_readiness"]["legacy_shim_policy"]["state"] == "transitional"
