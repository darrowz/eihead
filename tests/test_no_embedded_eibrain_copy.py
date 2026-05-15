from pathlib import Path


def test_eihead_has_no_embedded_eibrain_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "eibrain").exists()


def test_eihead_has_no_legacy_body_runtime_app_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "apps" / "body_runtime").exists()
