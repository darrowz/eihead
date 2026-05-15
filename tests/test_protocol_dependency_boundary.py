import importlib.util
from pathlib import Path


def test_eihead_has_no_embedded_eiprotocol_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "eiprotocol").exists()


def test_eihead_imports_protocol_from_standalone_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("eiprotocol")

    assert spec is not None
    assert spec.origin is not None
    assert repo_root not in Path(spec.origin).resolve().parents
