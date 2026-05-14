from pathlib import Path


def _read_python_tree(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    )


def test_mouth_does_not_import_voice_runtime() -> None:
    text = _read_python_tree(Path("eihead") / "mouth")
    assert "from eivoice_runtime" not in text
    assert "import eivoice_runtime" not in text


def test_voice_runtime_does_not_import_mouth_playback_directly() -> None:
    text = _read_python_tree(Path("eihead") / "eivoice_runtime")
    assert "from eihead.mouth" not in text
    assert "import eihead.mouth" not in text
