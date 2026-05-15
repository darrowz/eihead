import ast
from pathlib import Path


FORBIDDEN_ROOTS = {"eibrain"}
SCAN_ROOTS = ("eihead", "apps/head_runtime")


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_eihead_runtime_has_no_eibrain_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for scan_root in SCAN_ROOTS:
        root = repo_root / scan_root
        for path in sorted(root.rglob("*.py")):
            forbidden = _import_roots(path) & FORBIDDEN_ROOTS
            if forbidden:
                offenders.append(f"{path.relative_to(repo_root)}: {sorted(forbidden)}")

    assert offenders == []
