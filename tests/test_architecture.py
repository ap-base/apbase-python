from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "apbase"


def _python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_only_lib_modules_reference_fortran_bindings() -> None:
    violations: list[str] = []

    for path in _python_files():
        if path.name in {"_lib.py", "_native.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "_fortran" in text or ".so" in text or ".pyd" in text or "f2py" in text:
            violations.append(str(path.relative_to(PACKAGE_ROOT)))

    assert violations == []


def test_only_lib_modules_call_compiled_lib_symbols() -> None:
    violations: list[str] = []

    for path in _python_files():
        if path.name in {"_lib.py", "_native.py"}:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "lib"
            ):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")

    assert violations == []
