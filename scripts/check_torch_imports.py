#!/usr/bin/env python
"""Verify the ``no direct torch import`` invariant from CLAUDE.md/DEVELOPMENT.md.

``import torch`` / ``from torch import ...`` anywhere in ``src/`` as a
*module-level* statement is a violation: torch may only enter this package
transitively via xgrammar's tvm_ffi.

An import that is deferred inside a function body is allowed — the retained code
uses that form for optional ROCm/NPU probing, and it keeps import time
torch-free. A module-level ``import torch`` nested in a ``try:`` block is still
a violation: it runs unconditionally on import, and it is exactly the shape the
old ``grep '^import torch'`` check could not see (the ``^`` anchor pinned
matches to column 0).

Usage::

    python scripts/check_torch_imports.py      # exit 1 on violation
"""

from __future__ import annotations

import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def _is_torch(name: str) -> bool:
    return name == "torch" or name.startswith("torch.")


def _module_level_imports(tree: ast.Module):
    """Yield (lineno, name) for torch imports reached without entering a function."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Deferred import: allowed.
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_torch(alias.name):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and _is_torch(node.module):
                yield node.lineno, node.module
        for child in ast.iter_child_nodes(node):
            stack.append(child)


def main() -> int:
    violations: list[tuple[pathlib.Path, int, str]] = []
    files = sorted(SRC_ROOT.rglob("*.py"))
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - syntax is covered by tests
            print(f"FAIL: cannot parse {path}: {exc}", file=sys.stderr)
            return 2
        for lineno, name in _module_level_imports(tree):
            violations.append((path, lineno, name))

    if not violations:
        print(f"OK: no module-level torch import in src/ ({len(files)} files checked)")
        return 0

    print("FAIL: module-level torch import found in src/")
    for path, lineno, name in violations:
        print(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {name}")
    print(
        "\nMove the import inside the function that needs it, or drop it "
        "(DEVELOPMENT.md rule 2)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
