#!/usr/bin/env python
"""Verify the release version is consistent everywhere it is declared.

Checked:
  * ``pyproject.toml`` ``[project] version``
  * ``src/llm_router_utils/sglang/_version.py`` ``__version__``
  * the README "Version Mapping" table has a row for it
  * ``.gitmodules`` pins the same sglang release the README maps that version to

A mismatch here is the kind of thing that only surfaces after an irreversible
PyPI publish, so it is a gate rather than a convention.

Usage::

    python scripts/check_version.py       # exit 1 on any inconsistency
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
VERSION_FILE = REPO_ROOT / "src" / "llm_router_utils" / "sglang" / "_version.py"
README = REPO_ROOT / "README.md"
GITMODULES = REPO_ROOT / ".gitmodules"


def main() -> int:
    problems: list[str] = []

    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    m = re.search(r'__version__\s*=\s*"([^"]+)"', VERSION_FILE.read_text(encoding="utf-8"))
    if not m:
        problems.append(f"{VERSION_FILE.name}: no __version__ found")
        runtime = None
    else:
        runtime = m.group(1)
        if runtime != declared:
            problems.append(
                f"pyproject version {declared!r} != _version.py {runtime!r}"
            )

    readme_text = README.read_text(encoding="utf-8")
    row = re.search(
        rf"^\|\s*`v{re.escape(declared)}`\s*\|\s*\[(v[\d.]+)\]", readme_text, re.MULTILINE
    )
    if not row:
        problems.append(f"README Version Mapping has no row for v{declared}")
        upstream = None
    else:
        upstream = row.group(1)

    if upstream:
        gm = re.search(r"branch\s*=\s*(\S+)", GITMODULES.read_text(encoding="utf-8"))
        if not gm:
            problems.append(".gitmodules: no submodule branch pinned")
        elif gm.group(1) != f"release/{upstream}":
            problems.append(
                f".gitmodules pins {gm.group(1)!r} but README maps v{declared} "
                f"to {upstream!r} (expected 'release/{upstream}')"
            )

    if problems:
        print("FAIL: version consistency")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"OK: v{declared} consistent (pyproject, _version.py, README, .gitmodules -> {upstream})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
