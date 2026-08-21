#!/usr/bin/env python
"""License-compliance checker for the llm-router-utils derivative work.

This repo is a derivative work of sglang (Apache 2.0). This script verifies
the Apache 2.0 §4 redistribution obligations are met for the retained source
files under ``src/llm_router_utils/sglang/srt/``:

1. **§4(a) — License copy**: ``LICENSE`` exists at repo root and contains the
   Apache 2.0 header.
2. **§4(c) — NOTICE / attribution**: ``NOTICE`` exists at repo root and names
   sglang as the upstream.
3. **§4(c) — Header retention**: every retained file whose upstream counterpart
   (under ``vendor/sglang/python/sglang/srt/``) carries a SGLang copyright
   header must itself carry a SGLang copyright header. Files whose upstream
   counterpart has no header are not required to add one (Apache 2.0's appendix
   boilerplate is optional, not a §4 obligation).
4. **§4(b) — Modification notice**: every retained file that is *modified*
   (i.e. classified as logic-drift in ``scripts/parity_baseline.txt`` — the
   deliberately slimmed files) and carries a header must also carry a
   "Derivative work" notice stating the file was changed.

Repo-local files with no upstream counterpart (``__init__.py`` stubs, etc.)
are not derivative works of a specific upstream file, so rules 3 and 4 do not
apply to them.

Exit code is ``0`` when all checks pass, ``1`` on any violation.

Usage::

    python scripts/check_license.py
    python scripts/check_license.py -v          # verbose: list every file checked
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RETAINED_ROOT = REPO_ROOT / "src" / "llm_router_utils" / "sglang" / "srt"
UPSTREAM_ROOT = REPO_ROOT / "vendor" / "sglang" / "python" / "sglang" / "srt"
BASELINE_FILE = REPO_ROOT / "scripts" / "parity_baseline.txt"

# A retained file is considered to "carry a SGLang copyright header" if its
# first ~25 lines contain both a Copyright line mentioning SGLang and the
# "Licensed under the Apache License" line. This mirrors upstream sglang's
# header convention.
_HEADER_SCAN_LINES = 25
_COPYRIGHT_RE = re.compile(r"Copyright.*SGLang", re.IGNORECASE)
_LICENSE_LINE_RE = re.compile(r"Licensed under the Apache License", re.IGNORECASE)
_DERIVATIVE_RE = re.compile(r"Derivative work", re.IGNORECASE)
_APACHE_HEADER_RE = re.compile(r"Apache License.*Version 2\.0", re.IGNORECASE)
_SGLANG_MENTION_RE = re.compile(r"sglang", re.IGNORECASE)


@dataclass
class Violation:
    rule: str
    path: str
    detail: str


def _has_header(text: str) -> bool:
    head = "\n".join(text.splitlines()[:_HEADER_SCAN_LINES])
    return bool(_COPYRIGHT_RE.search(head) and _LICENSE_LINE_RE.search(head))


def _has_derivative_notice(text: str) -> bool:
    head = "\n".join(text.splitlines()[:_HEADER_SCAN_LINES])
    return bool(_DERIVATIVE_RE.search(head))


def _read_baselines() -> set[str]:
    """Return the set of slimmed (logic-drift) file paths from the parity baseline."""
    if not BASELINE_FILE.exists():
        return set()
    out: set[str] = set()
    for line in BASELINE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def _iter_retained_files() -> list[Path]:
    return sorted(
        p for p in RETAINED_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def _rel_srt(path: Path) -> str:
    """Path relative to ``srt/`` (matches parity_baseline.txt convention)."""
    return path.relative_to(RETAINED_ROOT).as_posix()


def check_license_and_notice(violations: list[Violation]) -> None:
    # §4(a): LICENSE present + Apache 2.0.
    license_file = REPO_ROOT / "LICENSE"
    if not license_file.exists():
        violations.append(
            Violation("§4(a)", "LICENSE", "missing — must include Apache 2.0 License")
        )
    else:
        text = license_file.read_text(encoding="utf-8", errors="replace")
        if not _APACHE_HEADER_RE.search(text):
            violations.append(
                Violation("§4(a)", "LICENSE", "does not contain Apache 2.0 header")
            )

    # §4(c): NOTICE present + mentions sglang as upstream.
    notice_file = REPO_ROOT / "NOTICE"
    if not notice_file.exists():
        violations.append(
            Violation(
                "§4(c)",
                "NOTICE",
                "missing — derivative work must retain upstream attribution",
            )
        )
    else:
        text = notice_file.read_text(encoding="utf-8", errors="replace")
        if not _SGLANG_MENTION_RE.search(text):
            violations.append(
                Violation("§4(c)", "NOTICE", "does not mention sglang (upstream)")
            )


def check_files(violations: list[Violation]) -> tuple[int, int, int]:
    baselined = _read_baselines()
    retained_files = _iter_retained_files()

    checked = 0
    headered = 0
    for retained in retained_files:
        rel = _rel_srt(retained)
        upstream = UPSTREAM_ROOT / rel
        if not upstream.exists():
            continue  # repo-local file; rules 3/4 do not apply
        checked += 1

        try:
            retained_text = retained.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            retained_text = retained.read_text(encoding="utf-8", errors="replace")
        try:
            upstream_text = upstream.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            upstream_text = upstream.read_text(encoding="utf-8", errors="replace")

        up_has = _has_header(upstream_text)
        ret_has = _has_header(retained_text)

        if ret_has:
            headered += 1

        # Rule 3 (§4(c)): if upstream has a header, retained must keep one.
        if up_has and not ret_has:
            violations.append(
                Violation(
                    "§4(c)",
                    f"src/llm_router_utils/sglang/srt/{rel}",
                    "upstream carries a SGLang copyright header but the retained "
                    "file stripped it — must retain upstream attribution",
                )
            )

        # Rule 4 (§4(b)): if the file is slimmed (baselined as logic-drift) and
        # carries a header, it must also carry a "Derivative work" notice.
        is_slimmed = rel in baselined
        if is_slimmed and ret_has and not _has_derivative_notice(retained_text):
            violations.append(
                Violation(
                    "§4(b)",
                    f"src/llm_router_utils/sglang/srt/{rel}",
                    "slimmed (modified) file carries a header but no 'Derivative "
                    "work' notice — modified files must state they were changed",
                )
            )

    return checked, headered, len(retained_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List every retained file checked and its header status.",
    )
    args = parser.parse_args(argv)

    if not UPSTREAM_ROOT.exists():
        print(
            f"ERROR: upstream root not found at {UPSTREAM_ROOT}\n"
            "Run: git submodule update --init vendor/sglang",
            file=sys.stderr,
        )
        return 2

    violations: list[Violation] = []
    check_license_and_notice(violations)
    checked, headered, total = check_files(violations)

    print("License compliance check:")
    print(f"  Retained files with upstream counterpart: {checked}")
    print(f"  Retained files carrying a SGLang header  : {headered}")
    print(f"  Total retained .py files                 : {total}")
    print()

    if args.verbose:
        baselined = _read_baselines()
        print("Per-file status:")
        for retained in _iter_retained_files():
            rel = _rel_srt(retained)
            upstream = UPSTREAM_ROOT / rel
            if not upstream.exists():
                continue
            ret_text = retained.read_text(encoding="utf-8", errors="replace")
            up_text = upstream.read_text(encoding="utf-8", errors="replace")
            up_h = "H" if _has_header(up_text) else "-"
            ret_h = "H" if _has_header(ret_text) else "-"
            slim = "S" if rel in baselined else "-"
            deriv = "D" if _has_derivative_notice(ret_text) else "-"
            print(f"  up={up_h} ret={ret_h} slim={slim} deriv={deriv}  {rel}")
        print()

    if violations:
        print(f"FAIL: {len(violations)} license violation(s):")
        for v in violations:
            print(f"  [{v.rule}] {v.path}")
            print(f"      {v.detail}")
        return 1

    print("OK: Apache 2.0 §4(a)/(b)/(c) obligations met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
