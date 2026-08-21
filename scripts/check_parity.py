#!/usr/bin/env python
"""Parity checker: diff retained sglang files against the upstream submodule.

For each retained ``*.py`` file under
``src/llm_router_utils/sglang/srt/`` this script locates its counterpart under
``vendor/sglang/python/sglang/srt/`` (the git submodule pinned to a
``release/vX.Y.Z`` branch) and compares the two after normalizing away the
slimming transformations documented in ``HOW_TO_UPGRADE.md``:

1. Import rewriting — ``from sglang.`` / ``import sglang.`` is rewritten to the
   ``llm_router_utils.sglang.`` dialect on *both* sides so they speak the same
   import language.
2. xgrammar ``try/except ImportError`` guards — the retained side wraps bare
   ``from xgrammar import ...`` lines in a ``try:``/``except ImportError:`` block
   with ``Symbol = Any`` fallbacks. This script collapses both the retained
   guard block and the upstream bare-import block down to the bare-import form
   so the two are comparable. (The guard itself is verified separately by the
   ``check-torch``-style invariant checks; here we only want to *ignore* it as
   an expected difference.)
3. ``cuda_coredump`` import — the line
   ``import sglang.srt.debug_utils.cuda_coredump  # noqa`` is dropped from the
   retained ``environ.py``; it is stripped on both sides.
4. Trailing whitespace and runs of blank lines are collapsed.

After normalization, files that are identical pass. Files that still differ are
reported as **drift** with a unified diff, and each diff hunk is classified as
``expected`` (import / xgrammar / cuda_coredump — should already have been
normalized away, so seeing one here means the normalizer needs updating) or
``logic`` (real divergence to investigate).

Exit code is ``0`` when no logic drift is found, ``1`` otherwise. Files with no
upstream counterpart (repo-local or fully-slimmed) are reported separately and
do not affect the exit code.

Usage::

    python scripts/check_parity.py
    python scripts/check_parity.py --rel-path srt/function_call/inkling_detector.py
    python scripts/check_parity.py --diff srt/parser/reasoning_parser.py
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RETAINED_ROOT = REPO_ROOT / "src" / "llm_router_utils" / "sglang" / "srt"
UPSTREAM_ROOT = REPO_ROOT / "vendor" / "sglang" / "python" / "sglang" / "srt"

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Rule 1: import rewriting. Rewrite upstream's ``sglang.`` imports into the
# retained dialect so both sides use the same module path. Applied to *both*
# sides (it is a no-op on the retained side, which is already rewritten).
_IMPORT_FROM_RE = re.compile(r"^(\s*from\s+)sglang\.", re.MULTILINE)
_IMPORT_BARE_RE = re.compile(r"^(\s*import\s+)sglang\.", re.MULTILINE)


def _rewrite_imports(text: str) -> str:
    text = _IMPORT_FROM_RE.sub(r"\1llm_router_utils.sglang.", text)
    text = _IMPORT_BARE_RE.sub(r"\1llm_router_utils.sglang.", text)
    return text


# Rule 2: xgrammar try/except guard collapse.
#
# The retained side looks like::
#
#     try:
#         from xgrammar import StructuralTag
#         from xgrammar.structural_tag import (
#             AnyTextFormat,
#             ...
#         )
#     except ImportError:
#         StructuralTag = Any
#         AnyTextFormat = Any
#         ...
#
# The upstream side is just the bare ``from xgrammar import ...`` lines with no
# surrounding ``try:``/``except ImportError:`` block.
#
# We normalize *both* sides to the bare-import form by:
#   - stripping the ``try:`` line that immediately precedes an xgrammar import,
#   - stripping the ``except ImportError:`` line and every ``X = Any`` fallback
#     line that follows it (up to the next blank line / non-assignment line).
#
# This makes the two sides byte-comparable on the import block while preserving
# every other line.

_TRY_LINE_RE = re.compile(r"^(\s*)try:\s*$")
_EXCEPT_IMPORT_RE = re.compile(r"^(\s*)except\s+ImportError\s*:\s*$")
_EXCEPT_BARE_RE = re.compile(r"^(\s*)except\s*:\s*$")
_XGRAMMAR_IMPORT_RE = re.compile(r"^\s*from\s+xgrammar(\.|\s)", re.MULTILINE)
_FALLBACK_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Any\s*$")


def _indent_of(line: str) -> str:
    return re.match(r"^(\s*)", line).group(1)


def _dedent(line: str, n: int) -> str:
    """Remove up to ``n`` leading spaces from ``line`` (clamped at 0)."""
    leading = len(line) - len(line.lstrip(" "))
    remove = min(n, leading)
    return line[remove:]


def _collapse_xgrammar_guard(lines: list[str]) -> list[str]:
    """Strip ``try:``/``except (ImportError)?:`` scaffolding around xgrammar imports.

    The retained side wraps bare ``from xgrammar ...`` imports in::

        try:                       # <- indent T
            from xgrammar ...      # <- indent T+4
            ...
        except ImportError:        # <- indent T
            StructuralTag = Any
            ...

    The upstream side has only the bare (un-indented) imports. To make the two
    comparable we normalize *both* sides to the bare-import form by, whenever we
    see a ``try:`` at indent ``T`` whose first non-blank body line is an xgrammar
    import:

      - dropping the ``try:`` line,
      - un-indenting every body line by 4 spaces (the try-block indent),
      - dropping the matching ``except (ImportError|:):`` line at indent ``T``
        and every ``Name = Any`` fallback line that follows it (until a blank
        line that is not followed by another fallback, or a non-fallback line).

    A ``try:`` whose body does *not* start with an xgrammar import is left
    untouched (it is real logic).
    """
    out: list[str] = []
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]
        stripped = line.rstrip("\n")

        m_try = _TRY_LINE_RE.match(stripped)
        if m_try:
            try_indent = m_try.group(1)
            # Look ahead (skipping blanks) for the first body line.
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and _XGRAMMAR_IMPORT_RE.match(lines[j]):
                # This is an xgrammar guard. Drop ``try:`` and walk the body,
                # un-indenting each line by 4 spaces, until the matching
                # ``except`` at ``try_indent``.
                i += 1
                while i < n:
                    body = lines[i].rstrip("\n")
                    # Matching except?
                    if _EXCEPT_IMPORT_RE.match(body) or _EXCEPT_BARE_RE.match(body):
                        if _indent_of(body) == try_indent:
                            # Drop the except and its ``Name = Any`` fallbacks.
                            i += 1
                            while i < n:
                                fb = lines[i].rstrip("\n")
                                if fb.strip() == "":
                                    # Drop a blank only if more fallbacks follow;
                                    # otherwise stop (keep real separators).
                                    k = i + 1
                                    while k < n and lines[k].strip() == "":
                                        k += 1
                                    if k < n and _FALLBACK_ASSIGN_RE.match(lines[k]):
                                        i += 1
                                        continue
                                    break
                                if _FALLBACK_ASSIGN_RE.match(fb):
                                    i += 1
                                    continue
                                break
                            break
                        # An except at a different indent inside the body — keep
                        # it (un-indented). Fall through to the body handler.
                    # Regular body line: un-indent by 4 and emit.
                    out.append(_dedent(body, 4) + "\n")
                    i += 1
                continue
            # Not an xgrammar guard — emit ``try:`` and continue normally.
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1

    return out


# Rule 3: drop the cuda_coredump import line.
_CUDACOREDUMP_RE = re.compile(
    r"^\s*import\s+sglang\.srt\.debug_utils\.cuda_coredump\s*(#.*)?$"
)


def _drop_cuda_coredump(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _CUDACOREDUMP_RE.match(line)
    )


# Rule 3b: when a file has an xgrammar import, the retained side adds ``Any`` to
# its ``from typing import ...`` line (needed for the ``Name = Any`` fallbacks).
# Since the collapser strips the fallback lines, the only remaining trace is the
# extra ``Any`` in the typing import. Strip ``Any`` from ``from typing import``
# on both sides so the two are comparable. (Only applied when the file contains
# an xgrammar import, so unrelated ``Any`` usages are untouched.)
_TYPING_ANY_RE = re.compile(r"^(\s*from\s+typing\s+import\s+)(.*)$")


def _strip_any_from_typing(text: str) -> str:
    if not _XGRAMMAR_IMPORT_RE.search(text):
        return text
    out: list[str] = []
    for line in text.splitlines():
        m = _TYPING_ANY_RE.match(line)
        if m:
            names = [n.strip() for n in m.group(2).split(",") if n.strip()]
            names = [n for n in names if n != "Any"]
            if names:
                out.append(f"{m.group(1)}{', '.join(names)}")
            # If ``Any`` was the only name, drop the line entirely.
            continue
        out.append(line)
    return "\n".join(out)


# Rule 4: whitespace cleanup.
def _normalize_whitespace(text: str) -> str:
    # Strip trailing whitespace per line.
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse runs of blank lines into a single blank line, and strip
    # leading/trailing blank lines.
    out: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        out.append(line)
        prev_blank = is_blank
    while out and out[-1] == "":
        out.pop()
    while out and out[0] == "":
        out.pop(0)
    return "\n".join(out) + "\n"


def normalize(text: str) -> str:
    """Apply all normalization rules to a file's text."""
    text = _rewrite_imports(text)
    text = _drop_cuda_coredump(text)
    text = _strip_any_from_typing(text)
    lines = text.splitlines(keepends=True)
    lines = _collapse_xgrammar_guard(lines)
    text = "".join(lines)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Diff classification
# ---------------------------------------------------------------------------

# Lines that are expected to differ because of slimming. After normalization
# these *should* be gone, so matching one in a diff means the normalizer missed
# a pattern (still not a logic drift, but worth flagging so the script stays
# accurate).
_EXPECTED_PATTERNS = [
    re.compile(r"^\s*from\s+xgrammar"),
    re.compile(r"^\s*import\s+xgrammar"),
    re.compile(r"^\s*try:\s*$"),
    re.compile(r"^\s*except\s"),
    re.compile(r"=\s*Any\s*$"),
    re.compile(r"cuda_coredump"),
    re.compile(r"llm_router_utils\.sglang\."),
    re.compile(r"^\s*from\s+sglang\."),
    re.compile(r"^\s*import\s+sglang\."),
]


def _classify_hunk(diff_lines: list[str]) -> str:
    """Classify a unified-diff hunk as 'expected', 'slimming', or 'logic'.

    - 'expected': every changed line matches a known slimming pattern
      (import rewrite, xgrammar guard, cuda_coredump). After normalization
      these should be rare; seeing one means the normalizer missed a pattern.
    - 'slimming': the hunk only *removes* lines (pure deletion) with no
      substantive additions. Deletions are sanctioned slimming per CLAUDE.md
      ("Only delete or slim"). Blank-only additions are allowed.
    - 'logic': anything else — real divergence to investigate.
    """
    added = [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in diff_lines if l.startswith("-") and not l.startswith("---")]
    changed = added + removed
    if not changed:
        return "expected"
    if all(any(p.search(l) for p in _EXPECTED_PATTERNS) for l in changed):
        return "expected"
    # Pure deletion (no substantive added lines) = sanctioned slimming.
    substantive_added = [l for l in added if l.strip() != ""]
    if not substantive_added:
        return "slimming"
    return "logic"


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------


@dataclass
class FileResult:
    rel_path: str
    status: str  # "match" | "drift" | "no-upstream" | "upstream-only"
    logic_drift: bool = False
    diff: str = ""
    hunk_classes: list[str] = field(default_factory=list)


def _iter_retained_files() -> list[Path]:
    return sorted(
        p for p in RETAINED_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def _iter_upstream_files() -> list[Path]:
    if not UPSTREAM_ROOT.exists():
        return []
    return sorted(
        p for p in UPSTREAM_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def compare_file(retained: Path) -> FileResult:
    rel = _rel(retained, RETAINED_ROOT)
    upstream = UPSTREAM_ROOT / rel
    if not upstream.exists():
        return FileResult(rel_path=rel, status="no-upstream")
    try:
        retained_text = retained.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        retained_text = retained.read_text(encoding="utf-8", errors="replace")
    try:
        upstream_text = upstream.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        upstream_text = upstream.read_text(encoding="utf-8", errors="replace")

    retained_norm = normalize(retained_text)
    upstream_norm = normalize(upstream_text)

    if retained_norm == upstream_norm:
        return FileResult(rel_path=rel, status="match")

    diff_lines = list(
        difflib.unified_diff(
            upstream_norm.splitlines(keepends=True),
            retained_norm.splitlines(keepends=True),
            fromfile=f"upstream/{rel}",
            tofile=f"retained/{rel}",
            n=2,
        )
    )
    diff_text = "".join(diff_lines)

    # Classify each hunk. A hunk starts with a ``@@`` line.
    hunk_classes: list[str] = []
    current: list[str] = []
    for line in diff_lines:
        if line.startswith("@@"):
            if current:
                hunk_classes.append(_classify_hunk(current))
            current = []
        current.append(line)
    if current:
        hunk_classes.append(_classify_hunk(current))

    logic_drift = any(c == "logic" for c in hunk_classes)
    return FileResult(
        rel_path=rel,
        status="drift",
        logic_drift=logic_drift,
        diff=diff_text,
        hunk_classes=hunk_classes,
    )


def print_report(results: list[FileResult], show_diffs: bool) -> tuple[int, int]:
    matches = [r for r in results if r.status == "match"]
    drift = [r for r in results if r.status == "drift"]
    no_upstream = [r for r in results if r.status == "no-upstream"]

    logic_drift_files = [r for r in drift if r.logic_drift]
    non_logic_drift_files = [r for r in drift if not r.logic_drift]

    print(f"Parity check: {RETAINED_ROOT.relative_to(REPO_ROOT)}")
    print(f"  vs upstream: {UPSTREAM_ROOT.relative_to(REPO_ROOT)}")
    print()
    print(f"  Retained files checked : {len(results)}")
    print(f"  Match (after normalize): {len(matches)}")
    print(f"  Drift (slimming/expected): {len(non_logic_drift_files)}")
    print(f"  Drift (logic)          : {len(logic_drift_files)}")
    print(f"  No upstream counterpart: {len(no_upstream)}")
    print()

    if non_logic_drift_files:
        print("Drift — sanctioned slimming (deletions / expected patterns only):")
        for r in non_logic_drift_files:
            print(f"  {r.rel_path}  hunks={r.hunk_classes}")
        print()

    if logic_drift_files:
        print("Drift — LOGIC divergence (investigate):")
        for r in logic_drift_files:
            print(f"  {r.rel_path}  hunks={r.hunk_classes}")
        if show_diffs:
            print()
            print("=" * 78)
            for r in logic_drift_files:
                print(f"--- {r.rel_path} ---")
                print(r.diff)
            print("=" * 78)
        print()

    if no_upstream:
        print("Retained files with no upstream counterpart (repo-local / fully slimmed):")
        for r in no_upstream:
            print(f"  {r.rel_path}")
        print()

    return len(logic_drift_files), len(non_logic_drift_files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rel-path",
        action="append",
        default=[],
        help="Limit to a retained file's path relative to srt/ (repeatable).",
    )
    parser.add_argument(
        "--diff",
        action="append",
        default=[],
        help="Show full diff for the given retained file path (repeatable).",
    )
    parser.add_argument(
        "--show-diffs",
        action="store_true",
        help="Show full unified diffs for all logic-drift files.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "scripts" / "parity_baseline.txt",
        help=(
            "Path to a baseline file listing known logic-drift paths (one per "
            "line, '#' comments allowed). When set, the script exits 0 if every "
            "logic-drift file is already in the baseline, and exits 1 only on "
            "NEW logic drift. Use --update-baseline to (re)write the file. "
            "Pass --no-baseline to disable."
        ),
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current set of logic-drift paths to --baseline and exit 0.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Ignore --baseline entirely; fail on any logic drift.",
    )
    args = parser.parse_args(argv)

    if not UPSTREAM_ROOT.exists():
        print(
            f"ERROR: upstream root not found at {UPSTREAM_ROOT}\n"
            "Run: git submodule update --init vendor/sglang",
            file=sys.stderr,
        )
        return 2

    retained_files = _iter_retained_files()

    if args.rel_path or args.diff:
        wanted = set(args.rel_path) | set(args.diff)
        retained_files = [
            p for p in retained_files if _rel(p, RETAINED_ROOT) in wanted
        ]
        if not retained_files:
            print(f"No retained files matched: {wanted}", file=sys.stderr)
            return 2

    results = [compare_file(p) for p in retained_files]

    # If --diff was used, force-show diffs for those specific files regardless
    # of classification.
    diff_requested = set(args.diff)
    if diff_requested:
        for r in results:
            if r.rel_path in diff_requested and r.status == "drift":
                print("=" * 78)
                print(f"--- {r.rel_path} ---  hunks={r.hunk_classes}")
                print(r.diff)
                print("=" * 78)
                print()

    logic_count, expected_count = print_report(results, show_diffs=args.show_diffs)

    logic_drift_paths = sorted(
        r.rel_path for r in results if r.status == "drift" and r.logic_drift
    )

    # --update-baseline: (re)write the baseline and succeed.
    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            "# Parity baseline: retained files with known logic drift vs upstream.\n"
            "# These are deliberately slimmed files (Protocol annotations, deleted\n"
            "# methods, hf_transformers_utils -> transformers substitutions, etc.)\n"
            "# that require manual review on each upgrade. Regenerate with:\n"
            "#   python scripts/check_parity.py --update-baseline\n"
            + ("\n".join(logic_drift_paths) + "\n" if logic_drift_paths else ""),
            encoding="utf-8",
        )
        print(f"Wrote baseline: {args.baseline} ({len(logic_drift_paths)} file(s))")
        return 0

    use_baseline = (not args.no_baseline) and args.baseline is not None
    baseline_paths: set[str] = set()
    if use_baseline and args.baseline.exists():
        for line in args.baseline.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                baseline_paths.add(line)

    new_drift = [p for p in logic_drift_paths if p not in baseline_paths]

    if use_baseline:
        if new_drift:
            print("NEW logic drift not in baseline (regression!):")
            for p in new_drift:
                print(f"  {p}")
            print()
            print(
                f"FAIL: {len(new_drift)} new logic-drift file(s). "
                f"({len(logic_drift_paths)} total, {len(baseline_paths)} baselined)."
            )
            print(
                "If these are intentional, update the baseline:\n"
                "  python scripts/check_parity.py --update-baseline"
            )
            return 1
        print(
            f"OK: no new logic drift. ({len(logic_drift_paths)} baselined slimmed "
            f"file(s) require manual review on upgrade.)"
        )
        return 0

    if logic_count:
        print(f"FAIL: {logic_count} file(s) with logic drift.")
        return 1
    print("OK: no logic drift detected (expected-only drift is informational).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
