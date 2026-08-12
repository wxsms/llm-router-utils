"""Generate hash golden values from a real sglang installation.

Run this script in an environment where sglang + torch + OpenSSL are
installed (i.e. Linux with sglang pip-installed).  It computes hashes
via sglang's native C++ extension and writes them to a JSON file.
The matching pytest (test/unit/mem_cache/test_hash_golden.py) then loads
this JSON and asserts the pure-Python implementation in
llm_router_utils.sglang.srt.mem_cache.utils produces identical bytes.

Usage on a real sglang host:

    python test/fixtures/gen_hash_golden.py --out test/fixtures/hash_golden.json

Then commit test/fixtures/hash_golden.json.  The pytest will pick it up
automatically.

The test cases mirror the actual call shape kv_tree.hash_tree uses:
    get_hash_str(token_ids, parent_hash_or_None)   # page_size never passed
    hash_str_to_int64(hash_str)                    # router stores int64

We also exercise paged mode (page_size > 1) for completeness, since
compute_node_hash_values in mem_cache_utils supports it and a future
caller might use it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/python ints / lists to JSON-serializable forms."""
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
    if np is not None and isinstance(obj, np.integer):
        return int(obj)
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _build_cases() -> list[dict]:
    """Construct the list of hash test cases.

    Each case has:
      - name: short identifier
      - token_ids: list[int]
      - parent_hash: str | None  (if str, must be a real get_hash_str output;
        we compute it from a parent token list during generation)
      - page_size: int | None

    The parent_hash field is filled at generation time, not hardcoded,
    so we exercise the real chain.
    """
    return [
        # --- router's actual call shape: no parent, no page_size ---
        {"name": "empty", "token_ids": [], "parent_token_ids": None, "page_size": None},
        {"name": "single", "token_ids": [1], "parent_token_ids": None, "page_size": None},
        {"name": "small", "token_ids": [1, 2, 3], "parent_token_ids": None, "page_size": None},
        {
            "name": "qwen_chat_prefix",
            "token_ids": [151644, 8948, 198, 2610, 525, 10950, 13, 151645, 198],
            "parent_token_ids": None,
            "page_size": None,
        },
        # uint32 boundary
        {"name": "uint32_max", "token_ids": [4294967295, 0, 1], "parent_token_ids": None, "page_size": None},
        # long list (page boundary stress)
        {"name": "range_256", "token_ids": list(range(256)), "parent_token_ids": None, "page_size": None},

        # --- with parent (chained) ---
        {
            "name": "child_of_small",
            "token_ids": [10, 20, 30],
            "parent_token_ids": [100, 200, 300],
            "page_size": None,
        },
        {
            "name": "child_of_qwen_prefix",
            "token_ids": [151644, 8948, 198, 785, 1234],
            "parent_token_ids": [151644, 8948, 198, 2610, 525, 10950, 13, 151645, 198],
            "page_size": None,
        },

        # --- paged mode (page_size > 1) ---
        {"name": "paged_2", "token_ids": [1, 2, 3, 4, 5, 6, 7, 8], "parent_token_ids": None, "page_size": 2},
        {"name": "paged_uneven", "token_ids": [1, 2, 3, 4, 5], "parent_token_ids": None, "page_size": 2},
        {"name": "paged_16", "token_ids": list(range(64)), "parent_token_ids": None, "page_size": 16},
        {"name": "paged_with_parent", "token_ids": [10, 20, 30, 40, 50, 60], "parent_token_ids": [1, 2, 3], "page_size": 2},
    ]


def _compute_sglang_hashes(cases: list[dict]) -> list[dict]:
    """Compute hashes via sglang.srt.mem_cache.utils.get_hash_str.

    Raises ImportError if sglang isn't installed.  Each case gets:
      - parent_hash: the actual hex string used as parent (or None)
      - hash: the returned hex string (or list of hex strings for paged)
      - int64: hash_str_to_int64(hash) for the single-hash cases
    """
    try:
        from sglang.srt.mem_cache.utils import (  # type: ignore
            get_hash_str,
            hash_str_to_int64,
        )
    except ImportError as e:
        print(
            "ERROR: could not import sglang.srt.mem_cache.utils.\n"
            "This script must run in an environment with sglang installed.\n"
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        raise

    results: list[dict] = []
    for case in cases:
        token_ids = case["token_ids"]
        parent_token_ids = case["parent_token_ids"]
        page_size = case["page_size"]

        # Compute parent_hash from parent_token_ids (chain)
        parent_hash = None
        if parent_token_ids is not None:
            parent_hash = get_hash_str(parent_token_ids)

        # Compute the actual hash
        try:
            if page_size is None:
                h = get_hash_str(token_ids, parent_hash)
                int64 = hash_str_to_int64(h)
            else:
                h = get_hash_str(token_ids, parent_hash, page_size=page_size)
                # For paged mode, hash_str_to_int64 isn't well-defined per-page;
                # store int64 of the first page only as a sanity check.
                int64 = hash_str_to_int64(h[0]) if h else None
        except Exception as e:
            h = None
            int64 = None
            error = f"{type(e).__name__}: {e}"
            print(f"WARN: case {case['name']!r} raised {error}", file=sys.stderr)
        else:
            error = None

        results.append(
            {
                "name": case["name"],
                "token_ids": list(token_ids),
                "parent_token_ids": list(parent_token_ids) if parent_token_ids else None,
                "parent_hash": parent_hash,
                "page_size": page_size,
                "hash": h,
                "int64": int64,
                "error": error,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="test/fixtures/hash_golden.json",
        help="Output JSON path (default: test/fixtures/hash_golden.json)",
    )
    args = parser.parse_args()

    cases = _build_cases()
    print(f"Computing hashes for {len(cases)} cases via sglang...")

    # Sanity: report sglang version if available
    try:
        import sglang  # type: ignore

        sglang_file = getattr(sglang, "__file__", "?")
        print(f"sglang loaded from: {sglang_file}")
    except Exception as e:
        print(f"WARN: could not import sglang top-level: {e}", file=sys.stderr)

    # Also probe the native_hash backend to confirm C++ was actually used
    try:
        from sglang.srt.mem_cache.cpp_utils.native_hash import (  # type: ignore
            _load_native_hash_module,
        )

        try:
            mod = _load_native_hash_module()
            print(f"native_hash C++ module loaded: {mod!r}")
        except Exception as e:
            print(f"WARN: native_hash C++ extension failed to load: {e}", file=sys.stderr)
            print(
                "      hashes below will fall back to whatever pure-Python path "
                "sglang uses when the C++ extension is unavailable.  This may "
                "not match the production algorithm.",
                file=sys.stderr,
            )
    except ImportError as e:
        print(f"WARN: cannot probe native_hash module: {e}", file=sys.stderr)

    try:
        results = _compute_sglang_hashes(cases)
    except ImportError:
        sys.exit(1)

    # Validate all cases produced a hash (no errors)
    errors = [r for r in results if r["error"] is not None]
    if errors:
        print(f"\n{len(errors)} case(s) failed; see warnings above.", file=sys.stderr)

    output = {
        "description": (
            "Golden hash values from sglang's native C++ SHA256 implementation. "
            "Generated by gen_hash_golden.py; consumed by "
            "test/unit/mem_cache/test_hash_golden.py to verify "
            "llm_router_utils.sglang.srt.mem_cache.utils produces "
            "byte-identical output."
        ),
        "algorithm": {
            "hash_func": "SHA256",
            "prior_digest_bytes": 32,
            "token_encoding": "uint32 little-endian (array 'I')",
            "output_hex_chars": 64,
            "int64_uses_first_hex_chars": 16,
        },
        "cases": _to_jsonable(results),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nWrote {len(results)} cases to {os.path.abspath(args.out)}")
    print("\nCases summary:")
    for r in results:
        if r["error"]:
            print(f"  [ERR] {r['name']}: {r['error']}")
            continue
        if isinstance(r["hash"], list):
            print(f"  [PAGED x{len(r['hash'])}] {r['name']}: first={r['hash'][0][:16]}...")
        else:
            print(f"  [OK] {r['name']}: hash={r['hash'][:16]}... int64={r['int64']}")


if __name__ == "__main__":
    main()
