# Upgrade Guide: Syncing with Upstream sglang

This repository is a lightweight extraction of sglang's frontend message-processing layer. When upstream sglang releases a new version, the retained files in this repo must be synced to the new version. This document describes the full upgrade workflow, trimming rules, and common pitfalls.

Upstream sglang is pinned as a git submodule at `vendor/sglang` (branch `release/vX.Y.Z`). Use it as the reference for diffs and file content during upgrades.

## Prerequisites

1. **Initialize the submodule** (if not already checked out):
   ```bash
   cd /path/to/llm_router_utils
   git submodule update --init vendor/sglang
   ```

2. **Create an upgrade branch from master**:
   ```bash
   git checkout master
   git checkout -b upgrade/sglang-0.5.X
   ```

3. **Switch the submodule to the target release** (when upgrading to a new sglang version):
   ```bash
   cd vendor/sglang
   git fetch origin
   git checkout release/v0.5.X
   cd ..
   git add vendor/sglang
   ```

4. **Confirm the baseline tag**: the current sglang version this repo tracks is recorded in the "Upstream source" line of `README.md` and in the version table at the bottom.

## Step 1: Determine the Change Set

Let `$UPSTREAM` denote the submodule path (e.g. `vendor/sglang`). **Note**: use the `python/sglang/` prefix (the submodule's repo root has a `python/` directory) when running `git diff`, and use the `sglang/` prefix (without `python/`) — both work, but `sglang/` is shorter.

```bash
cd $UPSTREAM
# List all files changed between the two versions under srt/
git diff --name-status v0.5.OLD v0.5.NEW -- sglang/srt/ > /tmp/changed.txt

# List the files retained in this repo
cd /path/to/llm_router_utils
find src/llm_router_utils/sglang -name "*.py" | sed 's|src/llm_router_utils/sglang/||' | sort > /tmp/kept.txt
```

Compute the intersection: for each `M`/`A` file in `/tmp/changed.txt`, check whether its path (stripping the `python/sglang/` prefix) appears in `/tmp/kept.txt`. The intersection is the set of files to update.

```bash
cd $UPSTREAM
while read status path; do
  rel=$(echo "$path" | sed 's|^python/sglang/||')
  if grep -qx "$rel" /tmp/kept.txt; then
    echo "$status $rel"
  fi
done < /tmp/changed.txt
```

**New files** (`A`): only adopt function-call detectors (this repo retains all detectors). Other new files (inference engine, config specs, observability, etc.) are not adopted.

## Step 2: Apply Changes

### Fully-retained files

For files that are byte-identical to upstream 0.5.OLD (modulo import rewriting), replace directly with the new upstream version:

```bash
cd $UPSTREAM
git show v0.5.NEW:python/sglang/srt/<path> \
  | sed -e 's|from sglang\.|from llm_router_utils.sglang.|g' \
        -e 's|import sglang\.|import llm_router_utils.sglang.|g' \
  > /path/to/llm_router_utils/src/llm_router_utils/sglang/srt/<path>
```

Typical fully-retained files: `parser/reasoning_parser.py`, the detectors under `function_call/`, `environ.py`.

### Slimmed files

For already-trimmed files (e.g. `configs/model_config.py`, `managers/tokenizer_manager.py`, `server_args.py`, `utils/common.py`, `entrypoints/openai/serving_chat.py`), **do not** replace wholesale. Manually diff against upstream and apply only the logic changes that fall on retained portions:

```bash
cd $UPSTREAM
git diff v0.5.OLD v0.5.NEW -- sglang/srt/<path>
git show v0.5.NEW:python/sglang/srt/<path>   # full content for reference
```

Apply changes hunk-by-hunk with an edit tool (e.g. `replace_string_in_file`). Skip upstream diff hunks that touch methods already removed in this repo.

## Step 3: Trimming & Adaptation Rules (Hard Invariants)

The following rules come from `CLAUDE.md` and **must be strictly followed**:

### 1. Import rewriting

- `from sglang.xxx` → `from llm_router_utils.sglang.xxx`
- `import sglang.xxx` → `import llm_router_utils.sglang.xxx`
- **Note**: `sed` only rewrites `from sglang.`/`import sglang.`; it does not touch `sglang.` inside string literals (e.g. `ModuleType("sglang.srt...")` must be handled separately).

### 2. No direct `import torch`

`src/` must not contain `import torch` or `from torch import ...`. Torch may only enter transitively via xgrammar's tvm_ffi. If an upstream diff adds a torch import used only by inference-engine code paths, drop that import.

### 3. xgrammar imports must be try/except

All xgrammar-related imports must be wrapped in `try/except ImportError`, falling back to `typing.Any`:

```python
try:
    from xgrammar import StructuralTag
except ImportError:
    StructuralTag = Any
```

If upstream adds xgrammar submodule imports (e.g. `from xgrammar.structural_tag import ...`), provide an `Any` fallback for every symbol.

### 4. Do not functionally rewrite retained code

Only delete or slim — never rewrite logic. Logic changes in this repo's files must align with upstream. If a bug is found in retained code, fix it upstream first, then re-import; do not patch it locally.

### 5. Migrated tests are the regression baseline

Test assertions must stay byte-identical to upstream, with only these adaptations:
- Remove `from sglang.test.ci.ci_register import register_cpu_ci` and `register_cpu_ci(...)` calls
- `CustomTestCase` → `unittest.TestCase`
- `from sglang.srt.utils.hf_transformers_utils import get_tokenizer` → `from transformers import AutoTokenizer as _AT; get_tokenizer = lambda name, **kw: _AT.from_pretrained(name, **kw)`
- `sglang.` → `llm_router_utils.sglang.`

## Step 4: Windows Test Adaptation

`tempfile.NamedTemporaryFile` holds an exclusive lock on Windows and cannot be re-opened by name. Tests that read a temporary jinja template must use `mkstemp` instead:

```python
import os, tempfile
fd, path = tempfile.mkstemp(suffix=".jinja")
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(template_content)
    # ... use path ...
finally:
    os.unlink(path)
```

## Step 5: Verify

```bash
cd /path/to/llm_router_utils
PYTHONPATH=src python -m pytest test/ -q
```

All tests must pass. If upstream changed an implementation (e.g. the inkling_detector rewrite), the corresponding tests must be synced to the new upstream version.

### Parity check

After syncing, run the parity checker to catch drift between retained files and the upstream submodule:

```bash
make check-parity
```

This diffs every retained `src/llm_router_utils/sglang/srt/**/*.py` against its `vendor/sglang/python/sglang/srt/**` counterpart, normalizing away sanctioned slimming (import rewrites, xgrammar `try/except` guards, `cuda_coredump` import, `Any` in typing imports). It classifies remaining diffs as:

- **match** — byte-identical after normalization (fully-retained files; the goal for detectors, `reasoning_parser.py`, `environ.py`, etc.),
- **sanctioned slimming** — pure deletions or expected-pattern diffs only,
- **logic drift** — substantive divergence requiring human review.

Deliberately slimmed files (`serving_chat.py`, `model_config.py`, `template_detection.py`, `server_args.py`, `utils/common.py`, etc.) legitimately show logic drift because of Protocol annotations, deleted methods, and `hf_transformers_utils` → `transformers` substitutions that a generic normalizer cannot safely collapse. Their paths are recorded in `scripts/parity_baseline.txt`. The check **fails only on NEW logic drift** (a regression); baselined files are reported but do not fail the check.

When you legitimately add or remove a slimmed file, regenerate the baseline:

```bash
make update-parity-baseline
```

Inspect the diff of `scripts/parity_baseline.txt` before committing — it should only change when the set of slimmed files intentionally changes, never as a side effect of an upgrade. To audit a specific file's diff in full:

```bash
python scripts/check_parity.py --diff srt/function_call/inkling_detector.py
python scripts/check_parity.py --show-diffs          # all logic-drift files
python scripts/check_parity.py --no-baseline         # fail on any drift, ignore baseline
```

## Step 6: Update Version Records

1. Update the "Upstream source" line in `README.md` to the new version.
2. Append a row to the version table in `README.md`.
3. Commit and tag.

## Common Pitfalls

- **`git diff` path prefix**: must use `sglang/`, not `python/sglang/`, or the result is empty.
- **Read-only subagents**: the `Explore` subagent cannot edit files; apply changes yourself.
- **Stale grep line numbers**: after editing a file, `grep_search` line numbers go stale — re-locate with `grep -n`.
- **Special characters**: when handling code containing XML-like tags (`<...>`), tool calls may be mis-parsed; use `create_file` to write a helper script or base64-encode the content.
- **CRLF**: a Windows checkout produces CRLF line endings; use `diff --strip-trailing-cr` when comparing against upstream LF.
- **"No change needed" for slimmed files**: if every upstream change lands in already-stripped inference-engine code (scheduling, dispatch, metrics, cuda-graph, etc.), the slimmed version needs no change. Verify this file-by-file.

## Version Table

See the "Version Mapping" section in `README.md`.
