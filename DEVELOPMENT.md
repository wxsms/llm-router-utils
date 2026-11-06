# Development Guide

This document is the development reference for the llm-router-utils repository — a lightweight extraction of sglang's frontend message-processing layer. It covers the upstream sync workflow, trimming rules, verification checks, release process, and common pitfalls.

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

   Then align pinned dependencies (see Step 3.5) — upstream frequently bumps `transformers`, `openai`, `xgrammar`, and the `fastokens` extra between releases.

4. **Confirm the baseline tag**: the current sglang version this repo tracks is recorded in the "Version Mapping" table in `README.md`.

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

### 6. License attribution (Apache 2.0 §4)

This repo is a derivative work of sglang. When adopting or modifying upstream files:

- **Retain upstream copyright headers** (§4(c)): if an upstream file carries a `# Copyright ... SGLang Team` header, the retained copy must keep it. Do not replace it with a repo-local docstring. Files whose upstream counterpart has no header need not add one.
- **Mark modified files** (§4(b)): every slimmed (modified) file that carries a header must also carry a "Derivative work: slimmed for llm-router-utils" notice block immediately after the upstream header. Fully-retained files (byte-identical to upstream modulo import rewriting) need no such notice.
- `LICENSE` and `NOTICE` at repo root must be preserved.

Run `make check-license` to verify (see Step 5).

## Step 3.5: Align Pinned Dependencies

Upstream sglang pins several third-party dependencies with exact versions (`==`) in `vendor/sglang/python/pyproject.toml`. This repo must mirror those pins in its own `pyproject.toml` for the packages the extracted code actually imports. Mismatched pins cause runtime drift between the migrated code and the upstream behavior the tests were written against (this actually happened: `transformers` stayed at the v0.5.16-era pin `5.12.1` across the v0.5.17→v0.5.22 upgrades until it was realigned to `5.19.0`).

After switching the submodule to a new release, diff the pins:

```bash
# Pins declared by the NEW upstream version
git -C vendor/sglang show v0.5.NEW:python/pyproject.toml | grep -E '"[a-zA-Z_-]+=='

# Pins currently in this repo
grep -E '"[a-zA-Z_-]+==' pyproject.toml
```

For every pinned package that exists in this repo's `dependencies` (or `optional-dependencies`), update the pin in `pyproject.toml` to match upstream exactly — including the `fastokens` extra, whose constraint upstream raises in minor releases (v0.5.21: `>=0.1.1,<0.2.0` → v0.5.22: `>=0.3.2`).

Notes:
- Packages upstream pins but this repo does not import (torch, flashinfer, CUDA libraries, sentencepiece, tokenizers, openai-harmony, etc.) are irrelevant — the extracted frontend layer never imports them.
- The reverse check also matters: if this repo's code imports a package that is only a transitive dependency today (e.g. `gguf`, used behind a `try/except ImportError` guard in `hf_transformers_patches.py`), it works without a pin but consider declaring it if an upgrade makes the import mandatory.
- Re-run `python -m pip install -e ".[test]"` after changing pins so the local environment matches the declared constraints, then run the full test suite (Step 5).

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

Run the full verification suite — tests plus the three invariant checks. All must pass before committing the upgrade.

```bash
cd /path/to/llm_router_utils
PYTHONPATH=src python -m pytest test/ -q   # 1. regression baseline
make check-torch                            # 2. no direct torch import
make check-parity                           # 3. no NEW drift vs upstream
make check-license                          # 4. Apache 2.0 §4 attribution
```

All tests must pass. If upstream changed an implementation (e.g. the inkling_detector rewrite), the corresponding tests must be synced to the new upstream version.

### Torch-import check

```bash
make check-torch
```

Verifies the `CLAUDE.md` invariant: no direct `import torch` / `from torch import ...` in `src/`. Torch may only enter transitively via xgrammar's tvm_ffi. If an upstream diff introduces a torch import used only by inference-engine code paths, drop that import (see Step 3, rule 2).

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

### License-compliance check

```bash
make check-license
```

Verifies the Apache 2.0 §4 redistribution obligations for this derivative work:

- **§4(a)** — `LICENSE` present with the Apache 2.0 header.
- **§4(c)** — `NOTICE` present and naming sglang as upstream; every retained file whose upstream counterpart carries a SGLang copyright header must retain one. Files whose upstream counterpart has no header are not required to add one (the Apache 2.0 appendix boilerplate is optional, not a §4 obligation).
- **§4(b)** — every slimmed (modified) file that carries a header must also carry a "Derivative work" notice. The checker reads `scripts/parity_baseline.txt` to know which files are slimmed, so it stays in sync with the parity checker.

When you adopt a new upstream file that carries a header, the checker will flag it if you strip the header (§4(c)) or, for a slimmed file, omit the "Derivative work" notice (§4(b)). Fix by restoring the upstream header and appending the notice block documented in `NOTICE`. See `scripts/check_license.py -v` for a per-file status table.

## Step 6: Update Version Records

1. Append a row to the "Version Mapping" table in `README.md`.
3. Commit (do not tag yet — tagging is part of the release process, see below).

## Releasing a New Version

Releases are triggered by pushing a `v*` tag to `origin`, which fires the `publish.yml` GitHub Actions workflow to build and publish to PyPI via Trusted Publishers. The process is irreversible — PyPI releases cannot be deleted or overwritten, so verify everything locally first.

### 1. Pre-release verification

Run the full verification suite on a clean checkout of `master`:

```bash
cd /path/to/llm_router_utils
git checkout master
git pull
PYTHONPATH=src python -m pytest test/ -q   # regression baseline
make check-torch                            # no direct torch import
make check-parity                           # no NEW drift vs upstream
make check-license                          # Apache 2.0 §4 attribution
```

All four must pass. Also confirm the `vendor/sglang` submodule is initialized (`git submodule status`) so the parity/license checks have their upstream reference.

### 2. Bump the version

The version lives in two places and they must agree:

- `pyproject.toml` — `[project] version = "X.Y.Z"`
- `src/llm_router_utils/sglang/_version.py` — `__version__ = "X.Y.Z"`

Update both to the new version. Verify at runtime:

```bash
PYTHONPATH=src python -c "from llm_router_utils import __version__; print(__version__)"
```

### 3. Update version records

- Ensure the "Version Mapping" table in `README.md` has a row for the new release pointing at the sglang version this release tracks (added during the upgrade, see Step 6 above).

### 4. Build and verify locally

```bash
rm -rf dist/ build/ src/*.egg-info
python -m pip install --quiet build
python -m build
ls -la dist/   # expect <name>-X.Y.Z.tar.gz and <name>-X.Y.Z-py3-none-any.whl
```

Confirm both an sdist (`.tar.gz`) and a wheel (`.whl`) are produced with the correct version in the filename. Clean up the artifacts afterward so they are not committed:

```bash
rm -rf dist/ build/ src/*.egg-info
```

### 5. Commit, tag, and push

```bash
git add pyproject.toml src/llm_router_utils/sglang/_version.py README.md
git commit -m "release: vX.Y.Z (sglang vA.B.C)"
git tag -a vX.Y.Z -m "vX.Y.Z — sglang vA.B.C"
git push origin master
git push origin vX.Y.Z
```

Pushing the tag triggers the `publish.yml` workflow. **This is the point of no return** — once the workflow succeeds, the package is live on PyPI.

### 6. Confirm the publish

Monitor the workflow:

```bash
gh run watch --exit-status   # or check https://github.com/<owner>/llm-router-utils/actions
```

Once it completes, verify the package is live on PyPI (allow a minute for index propagation):

```bash
python -c "import urllib.request, json; d=json.load(urllib.request.urlopen('https://pypi.org/pypi/llm-router-utils/json')); print('latest:', d['info']['version'])"
```

### Release checklist

- [ ] All four checks pass (pytest, check-torch, check-parity, check-license)
- [ ] Dependency pins in `pyproject.toml` match the tracked upstream release (Step 3.5)
- [ ] `pyproject.toml` and `_version.py` bumped to the same version
- [ ] `PYTHONPATH=src python -c "from llm_router_utils import __version__"` prints the new version
- [ ] README version table has a row for the new release
- [ ] `python -m build` produces correct sdist + wheel
- [ ] Committed on `master`, tagged `vX.Y.Z` (annotated)
- [ ] Pushed `master` + tag to `origin`
- [ ] `publish.yml` workflow succeeded
- [ ] Package visible on PyPI

## Common Pitfalls

- **`git diff` path prefix**: must use `sglang/`, not `python/sglang/`, or the result is empty.
- **Read-only subagents**: the `Explore` subagent cannot edit files; apply changes yourself.
- **Stale grep line numbers**: after editing a file, `grep_search` line numbers go stale — re-locate with `grep -n`.
- **Special characters**: when handling code containing XML-like tags (`<...>`), tool calls may be mis-parsed; use `create_file` to write a helper script or base64-encode the content.
- **CRLF**: a Windows checkout produces CRLF line endings; use `diff --strip-trailing-cr` when comparing against upstream LF.
- **"No change needed" for slimmed files**: if every upstream change lands in already-stripped inference-engine code (scheduling, dispatch, metrics, cuda-graph, etc.), the slimmed version needs no change. Verify this file-by-file.
- **Stale dependency pins after an upgrade**: bumping the submodule does not update `pyproject.toml`. Always run the pin diff in Step 3.5 — a missed pin (like `transformers==5.12.1` surviving three upstream bumps) silently tests against an old library version.
- **Capabilities extracted into compiled platform-specific packages**: check upstream's `pyproject.toml` for new pins guarded by `sys_platform` / `platform_machine` markers. v0.5.23 moved DeepSeek-V4/V4.1 encoding into `deepseek-recipe` and Inkling rendering into `tml-renderers`; neither ships a Windows wheel. Mirror the marker exactly — do not silently promote such a package to an install-required dependency.
- **Tests that import a newly-extracted package**: sync the test from upstream first, then guard each class that reaches the extension with `unittest.skipUnless(<package imports>, "...")`. Guard the *whole* file's dependency surface — a module-level helper (e.g. `_ids()` calling `load_tml_renderers()`) can drag a second class into the dependency even when the class body looks independent. See `test/unit/parser/test_inkling_renderer.py` and `test/unit/function_call/test_deepseekv41_detector.py`.
- **Do not leave a test at the old version when the code under test changed**: a stale test plus a `skipUnless` guard is the worst combination — it looks protected, passes green everywhere the extension is absent, and silently drops the golden-reference assertions where it is present. Compare method sets against upstream (`python - <<` an AST diff) after every sync.
- **Upstream test fixtures that assume `msgspec.Struct`**: this repo's `ServerArgs` is a plain dataclass, so `msgspec.Struct.__setattr__(args, ...)` must become `object.__setattr__(args, ...)` when syncing a test.
- **JSON/`.jinja` fixtures copied via PowerShell**: `>` and `Out-File` emit UTF-16/BOM, which breaks `json.loads` and jinja reads with a `UnicodeDecodeError`. Always re-encode as UTF-8 without BOM (`[System.IO.File]::WriteAllText(path, text, (New-Object System.Text.UTF8Encoding($false)))`) and diff against the upstream fixture to confirm.

## Shim Fidelity

The hand-written shims (`runtime_context.py`, `arg_groups/*`, `sampling/sampling_params.py`) are the highest-risk part of any upgrade, because `check-parity` can only see that they are *different* — not that they are *wrong*. Two failure modes have already occurred; guard against both:

1. **Hand-maintained allowlists drift.** `_ServingConfig` originally listed the serving fields explicitly and omitted `hf_chat_template_name`, which silently downgraded every model with a dict of named HF chat templates (the `AttributeError` was swallowed by an upstream `except Exception` and degraded to a warning). Shims must **pass through to the real object** and use a defaults dict only when nothing is installed, so an upstream rename fails loudly once instead of silently forever.
2. **API-shaped but behaviourally wrong.** `resolving_view()` returned a read-only proxy but dropped upstream's live re-read of `_resolved_overrides`, making `resolve_auto_parsers`' declarations invisible. When slimming a class, port the `__getattr__` body, not just the shape.

**Before merging any upgrade**, grep the retained tree for every attribute read off each shim and assert it resolves:

```python
# e.g. for runtime_context
import re, pathlib
src = pathlib.Path("src/llm_router_utils/sglang/srt/entrypoints/openai/serving_chat.py").read_text()
for f in set(re.findall(r"get_serving\(\)\.(\w+)", src)):
    getattr(get_serving(), f)          # raises if the shim is missing it
```

Also assert the *wiring* exists: a shim whose setter has zero callers fails closed at runtime. `set_runtime_context` must be called from `TokenizerManager.init_model_config` before `OpenAIServingChat` reads `get_model()`.

## Upgrade Deviations

Deliberate, documented departures from upstream that a future upgrade must preserve.

### v0.9.0 (sglang v0.5.23)

Upstream v0.5.23 replaced its in-tree DeepSeek-V4/V4.1 encoders
(`entrypoints/openai/encoding_dsv4.py`, `encoding_dsv41.py`) with DeepSeek's
native `deepseek-recipe` extension, and routed Inkling through `tml-renderers`.
Both are compiled, both lack Windows wheels, and both are pinned upstream behind
platform markers.

This repo follows upstream: the two vendored encoders were **deleted**, and
`pyproject.toml` carries the same platform markers. Consequences:

- `deepseek-recipe` is a **hard** dependency on Linux/macOS and absent on
  Windows, where `dsv4`/`dsv41` encodings raise an `ImportError` naming the
  package.
- `tml-renderers` is an **optional** `inkling` extra, because it additionally
  requires PyTorch >= 2.10, which this package otherwise avoids.
- The corresponding tests are platform-guarded, not removed.

### Stub modules

Three upstream subsystems were replaced by minimal behavioural shims rather
than being carried over. They are recorded in `scripts/parity_baseline.txt`, so
the parity checker will not flag them; keep them slim rather than adopting
upstream wholesale on later upgrades.

| Module | Upstream | Shim |
|---|---|---|
| `srt/runtime_context.py` | ~2,200-line process-wide config publication layer (arg-group namespaces, role-scoped reads) | `get_serving()` / `get_model()` read-only proxies over `ServerArgs` / `ModelConfig` |
| `srt/arg_groups/` | CLI/env/model-override resolution machinery | resolution stash (`declare_resolution` / `resolution_result`) + read-only resolving view |
| `srt/sampling/sampling_params.py` | full `SamplingParams` struct + watermark validation (torch) | the two private request-transport helpers `serving_chat` uses |

Two branches inside these shims are **inert by construction** — do not mistake
them for working features when reviewing a future upgrade:

| Inert branch | Why |
|---|---|
| `_architecture_parser_defaults` (`template_detection.py`) | early-returns `{}` because nothing ever attaches `ServerArgs._model_config`; no `src/` code assigns it |
| `muse_spark_chat_template_default` (`arg_groups/muse_spark_hook.py`) | hard-coded `return None`; upstream derives `--chat-template` from the checkpoint during launch, which this repo does not do |

Architecture-derived parser defaults and Muse Spark template defaulting are
therefore unsupported. If a future version needs them, the shim has to grow a
real `_model_config` attachment first — not just delete the early return.

`managers/tokenizer_manager.py` gained a `config_value()` method returning the
*resolved* value — it consults the declaration stash via `resolution_result`, not
just the raw `ServerArgs` field. Reading the raw field silently returned `"auto"`
for every `--*-parser auto` field, disabling tool-call parsing and turn grammar.
`_DECLARED_ONLY_DEFAULTS` covers fields that exist upstream only as a
declaration (`grammar_backend` → `"xgrammar"`).

**The stash is only useful if something populates it.** Upstream calls
`resolve_auto_parsers` once from `entrypoints/engine.py` before the tokenizer is
built; that module does not exist here, so the resolution never ran and every
`auto` field survived as the literal sentinel. `TokenizerManager` now calls it
in `__init__`, *before* `init_tokenizer_and_processor` (detection loads its own
tokenizer), best-effort. As a backstop, `config_value` raises on an unresolved
sentinel for the parser fields rather than handing a caller `"auto"` — a value
`ToolCallParserEnum.get()` maps to `None`, which accepts tool calls and then
silently never parses them.

`runtime_context` keeps two levels: `set_runtime_context(...)` registers the
process default (last writer wins), and `use_runtime_context(...)` binds a
`ContextVar` scope that both accessors prefer. Pass `None` explicitly to clear a
slot — omitting an argument leaves it unchanged.

**Scope construction, not requests.** `hf_chat_template_name`,
`default_chat_template_kwargs` and the encoding adapter's `revision` are read
during `TokenizerManager`/`TemplateManager`/`OpenAIServingChat` construction, so
they are captured into the instance and a request-time scope cannot restore
them. A multi-model router must build each model's objects inside that model's
`use_runtime_context` block. `test/unit/test_runtime_context.py` pins both
halves (`test_construction_scoped_builds_see_their_own_config` and
`test_request_time_scope_cannot_restore_setup_time_values`).

Because these shims are in the baseline, `check-parity` will report them as
"baselined" rather than failing. Review them by hand when upstream touches the
corresponding real modules, and widen a shim only when a retained caller needs
more of the upstream surface.

## Release Checklist Addition

Before tagging, confirm for every upgrade that removes or gates a capability:

- [ ] README has an explicit **breaking change** note naming the version it
      changed in and what users lose (e.g. `dsv4`/`dsv41` on Windows in v0.9.0).
- [ ] Deleted public module paths are called out by name.
- [ ] Any new platform-gated extra records the Python floor its wheels impose
      (`tml-renderers` is `cp311-abi3`, so the `inkling` extra needs >= 3.11
      even though the base package supports 3.10).
- [ ] `make check-version` passes: `pyproject.toml`, `_version.py`, the README
      Version Mapping row and the `.gitmodules` submodule branch all agree.
- [ ] `platform-extras` CI asserts an exact pass count, not merely "no skips" —
      `no tests ran`, `N deselected` and `N xfailed` all lack the word
      "skipped" and would otherwise pass green.

## Version Table

See the "Version Mapping" section in `README.md`.
