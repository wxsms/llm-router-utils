# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A lightweight extraction of sglang's frontend message-processing layer (chat template rendering, reasoning parser, tool-call parser, `OpenAIServingChat._process_messages` call chain) for custom router services. All inference engine code (schedulers, model loaders, sampling, HTTP server, CUDA kernels, distributed runtime) is stripped.

## Test command

```bash
PYTHONPATH=src python -m pytest test/
```

## Directory layout

```
src/llm_router_utils/sglang/         # sglang submodule (kept as a path segment for future vllm/etc. siblings)
└── srt/
    ├── parser/                      # conversation templates, reasoning parser, harmony/inkling, template detection/manager
    ├── function_call/               # 33 detector implementations + FunctionCallParser + utils
    ├── entrypoints/openai/          # protocol, serving_base, slimmed serving_chat, encoding_dsv32/dsv4, etc.
    ├── managers/                    # slimmed tokenizer_manager + io_struct + embed_types
    ├── configs/                     # slimmed model_config
    ├── tokenizer/                   # tiktoken_tokenizer
    ├── environ.py                   # env var registry (cuda_coredump import dropped)
    ├── server_args.py               # slimmed ServerArgs/PortArgs
    └── utils/                       # slimmed common.py (ImageData/VideoData/read_system_prompt_from_file)
test/
├── unit/{parser,function_call}/     # migrated from upstream sglang test/registered/unit/ — golden regression baseline
└── integration/test_process_messages.py
```

The single public entry point is `OpenAIServingChat._process_messages`; everything else exists to serve it.

## Critical rules

- **Do not functionally rewrite retained sglang code.** Only delete or slim (cut methods, cut imports, replace type annotations with Protocols). Logic changes break parity with upstream and invalidate the 668 migrated unit tests as a regression baseline. Bugs found in retained code should be fixed upstream first, then re-imported.
- **Migrated unit tests are the regression baseline.** Test assertions must stay byte-identical to upstream; only imports/`CustomTestCase`→`unittest.TestCase`/`register_cpu_ci` calls were adapted.
- **No direct `import torch` in `src/`.** Torch comes in only transitively via xgrammar's tvm_ffi (required by glm47_moe_detector/inkling_detector for tool_call_constraint construction). Keep this invariant.
- **xgrammar imports must be `try/except ImportError`** at module top-level so the package still imports when xgrammar is absent (falls back to `typing.Any`).
