# llm-router-utils

Lightweight extraction of sglang's reasoning parser, tool-call parser, and chat template rendering, for use in custom router services and lightweight LLM applications.

This library does **not** include any inference engine code. It only provides the "frontend" message processing pipeline: `OpenAIServingChat._process_messages` and its dependencies.

**Upstream source:** sglang [release/v0.5.17](https://github.com/sgl-project/sglang/tree/release/v0.5.17).

## Installation

```bash
pip install llm-router-utils
```

### CPU-only torch (optional, smaller on Linux)

This library depends on `xgrammar`, which declares `torch>=1.10.0`. By default pip pulls the CUDA-enabled torch wheel from PyPI (~502 MB on Linux x86_64). Routers don't use torch for GPU ops — only a lazy `torch.version.hip` / `torch.npu.is_available()` probe that returns `False` when torch is absent — so CPU-only torch is sufficient.

To install CPU-only torch, install it from the PyTorch CPU index **before** this package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llm-router-utils
```

The second command sees torch already satisfied and skips the CUDA download. On Linux x86_64 this saves ~319 MB (502 → 183). On Windows there is no size difference (both ~116 MB), so this step is only useful on Linux.

## Usage

```python
from llm_router_utils.sglang.srt.configs.model_config import ModelConfig
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from llm_router_utils.sglang.srt.server_args import ServerArgs, PortArgs

server_args = ServerArgs(model_path="Qwen/Qwen3-32B", tool_call_parser="qwen3_coder")
port_args = PortArgs()
tokenizer_manager = TokenizerManager(server_args, port_args)
template_manager = TemplateManager()
template_manager.initialize_templates(
    tokenizer_manager=tokenizer_manager,
    model_path=server_args.model_path,
    chat_template=server_args.chat_template,
)
openai_serving_chat = OpenAIServingChat(tokenizer_manager, template_manager)
result = openai_serving_chat._process_messages(request, is_multimodal=False)
```

## What's included

Migrated modules under `llm_router_utils/sglang/srt/` (paths preserved from upstream):

| Module | Description |
|---|---|
| `parser/` | Conversation templates (~50 model families), `ReasoningParser` + detectors, harmony/inkling, jinja utils, template detection, `TemplateManager` with `TokenizerLike` Protocol |
| `function_call/` | `FunctionCallParser` + 33 detectors (hermes, glm, deepseek, qwen, kimi, mistral, …), `JsonArrayParser`, schema utils |
| `entrypoints/openai/` | `protocol.py` (~1900 lines, full OpenAI types), slimmed `serving_chat.py` (only `_process_messages` chain), `serving_base.py`, `chat_encoding.py`, `encoding_dsv32/dsv4.py`, `sse_utils.py`, `usage_processor.py`, `utils.py` |
| `managers/` | Slimmed `TokenizerManager` (uses upstream `get_tokenizer` for byte-parity incl. `SGLANG_PATCH_TOKENIZER`), slimmed `io_struct.py`, `embed_types.py` stub |
| `configs/` | Slimmed `ModelConfig` (uses upstream `get_config`; exposes `hf_config`/`is_multimodal`/`get_default_sampling_params`/`context_length`), `model_config_parser_registry.py` |
| `tokenizer/` | `tiktoken_tokenizer.py` |
| `disaggregation/` | `kv_events.py` — KV cache event structs (`EventBatch`, `KVCacheEvent`, `StorageMedium`, `BlockStored`, `BlockRemoved`, `AllBlocksCleared`, `KVEventBatch`) |
| `mem_cache/` | `utils.py` — pure-Python SHA256 hash helpers, byte-identical to sglang's C++ extension |
| `observability/` | `metrics_collector.py` — data classes only: `QueueCount`, `SchedulerStats`, `compute_routing_key_stats`. Heavy `*MetricsCollector` classes stripped |
| `utils/hf_transformers/` | Restored `common.py`/`config.py`/`tokenizer.py`/`mistral_utils.py` — upstream `get_tokenizer`/`get_config`. Slimmed `hf_transformers_patches.py` (torch-free only). `patch_tokenizer.py` verbatim |
| `connector/` (stub) | `create_remote_connector` raises `NotImplementedError` |
| Top-level slimmed files | `environ.py` (env var registry) · `server_args.py` (`device`/`revision`/`tokenizer_backend` + `PortArgs.init_new`) · `srt/utils/common.py` (`ImageData`/`VideoData`/`read_system_prompt_from_file` + hf helpers) · `sglang/utils.py` (`convert_json_schema_to_str`/`is_in_ci`/`TypeBasedDispatcher`/`LazyImport`) |

## What's NOT included

Inference engine code is intentionally stripped: schedulers, model loaders, layer implementations, CUDA/Triton kernels, sampling, constrained decoding, speculative decoding, LoRA runtime, distributed runtime, KV cache manager, HTTP server, multimodal processing, observability, and all CLI/launch scripts.

## Version Mapping

The mapping between this repo's releases and upstream sglang versions, so users can find the release that matches a given sglang version:

| This repo | Upstream sglang |
|---|---|
| `v0.2.2` | [v0.5.16](https://github.com/sgl-project/sglang/tree/release/v0.5.16) |
| `v0.3.0` | [v0.5.17](https://github.com/sgl-project/sglang/tree/release/v0.5.17) |

> See [HOW_TO_UPGRADE.md](HOW_TO_UPGRADE.md) for the upgrade workflow.

## License

Apache 2.0, adapted from [sglang](https://github.com/sgl-project/sglang).
