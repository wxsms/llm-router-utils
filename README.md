# llm-router-utils

Lightweight extraction of sglang's reasoning parser, tool-call parser, and chat template rendering, for use in custom router services and lightweight LLM applications.

This library does **not** include any inference engine code. It only provides the "frontend" message processing pipeline: `OpenAIServingChat._process_messages` and its dependencies.

**Upstream source:** sglang [release/v0.5.16](https://github.com/sgl-project/sglang/tree/release/v0.5.16).

## What's included

Migrated modules under `llm_router_utils/sglang/srt/` (paths preserved from upstream):

- **`parser/`** — conversation templates (~50 model families, `SeparatorStyle` enum, `generate_chat_conv`), reasoning parser (`ReasoningParser` + detectors for deepseek/qwen3/mistral/gemma4/apertus/glm/inkling/...), harmony parser, inkling renderer/tokenizer, jinja template utils, template detection (auto-detection of reasoning/tool-call parser from chat template + tokenizer vocab), `TemplateManager` with `TokenizerLike` Protocol.
- **`function_call/`** — `FunctionCallParser` + 33 detector implementations (hermes, glm4/glm47, deepseekv3/v31/v32/v4, qwen3_coder, qwen25, kimik2, mistral, llama3.2, minicpm5, minimax_m2/m3, poolside_v1, step3, internlm, lfm2, mimo, gpt-oss, gigachat3, apertus2509, trinity, pythonic, inkling, cohere_command4, gemma4, hunyuan), `JsonArrayParser`, schema utils.
- **`entrypoints/openai/`** — `protocol.py` (full OpenAI-compatible request/response types, ~1900 lines), `serving_base.py`, slimmed `serving_chat.py` (only `_process_messages` call chain kept: `_apply_jinja_template`, `_apply_conversation_template`, `_encode_messages`, `_get_reasoning_from_request`, `_patch_reasoning_skip_special_tokens`, etc.), `chat_encoding.py`, `encoding_dsv32.py`, `encoding_dsv4.py`, `sse_utils.py`, `usage_processor.py`, `utils.py`.
- **`managers/`** — slimmed `TokenizerManager` (initializer + `init_model_config` + `init_tokenizer_and_processor` only), slimmed `io_struct.py` (`GenerateReqInput`/`EmbeddingReqInput`), `embed_types.py` stub.
- **`configs/`** — slimmed `ModelConfig` (loads HF config via `transformers.AutoConfig`; exposes `hf_config`, `is_multimodal`, `get_default_sampling_params`, `context_length`).
- **`tokenizer/`** — `tiktoken_tokenizer.py`.
- **`disaggregation/`** — `kv_events.py` (KV cache event structs: `EventBatch`, `KVCacheEvent`, `StorageMedium`, `BlockStored`, `BlockRemoved`, `AllBlocksCleared`, `KVEventBatch` — publisher machinery stripped).
- **Top-level slimmed files** — `environ.py` (env var registry), `server_args.py` (`ServerArgs`/`PortArgs` with `_process_messages`-relevant fields), `srt/utils/common.py` (`ImageData`/`VideoData`/`read_system_prompt_from_file`), `sglang/utils.py` (`convert_json_schema_to_str`/`is_in_ci`/`TypeBasedDispatcher`/`LazyImport`).

## What's NOT included

Inference engine code is intentionally stripped: schedulers, model loaders, layer implementations, CUDA/Triton kernels, sampling, constrained decoding, speculative decoding, LoRA runtime, distributed runtime, KV cache manager, HTTP server, multimodal processing, observability, and all CLI/launch scripts.

## Installation

```bash
pip install llm-router-utils
```

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

## License

Apache 2.0, adapted from [sglang](https://github.com/sgl-project/sglang).
