# llm-router-utils Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 sglang 抽取 parser/function_call/template_manager 与 `_process_messages` 调用链，构建轻量级 `llm_router_utils` 库。

**Architecture:** 先整体拷贝 `sglang_ksogit/python/sglang/` 到 `src/llm_router_utils/sglang/`，分阶段删除推理相关子树，再按 `_process_messages` 反向依赖闭包精细裁剪瘦身文件，迁移原版单元测试作为黄金基准，最后补集成测试与 pyproject。

**Tech Stack:** Python 3.10+, pydantic, jinja2, transformers, openai, partial_json_parser, tiktoken, pytest

---

## File Structure

迁移后的关键文件清单（每个文件的职责）：

**完整迁移（不裁剪）：**
- `src/llm_router_utils/sglang/srt/parser/*.py`（9 个文件）— chat 模板、reasoning parser、harmony/inkling、template detection/manager
- `src/llm_router_utils/sglang/srt/function_call/*.py`（34 个文件）— 33 个 detector + FunctionCallParser + utils
- `src/llm_router_utils/sglang/srt/tokenizer/tiktoken_tokenizer.py` — tiktoken 包装
- `src/llm_router_utils/sglang/srt/entrypoints/openai/protocol.py` — OpenAI 协议类型（完整保留 1,879 行）
- `src/llm_router_utils/sglang/srt/entrypoints/openai/serving_base.py` — OpenAIServingBase 基类
- `src/llm_router_utils/sglang/srt/entrypoints/openai/chat_encoding.py` — chat encoding spec 解析
- `src/llm_router_utils/sglang/srt/entrypoints/openai/encoding_dsv32.py` — DeepSeek v3.2 编码
- `src/llm_router_utils/sglang/srt/entrypoints/openai/encoding_dsv4.py` — DeepSeek v4 编码
- `src/llm_router_utils/sglang/srt/entrypoints/openai/sse_utils.py` — SSE 工具
- `src/llm_router_utils/sglang/srt/entrypoints/openai/usage_processor.py` — usage 计算
- `src/llm_router_utils/sglang/srt/entrypoints/openai/utils.py` — OpenAI serving 工具函数
- `src/llm_router_utils/sglang/srt/entrypoints/openai/harmony_utils.py` — harmony 解析工具

**裁剪后迁移（仅保留 `_process_messages` 链路用到的部分）：**
- `src/llm_router_utils/sglang/srt/entrypoints/openai/serving_chat.py` — 保留 `_process_messages` 及其调用链，删除 `_generate_stream_content` 等调度方法
- `src/llm_router_utils/sglang/srt/managers/tokenizer_manager.py` — 保留 `__init__`/`init_model_config`/`init_tokenizer_and_processor` 与属性
- `src/llm_router_utils/sglang/srt/managers/io_struct.py` — 保留 `GenerateReqInput`（如仍被引用）
- `src/llm_router_utils/sglang/srt/configs/model_config.py` — 保留 `ModelConfig.__init__`/`hf_config`/`is_multimodal`/`get_default_sampling_params`/`context_length`
- `src/llm_router_utils/sglang/srt/server_args.py` — 保留 `ServerArgs` dataclass 的 `_process_messages` 相关字段 + `PortArgs` 最小版
- `src/llm_router_utils/sglang/srt/environ.py` — 保留 `EnvField`/`ToolStrictLevel`/`Envs` 类相关字段，删末尾 cuda_coredump
- `src/llm_router_utils/sglang/srt/utils/common.py` — 保留 `ImageData`/`VideoData`/`read_system_prompt_from_file`/`find_local_repo_dir`
- `src/llm_router_utils/sglang/utils.py` — 保留 `convert_json_schema_to_str` 等被引用函数
- `src/llm_router_utils/sglang/srt/parser/template_manager.py` — 新增 `TokenizerLike` Protocol

**新建文件：**
- `src/llm_router_utils/__init__.py` — 顶层包导出
- `src/llm_router_utils/sglang/__init__.py` — 子包导出（仅 `__version__` 与便利 re-export）
- `src/llm_router_utils/sglang/_version.py` — 版本号
- `pyproject.toml` — 项目元数据与依赖
- `README.md` — 项目说明
- `LICENSE` — Apache 2.0
- `test/__init__.py`
- `test/unit/__init__.py`
- `test/unit/parser/__init__.py`
- `test/unit/function_call/__init__.py`
- `test/integration/__init__.py`
- `test/integration/test_process_messages.py` — `_process_messages` 端到端测试

**迁移的测试文件（从 `sglang_ksogit/test/registered/unit/`）：**
- `test/unit/parser/test_code_completion_parser.py`
- `test/unit/parser/test_conversation.py`
- `test/unit/parser/test_harmony_parser.py`
- `test/unit/parser/test_inkling_renderer.py`
- `test/unit/parser/test_jinja_template_utils.py`
- `test/unit/parser/test_reasoning_content_without_parser.py`
- `test/unit/parser/test_reasoning_parser.py`
- `test/unit/parser/test_template_manager.py`
- `test/unit/function_call/test_function_call_parser.py`
- `test/unit/function_call/test_hermes_detector.py`
- `test/unit/function_call/test_hunyuan_detector.py`
- `test/unit/function_call/test_json_schema_constraint.py`
- `test/unit/function_call/test_llama32_detector.py`
- `test/unit/function_call/test_minicpm5_detector.py`
- `test/unit/function_call/test_minimax_m3_detector.py`
- `test/unit/function_call/test_mistral_detector.py`
- `test/unit/function_call/test_normalize_json_schema_types.py`
- `test/unit/function_call/test_parallel_tool_calls.py`
- `test/unit/function_call/test_poolside_v1_detector.py`
- `test/unit/function_call/test_unknown_tool_name.py`

---

## Task 1: 创建项目骨架与 pyproject

**Files:**
- Create: `E:/githome-windows/llm_router_utils/pyproject.toml`
- Create: `E:/githome-windows/llm_router_utils/README.md`
- Create: `E:/githome-windows/llm_router_utils/LICENSE`
- Create: `E:/githome-windows/llm_router_utils/src/llm_router_utils/__init__.py`
- Create: `E:/githome-windows/llm_router_utils/src/llm_router_utils/sglang/__init__.py`
- Create: `E:/githome-windows/llm_router_utils/src/llm_router_utils/sglang/_version.py`

- [ ] **Step 1: 写 pyproject.toml**

Create `E:/githome-windows/llm_router_utils/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-router-utils"
version = "0.1.0"
description = "Lightweight extraction of sglang's reasoning/tool-call parsers and chat template rendering for router services."
requires-python = ">=3.10"
license = { file = "LICENSE" }
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: Apache Software License",
]
dependencies = [
  "pydantic",
  "jinja2",
  "partial_json_parser",
  "orjson",
  "typing_extensions",
  "tiktoken",
  "transformers",
  "openai",
  "mistral_common>=1.11.5",
  "packaging",
]

[project.optional-dependencies]
test = ["pytest", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: 写 README.md**

Create `E:/githome-windows/llm_router_utils/README.md`:

```markdown
# llm-router-utils

Lightweight extraction of sglang's reasoning parser, tool-call parser, and chat template rendering, for use in custom router services and lightweight LLM applications.

This library does **not** include any inference engine code. It only provides the "frontend" message processing pipeline: `OpenAIServingChat._process_messages` and its dependencies.

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
```

- [ ] **Step 3: 写 LICENSE**

Create `E:/githome-windows/llm_router_utils/LICENSE` with Apache 2.0 full text (copy from `sglang_ksogit/LICENSE`).

Run:
```bash
cp /e/githome-windows/sglang_ksogit/LICENSE /e/githome-windows/llm_router_utils/LICENSE
```

- [ ] **Step 4: 写顶层 __init__.py 与 _version.py**

Create `E:/githome-windows/llm_router_utils/src/llm_router_utils/__init__.py`:

```python
"""llm-router-utils: lightweight extraction of sglang's frontend message processing."""

from llm_router_utils.sglang._version import __version__

__all__ = ["__version__"]
```

Create `E:/githome-windows/llm_router_utils/src/llm_router_utils/sglang/__init__.py`:

```python
"""sglang submodule of llm-router-utils."""

from llm_router_utils.sglang._version import __version__

__all__ = ["__version__"]
```

Create `E:/githome-windows/llm_router_utils/src/llm_router_utils/sglang/_version.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 5: 验证包可被 ast.parse**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast; ast.parse(open('src/llm_router_utils/__init__.py').read()); ast.parse(open('src/llm_router_utils/sglang/__init__.py').read()); ast.parse(open('src/llm_router_utils/sglang/_version.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add pyproject.toml README.md LICENSE src/ && git commit -m "feat: scaffold llm-router-utils project structure"
```

---

## Task 2: 全量拷贝 sglang 源码到 src/llm_router_utils/sglang/

**Files:**
- Create: `src/llm_router_utils/sglang/` (整目录拷贝)

- [ ] **Step 1: 拷贝 sglang 源码**

Run:
```bash
cp -r /e/githome-windows/sglang_ksogit/python/sglang/* /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/
```

- [ ] **Step 2: 删除已存在的 __init__.py（用我们 Task 1 写的版本覆盖）**

Run:
```bash
cd /e/githome-windows/llm_router_utils && cat src/llm_router_utils/sglang/__init__.py | head -5
```

如果 `__init__.py` 是 sglang 原版（有 `_mps_stub` import），则用 Task 1 的版本覆盖。检查方法：第一行应该是 `"""sglang submodule of llm-router-utils."""`。如果是，跳过；如果不是，重写为 Task 1 Step 4 的内容。

- [ ] **Step 3: 验证所有 .py 文件语法正确**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast, glob; files = glob.glob('src/llm_router_utils/sglang/**/*.py', recursive=True); [ast.parse(open(f, encoding='utf-8').read()) for f in files]; print(f'Parsed {len(files)} files OK')"
```
Expected: `Parsed N files OK`（N 应在 500-1000 之间）

- [ ] **Step 4: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/ && git commit -m "feat: copy sglang source into src/llm_router_utils/sglang/"
```

---

## Task 3: 删除明显无关的顶层目录

**Files:**
- Delete: `src/llm_router_utils/sglang/benchmark/`
- Delete: `src/llm_router_utils/sglang/eval/`
- Delete: `src/llm_router_utils/sglang/cli/`
- Delete: `src/llm_router_utils/sglang/lang/`
- Delete: `src/llm_router_utils/sglang/multimodal_gen/`
- Delete: `src/llm_router_utils/sglang/jit_kernel/`
- Delete: `src/llm_router_utils/sglang/kernels/`
- Delete: `src/llm_router_utils/sglang/test/`（sglang 内部测试目录，我们用自己的 test/）
- Delete: 顶层推理启动脚本

- [ ] **Step 1: 删除顶层无关目录**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang && rm -rf benchmark eval cli lang multimodal_gen jit_kernel kernels test
```

- [ ] **Step 2: 删除顶层推理启动脚本**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang && rm -f auto_benchmark.py auto_benchmark_lib.py bench_offline_throughput.py bench_one_batch.py bench_one_batch_server.py bench_serving.py check_env.py compile_deep_gemm.py launch_server.py profiler.py utils.py global_config.py version.py _mps_stub.py _triton_stub.py
```

注意：`utils.py` 和 `version.py` 也删除——`utils.py` 会被我们在 Task 8 重新创建为裁剪版，`version.py` 由 `_version.py` 替代。`global_config.py` 是 lang DSL 用的，不需要。

- [ ] **Step 3: 验证语法仍正确**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast, glob; files = glob.glob('src/llm_router_utils/sglang/**/*.py', recursive=True); [ast.parse(open(f, encoding='utf-8').read()) for f in files]; print(f'Parsed {len(files)} files OK')"
```
Expected: `Parsed N files OK`，N 应该比 Task 2 Step 3 显著减少。

- [ ] **Step 4: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "chore: remove top-level inference/benchmark/cli modules"
```

---

## Task 4: 删除 srt/ 下的推理子树

**Files:**
- Delete: `src/llm_router_utils/sglang/srt/layers/`
- Delete: `src/llm_router_utils/sglang/srt/model_executor/`
- Delete: `src/llm_router_utils/sglang/srt/model_loader/`
- Delete: `src/llm_router_utils/sglang/srt/models/`
- Delete: `src/llm_router_utils/sglang/srt/sampling/`
- Delete: `src/llm_router_utils/sglang/srt/constrained/`
- Delete: `src/llm_router_utils/sglang/srt/speculative/`
- Delete: `src/llm_router_utils/sglang/srt/distributed/`
- Delete: `src/llm_router_utils/sglang/srt/disaggregation/`
- Delete: `src/llm_router_utils/sglang/srt/connector/`
- Delete: `src/llm_router_utils/sglang/srt/eplb/`
- Delete: `src/llm_router_utils/sglang/srt/elastic_ep/`
- Delete: `src/llm_router_utils/sglang/srt/dllm/`
- Delete: `src/llm_router_utils/sglang/srt/multiplex/`
- Delete: `src/llm_router_utils/sglang/srt/mem_cache/`
- Delete: `src/llm_router_utils/sglang/srt/kv_canary/`
- Delete: `src/llm_router_utils/sglang/srt/state_capturer/`
- Delete: `src/llm_router_utils/sglang/srt/checkpoint_engine/`
- Delete: `src/llm_router_utils/sglang/srt/lora/`
- Delete: `src/llm_router_utils/sglang/srt/observability/`
- Delete: `src/llm_router_utils/sglang/srt/compilation/`
- Delete: `src/llm_router_utils/sglang/srt/batch_invariant_ops/`
- Delete: `src/llm_router_utils/sglang/srt/batch_overlap/`
- Delete: `src/llm_router_utils/sglang/srt/platforms/`
- Delete: `src/llm_router_utils/sglang/srt/hardware_backend/`
- Delete: `src/llm_router_utils/sglang/srt/plugins/`
- Delete: `src/llm_router_utils/sglang/srt/session/`
- Delete: `src/llm_router_utils/sglang/srt/arg_groups/`
- Delete: `src/llm_router_utils/sglang/srt/debug_utils/`
- Delete: `src/llm_router_utils/sglang/srt/function_call/`（保留！— 不要删）
- Delete: `src/llm_router_utils/sglang/srt/parser/`（保留！— 不要删）
- Delete: `src/llm_router_utils/sglang/srt/tokenizer/`（保留！— 不要删）

**注意：** `parser/`、`function_call/`、`tokenizer/`、`managers/`、`entrypoints/`、`configs/`、`utils/` 这些目录**保留**，后续 Task 精细裁剪。本 Task 只删大块推理子树。

- [ ] **Step 1: 批量删除推理子树**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt && rm -rf layers model_executor model_loader models sampling constrained speculative distributed disaggregation connector eplb elastic_ep dllm multiplex mem_cache kv_canary state_capturer checkpoint_engine lora observability compilation batch_invariant_ops batch_overlap platforms hardware_backend plugins session arg_groups debug_utils
```

- [ ] **Step 2: 删除 managers 下的调度相关文件（保留 tokenizer_manager.py / io_struct.py / embed_types.py）**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/managers && rm -f scheduler.py scheduler_components scheduler_input_blocker.py tp_worker.py data_parallel_controller.py detokenizer_manager.py schedule_batch.py schedule_policy.py communicator.py hisparse_coordinator.py cache_controller.py overlap_utils.py prefill_delayer.py min_free_slots_delayer.py multi_tokenizer_mixin.py multimodal_processor.py mm_utils.py load_snapshot.py disagg_service.py utils.py async_dynamic_batch_tokenizer.py configure_logging.py tokenizer_control_mixin.py tokenizer_manager_score_mixin.py scheduler_pp_mixin.py
```

保留：`tokenizer_manager.py`、`io_struct.py`、`embed_types.py`（若存在）、`__init__.py`、`__pycache__/`（之后清）。

- [ ] **Step 3: 删除 entrypoints 下的推理服务（保留 openai/）**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints && rm -rf anthropic ollama search realtime && rm -f grpc_bridge.py grpc_server.py http_server.py http_server_engine.py engine.py engine_info_bootstrap_server.py engine_score_mixin.py EngineBase.py sidecar.py warmup.py v1_loads.py ssl_utils.py http_request_decompression.py request_headers.py tool.py context.py elastic_ep.py
```

保留：`openai/`、`harmony_utils.py`（待裁剪）、`__init__.py`。

- [ ] **Step 4: 删除 entrypoints/openai/ 下推理专用文件（保留 protocol/serving_base/serving_chat/chat_encoding/encoding_dsv32/encoding_dsv4/sse_utils/usage_processor/utils/__init__）**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints/openai && rm -rf realtime transcription_adapters && rm -f serving_classify.py serving_completions.py serving_embedding.py serving_rerank.py serving_responses.py serving_score.py serving_tokenize.py serving_transcription.py streaming_asr.py tool_server.py
```

保留：`protocol.py`、`serving_base.py`、`serving_chat.py`、`chat_encoding.py`、`encoding_dsv32.py`、`encoding_dsv4.py`、`sse_utils.py`、`usage_processor.py`、`utils.py`、`__init__.py`。

- [ ] **Step 5: 删除 srt/utils/ 下推理相关文件**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/utils && rm -f aio_rwlock.py aiter.py async_probe.py auth.py bench_utils.py cuda_ipc_transport_utils.py cudacore_pyspy_dump_utils.py custom_op.py device_timer.py field_validators.py gauge_histogram.py host_shared_memory.py http_middleware_patch.py json_response.py log_utils.py model_file_verifier.py msgspec_utils.py multi_stream_utils.py network.py numa_utils.py nvtx_pytorch_hooks.py nvtx_utils.py offloader.py patch_tokenizer.py patch_torch.py phase_checker.py poll_based_barrier.py profile_merger.py profile_utils.py request_logger.py rpd_utils.py runai_utils.py scheduler_status_logger.py slow_rank_detector.py stale_shm_cleanup.py tensor_bridge.py torch_memory_saver_adapter.py torch_npu_patch_utils.py video_decoder.py watchdog.py weight_checker.py weight_checker_comparator.py hf_transformers_patches.py hf_transformers_utils.py
```

保留：`common.py`（待裁剪）、`__init__.py`。

- [ ] **Step 6: 删除 configs/ 下推理专用文件（保留 model_config.py / __init__.py）**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/configs && ls | grep -v "^model_config\.py$" | grep -v "^__init__\.py$" | xargs rm -f
```

- [ ] **Step 7: 清理 __pycache__**

Run:
```bash
cd /e/githome-windows/llm_router_utils && find src/ -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; echo "done"
```

- [ ] **Step 8: 验证语法**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast, glob; files = glob.glob('src/llm_router_utils/sglang/**/*.py', recursive=True); [ast.parse(open(f, encoding='utf-8').read()) for f in files]; print(f'Parsed {len(files)} files OK')"
```
Expected: `Parsed N files OK`，N 应进一步大幅减少。

- [ ] **Step 9: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "chore: remove inference subtrees from srt/"
```

---

## Task 5: 全局替换 import 路径 sglang → llm_router_utils.sglang

**Files:**
- Modify: 所有 `src/llm_router_utils/sglang/**/*.py`

- [ ] **Step 1: 用 sed 全局替换 import 路径**

Run:
```bash
cd /e/githome-windows/llm_router_utils && find src/llm_router_utils/sglang -name "*.py" -exec sed -i 's/from sglang\./from llm_router_utils.sglang./g; s/import sglang\./import llm_router_utils.sglang./g' {} +
```

- [ ] **Step 2: 验证无残留 `from sglang` 或 `import sglang`**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -rn "^from sglang\|^import sglang\| from sglang\.\| import sglang\." src/llm_router_utils/sglang/ 2>&1 | head -20
```
Expected: 无输出（或仅匹配字符串字面量，不是 import 语句）。

- [ ] **Step 3: 验证语法**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast, glob; files = glob.glob('src/llm_router_utils/sglang/**/*.py', recursive=True); [ast.parse(open(f, encoding='utf-8').read()) for f in files]; print(f'Parsed {len(files)} files OK')"
```
Expected: `Parsed N files OK`

- [ ] **Step 4: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "refactor: rewrite import paths from sglang to llm_router_utils.sglang"
```

---

## Task 6: 裁剪 srt/utils/common.py 到最小集

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/utils/common.py`

- [ ] **Step 1: 用新文件覆盖 common.py，仅保留必需内容**

将 `src/llm_router_utils/sglang/srt/utils/common.py` 整体替换为以下内容：

```python
"""Lightweight utils extracted from sglang.srt.utils.common.

Only contains symbols referenced by the _process_messages call chain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

try:
    from huggingface_hub import try_to_load_from_cache  # type: ignore
except ImportError:  # pragma: no cover
    try_to_load_from_cache = None  # type: ignore


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    max_dynamic_patch: Optional[int] = None
    preprocess_kwargs: Optional[Dict] = None


@dataclass
class VideoData:
    url: str
    preprocess_kwargs: Optional[Dict] = None


def find_local_repo_dir(repo_id: str, revision: Optional[str] = None) -> Optional[str]:
    """Best-effort lookup of a local HF cache dir for ``repo_id``."""
    if try_to_load_from_cache is None:
        return None
    try:
        path = try_to_load_from_cache(repo_id, "config.json", revision=revision)
        if path is None or not os.path.exists(path):
            return None
        return str(Path(path).parent)
    except Exception:
        return None


def read_system_prompt_from_file(model_name: str) -> str:
    """Read SYSTEM_PROMPT.txt from the HuggingFace cache directory if present."""
    try:
        local_repo_dir = find_local_repo_dir(model_name)
        if local_repo_dir:
            system_prompt_file = os.path.join(local_repo_dir, "SYSTEM_PROMPT.txt")
            if os.path.exists(system_prompt_file):
                with open(system_prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
        return ""
    except Exception:
        return ""
```

- [ ] **Step 2: 重写 srt/utils/__init__.py 为显式导出**

将 `src/llm_router_utils/sglang/srt/utils/__init__.py` 替换为：

```python
from llm_router_utils.sglang.srt.utils.common import (
    ImageData,
    VideoData,
    find_local_repo_dir,
    read_system_prompt_from_file,
)

__all__ = [
    "ImageData",
    "VideoData",
    "find_local_repo_dir",
    "read_system_prompt_from_file",
]
```

- [ ] **Step 3: 验证 import 不触发 torch**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
import sys
# Ensure torch is not preloaded
for mod in list(sys.modules):
    if mod.startswith('torch'):
        del sys.modules[mod]
import llm_router_utils.sglang.srt.utils.common as c
assert 'torch' not in sys.modules, 'torch was imported!'
assert hasattr(c, 'ImageData')
assert hasattr(c, 'VideoData')
assert hasattr(c, 'read_system_prompt_from_file')
print('OK')
"
```
Expected: `OK`

注意：此步可能因 `huggingface_hub` 未安装而失败。如果失败，先 `pip install huggingface_hub` 或在 common.py 顶部加 `try/except`（已在代码中处理）。

- [ ] **Step 4: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/utils/ && git commit -m "refactor: slim srt/utils/common.py to ImageData/VideoData/read_system_prompt_from_file only"
```

---

## Task 7: 裁剪 srt/environ.py

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/environ.py`

- [ ] **Step 1: 删除末尾的 cuda_coredump import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -n "cuda_coredump" src/llm_router_utils/sglang/srt/environ.py
```

应看到类似 `1252:import sglang.srt.debug_utils.cuda_coredump`（已被替换为 `llm_router_utils.sglang.srt.debug_utils.cuda_coredump`）。

删除该行。用 Edit 工具或 sed：

```bash
cd /e/githome-windows/llm_router_utils && sed -i '/import llm_router_utils\.sglang\.srt\.debug_utils\.cuda_coredump/d' src/llm_router_utils/sglang/srt/environ.py
```

- [ ] **Step 2: 验证 environ.py 可独立 import（不依赖推理模块）**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
import sys
for mod in list(sys.modules):
    if mod.startswith('torch'):
        del sys.modules[mod]
from llm_router_utils.sglang.srt.environ import envs, ToolStrictLevel
assert 'torch' not in sys.modules, 'torch was imported!'
print('OK', envs, ToolStrictLevel)
"
```
Expected: `OK <Envs object> <enum 'ToolStrictLevel'>`

- [ ] **Step 3: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/environ.py && git commit -m "refactor: drop cuda_coredump import from environ.py"
```

---

## Task 8: 重写 sglang/utils.py（顶层 utils）

**Files:**
- Create: `src/llm_router_utils/sglang/utils.py`

原 `sglang/utils.py` 已在 Task 3 删除。但 `protocol.py` 顶部有 `from sglang.utils import convert_json_schema_to_str`（已被替换为 `from llm_router_utils.sglang.utils import convert_json_schema_to_str`）。需要重新创建一个仅含被引用函数的版本。

- [ ] **Step 1: 检查 protocol.py 引用了 utils.py 的哪些函数**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -n "from llm_router_utils.sglang.utils import\|from llm_router_utils\.sglang\.utils import" src/llm_router_utils/sglang/srt/entrypoints/openai/protocol.py
```

记录被引用的函数名（应该是 `convert_json_schema_to_str`，可能还有其他）。

- [ ] **Step 2: 在原 sglang 仓库中找到这些函数的实现**

Run:
```bash
grep -n "^def convert_json_schema_to_str\|^def LazyImport\|^class LazyImport" /e/githome-windows/sglang_ksogit/python/sglang/utils.py
```

- [ ] **Step 3: 创建新的 utils.py**

Create `src/llm_router_utils/sglang/utils.py`，包含 protocol.py 用到的所有函数。从原 `sglang_ksogit/python/sglang/utils.py` 拷贝 `convert_json_schema_to_str` 函数实现（与原版一字不差）。

示例（实际内容需从原文件拷贝）：

```python
"""Lightweight utils extracted from sglang.utils."""
from __future__ import annotations

import json
from typing import Any


def convert_json_schema_to_str(schema: Any) -> str:
    """Convert a JSON schema (dict or Pydantic model) to a string."""
    # ... copy implementation verbatim from sglang/utils.py ...
```

- [ ] **Step 4: 验证 protocol.py 可 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice, ChatCompletionRequest
print('OK', Tool, ToolChoice, ChatCompletionRequest)
"
```
Expected: `OK <class '...Tool'> <class '...ToolChoice'> <class '...ChatCompletionRequest'>`

如果报错，根据错误信息补全 utils.py 中缺失的函数。

- [ ] **Step 5: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/utils.py && git commit -m "feat: add slim sglang/utils.py with convert_json_schema_to_str"
```

---

## Task 9: 裁剪 server_args.py

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/server_args.py`

原 `server_args.py` 是 8,400 行，依赖 `arg_groups/`、`configs/` 等已删模块。需要大幅瘦身。

- [ ] **Step 1: 创建瘦身版 ServerArgs/PortArgs**

将 `src/llm_router_utils/sglang/srt/server_args.py` 整体替换为以下内容（保留原版 `ServerArgs` 类的字段定义，但仅保留 `_process_messages` 链路用到的字段，去掉所有 Arg/Annotated 元数据以避免依赖 arg_groups）：

```python
"""Lightweight ServerArgs/PortArgs extracted from sglang.srt.server_args.

Only fields referenced by the _process_messages call chain are kept.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ServerArgs:
    """Subset of sglang.srt.server_args.ServerArgs."""

    model_path: str
    tokenizer_path: Optional[str] = None
    chat_template: Optional[str] = None
    served_model_name: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    context_length: Optional[int] = None
    default_chat_template_kwargs: Optional[dict] = None
    enable_cache_report: bool = False
    incremental_streaming_output: bool = False
    stream_response_default_include_usage: bool = False
    allow_auto_truncate: bool = False
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path


@dataclasses.dataclass
class PortArgs:
    """Minimal PortArgs — fields kept for signature compatibility."""

    tokenizer_ipc_name: str = ""
    scheduler_input_ipc_name: str = ""
    detokenizer_ipc_name: str = ""
    nccl_port: int = 0
    rpc_ipc_name: str = ""
    metrics_ipc_name: str = ""
    tokenizer_worker_ipc_name: Optional[str] = None
    load_collector_ipc_name: str = ""
    instance_id: str = ""
```

- [ ] **Step 2: 验证 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.server_args import ServerArgs, PortArgs
sa = ServerArgs(model_path='Qwen/Qwen3-32B', tool_call_parser='qwen3_coder')
assert sa.tokenizer_path == 'Qwen/Qwen3-32B'
assert sa.served_model_name == 'Qwen/Qwen3-32B'
pa = PortArgs()
print('OK', sa, pa)
"
```
Expected: `OK ServerArgs(...) PortArgs(...)`

- [ ] **Step 3: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/server_args.py && git commit -m "refactor: slim server_args.py to _process_messages-relevant fields"
```

---

## Task 10: 裁剪 configs/model_config.py

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/configs/model_config.py`

原文件 2,059 行，依赖 torch、configs/linear_attn_model_registry 等已删模块。

- [ ] **Step 1: 创建瘦身版 ModelConfig**

将 `src/llm_router_utils/sglang/srt/configs/model_config.py` 整体替换为：

```python
"""Lightweight ModelConfig extracted from sglang.srt.configs.model_config.

Only fields/methods referenced by the _process_messages call chain are kept.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelConfig:
    """Minimal ModelConfig — loads HF config via transformers.AutoConfig."""

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        is_embedding: Optional[bool] = None,
        enable_multimodal: Optional[bool] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        override_config_file: Optional[str] = None,
        sampling_defaults: str = "openai",
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self.quantization = quantization
        self.sampling_defaults = sampling_defaults
        self.context_length = context_length
        self.is_embedding = is_embedding if is_embedding is not None else False

        # Load HF config
        from transformers import AutoConfig
        try:
            self.hf_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
        except Exception as e:
            logger.warning(f"Failed to load HF config for {model_path}: {e}")
            self.hf_config = None

        # Determine multimodal
        self.is_multimodal = False
        if enable_multimodal and self.hf_config is not None:
            # Heuristic: check for common multimodal model type markers
            model_type = getattr(self.hf_config, "model_type", "")
            multimodal_markers = ["vlm", "vl", "vision", "multimodal", "image"]
            self.is_multimodal = any(m in model_type.lower() for m in multimodal_markers)

        # Context length from config if not specified
        if self.context_length is None and self.hf_config is not None:
            self.context_length = getattr(self.hf_config, "max_position_embeddings", None)

    def get_default_sampling_params(self) -> dict:
        """Return default sampling params. Currently returns empty dict.

        In original sglang this reads from sampling_defaults; here we keep
        a minimal stub since _process_messages only needs the method to exist.
        """
        return {}
```

- [ ] **Step 2: 删除 configs/ 下其他文件（若 Task 4 未删完）**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/configs && ls | grep -v "^model_config\.py$" | grep -v "^__init__\.py$" | xargs rm -f 2>/dev/null; echo done
```

- [ ] **Step 3: 检查 configs/__init__.py**

Run:
```bash
cat /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/configs/__init__.py 2>/dev/null || echo "no init"
```

如果 `__init__.py` 引用了已删模块，清空它：

```bash
echo "" > /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/configs/__init__.py
```

- [ ] **Step 4: 验证 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.configs.model_config import ModelConfig
print('OK', ModelConfig)
"
```
Expected: `OK <class '...ModelConfig'>`

- [ ] **Step 5: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/configs/ && git commit -m "refactor: slim configs/model_config.py to minimal ModelConfig"
```

---

## Task 11: 裁剪 managers/tokenizer_manager.py

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/managers/tokenizer_manager.py`

原文件 3,295 行，依赖 34 个 sglang 内部模块。需要大幅瘦身。

- [ ] **Step 1: 创建瘦身版 TokenizerManager**

将 `src/llm_router_utils/sglang/srt/managers/tokenizer_manager.py` 整体替换为：

```python
"""Lightweight TokenizerManager extracted from sglang.srt.managers.tokenizer_manager.

Only the initializer and the attributes referenced by the _process_messages
call chain are kept. All scheduling/dispatch methods are removed.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from llm_router_utils.sglang.srt.configs.model_config import ModelConfig
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


class TokenizerManager:
    """Minimal TokenizerManager that loads a tokenizer + model config.

    Mirrors the public surface used by OpenAIServingChat._process_messages:
    exposes ``.tokenizer``, ``.processor``, ``.model_config``, ``.server_args``.
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args
        self.model_config: Optional[ModelConfig] = None
        self.tokenizer: Any = None
        self.processor: Any = None

        self.init_model_config()
        if not server_args.skip_tokenizer_init:
            self.init_tokenizer_and_processor()

    def init_model_config(self) -> None:
        self.model_config = ModelConfig(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            context_length=self.server_args.context_length,
        )

    def init_tokenizer_and_processor(self) -> None:
        from transformers import AutoTokenizer

        tokenizer_path = self.server_args.tokenizer_path or self.server_args.model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.server_args.trust_remote_code,
                use_fast=self.server_args.tokenizer_mode == "auto",
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            self.tokenizer = None

        # Try to load multimodal processor (optional)
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.server_args.trust_remote_code,
            )
        except Exception:
            self.processor = None
```

- [ ] **Step 2: 检查 managers/__init__.py**

Run:
```bash
cat /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/managers/__init__.py 2>/dev/null | head -20
```

如果引用了已删模块，清空：

```bash
echo "" > /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/managers/__init__.py
```

- [ ] **Step 3: 检查 managers/io_struct.py 是否仍被引用**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -rn "from llm_router_utils.sglang.srt.managers.io_struct import\|from llm_router_utils\.sglang\.srt\.managers\.io_struct import" src/llm_router_utils/sglang/ 2>&1 | head -10
```

- [ ] **Step 4: 如果 io_struct 仍被引用，创建瘦身版**

如果 Step 3 有匹配（例如 `serving_chat.py` 引用 `GenerateReqInput`），创建瘦身版 `managers/io_struct.py`：

```python
"""Lightweight io_struct — only GenerateReqInput kept if still referenced."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class GenerateReqInput:
    """Minimal GenerateReqInput — only fields referenced by retained code."""
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    input_ids: Optional[List[int]] = None
    text: Optional[str] = None
    rid: Optional[str] = None
    return_logprob: bool = False
    stream: bool = False
    logprob_start_len: int = 0
    top_logprobs_num: int = 0
    token_ids_logprob: Optional[List[int]] = None
    lora_path: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    image_data: Optional[Any] = None
    video_data: Optional[Any] = None
    audio_data: Optional[Any] = None
    modalities: Optional[List[str]] = None
    stop: Optional[Union[str, List[str]]] = None
```

如果 Step 3 无匹配，直接删除 io_struct.py：

```bash
rm /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/managers/io_struct.py
```

- [ ] **Step 5: 删除 managers/ 下其他残留文件**

Run:
```bash
cd /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/managers && ls | grep -v "^__init__\.py$" | grep -v "^tokenizer_manager\.py$" | grep -v "^io_struct\.py$" | grep -v "^embed_types\.py$" | xargs rm -f 2>/dev/null; echo done
```

- [ ] **Step 6: 验证 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
print('OK', TokenizerManager)
"
```
Expected: `OK <class '...TokenizerManager'>`

- [ ] **Step 7: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/managers/ && git commit -m "refactor: slim managers/tokenizer_manager.py to initializer + tokenizer loading"
```

---

## Task 12: 给 template_manager.py 添加 TokenizerLike Protocol

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/parser/template_manager.py`

- [ ] **Step 1: 替换 TokenizerManager 类型注解为 TokenizerLike Protocol**

在 `template_manager.py` 顶部（import 之后、`TemplateManager` 类之前）添加 Protocol 定义：

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerLike(Protocol):
    """Protocol for objects passed to TemplateManager methods.

    Any object with ``.tokenizer``, ``.processor``, ``.model_config``,
    ``.server_args`` attributes is acceptable. ``TokenizerManager`` is the
    default implementation.
    """

    @property
    def tokenizer(self) -> Any: ...
    @property
    def processor(self) -> Any: ...
    @property
    def model_config(self) -> Any: ...
    @property
    def server_args(self) -> Any: ...
```

并在文件顶部添加 `from typing import Any` 导入（若已有则跳过）。

- [ ] **Step 2: 替换所有 `tokenizer_manager: TokenizerManager` 类型注解**

Run:
```bash
cd /e/githome-windows/llm_router_utils && sed -i 's/tokenizer_manager: TokenizerManager/tokenizer_manager: TokenizerLike/g' src/llm_router_utils/sglang/srt/parser/template_manager.py
```

- [ ] **Step 3: 替换 import**

将 `template_manager.py` 中：
```python
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
```
替换为：
```python
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: F401 — re-exported for backward compat
```

（保留 import 以便 `TokenizerManager` 仍可被外部代码引用，但类型注解用 Protocol。）

- [ ] **Step 4: 验证 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager, TokenizerLike
print('OK', TemplateManager, TokenizerLike)
"
```
Expected: `OK <class '...TemplateManager'> <class '...TokenizerLike'>`

- [ ] **Step 5: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/parser/template_manager.py && git commit -m "refactor: add TokenizerLike Protocol in template_manager.py"
```

---

## Task 13: 裁剪 serving_chat.py 删除推理调度方法

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/entrypoints/openai/serving_chat.py`

- [ ] **Step 1: 识别要删除的方法**

原文件 2,252 行，需删除以下方法（保留 `_process_messages` 及其调用链）：

要删除的方法：
- `_generate_stream_content`
- `_handle_streaming_request`
- `_handle_non_streaming_request`
- `_generate_chat_stream`
- `_process_tool_call_stream`
- `_process_tool_call_non_stream`
- `handle_request`（若仅调度用）
- 任何引用 `create_abort_task` / `generate_request` / `_dispatch_to_scheduler` 的方法

- [ ] **Step 2: 删除调度方法**

用 Edit 工具按方法逐个删除。每个方法的范围从 `def method_name(...):` 到下一个 `def ` 或文件末尾。

具体方法：用 Grep 找到每个方法的行号，然后用 Edit 删除该范围。

```bash
cd /e/githome-windows/llm_router_utils && grep -n "^    async def \|^    def " src/llm_router_utils/sglang/srt/entrypoints/openai/serving_chat.py
```

根据列出的方法，删除所有不在 `_process_messages` 调用链上的方法。保留：
- `__init__`
- `_resolve_chat_encoding_spec`
- `_encode_messages`
- `_patch_reasoning_skip_special_tokens`
- `_get_reasoning_from_request`
- `_apply_jinja_template`
- `_apply_conversation_template`
- `_process_messages`

- [ ] **Step 3: 清理已删方法的 import**

删除 serving_chat.py 顶部不再需要的 import，如：
- `from llm_router_utils.sglang.srt.managers.io_struct import GenerateReqInput`（如果 GenerateReqInput 仅在被删方法中使用）
- 其他仅被已删方法引用的 import

- [ ] **Step 4: 验证语法**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast; ast.parse(open('src/llm_router_utils/sglang/srt/entrypoints/openai/serving_chat.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 5: 验证 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
print('OK', OpenAIServingChat)
assert hasattr(OpenAIServingChat, '_process_messages')
assert not hasattr(OpenAIServingChat, '_generate_stream_content'), 'scheduling method should be removed'
print('all checks passed')
"
```
Expected: `OK <class '...OpenAIServingChat'>` + `all checks passed`

- [ ] **Step 6: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add src/llm_router_utils/sglang/srt/entrypoints/openai/serving_chat.py && git commit -m "refactor: drop scheduling methods from serving_chat.py, keep _process_messages chain"
```

---

## Task 14: 清理 entrypoints/ 残留与 harmony_utils.py

**Files:**
- Modify: `src/llm_router_utils/sglang/srt/entrypoints/harmony_utils.py`
- Modify: `src/llm_router_utils/sglang/srt/entrypoints/__init__.py`
- Modify: `src/llm_router_utils/sglang/srt/entrypoints/openai/__init__.py`

- [ ] **Step 1: 检查 harmony_utils.py 是否被 _process_messages 链路引用**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -rn "harmony_utils" src/llm_router_utils/sglang/srt/entrypoints/openai/ src/llm_router_utils/sglang/srt/parser/ src/llm_router_utils/sglang/srt/function_call/
```

如果无匹配，删除：
```bash
rm /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints/harmony_utils.py
```

如果有匹配，保留但检查其 import 是否有已删模块。

- [ ] **Step 2: 清空 entrypoints/__init__.py 与 entrypoints/openai/__init__.py（若引用已删模块）**

Run:
```bash
cat /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints/__init__.py 2>/dev/null | head -5
cat /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints/openai/__init__.py 2>/dev/null | head -5
```

如果引用了已删模块或非空，清空：
```bash
echo "" > /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints/__init__.py
echo "" > /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/srt/entrypoints/openai/__init__.py
```

- [ ] **Step 3: 验证语法**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast, glob; files = glob.glob('src/llm_router_utils/sglang/srt/entrypoints/**/*.py', recursive=True); [ast.parse(open(f, encoding='utf-8').read()) for f in files]; print(f'Parsed {len(files)} files OK')"
```
Expected: `Parsed N files OK`

- [ ] **Step 4: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "chore: clean up entrypoints residuals"
```

---

## Task 15: 修复全链路 import 错误

**Files:**
- Modify: 多个文件（根据实际报错）

- [ ] **Step 1: 尝试 import OpenAIServingChat**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
print('OK')
" 2>&1 | head -50
```

- [ ] **Step 2: 根据报错逐个修复**

常见问题：
1. `ImportError: No module named 'llm_router_utils.sglang.srt.xxx'` — 该模块被引用但已删，需在引用处删除该 import 或创建瘦身版
2. `AttributeError: module 'xxx' has no attribute 'yyy'` — 引用的符号在瘦身版中不存在，需补回该符号或修改引用
3. `ImportError: cannot import name 'xxx' from 'yyy'` — 同上

每个错误用 Edit 工具修复，修复后重新跑 Step 1。

- [ ] **Step 3: 尝试 import parser 与 function_call**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.parser.reasoning_parser import ReasoningParser
from llm_router_utils.sglang.function_call.function_call_parser import FunctionCallParser
from llm_router_utils.sglang.parser.conversation import Conversation, generate_chat_conv
from llm_router_utils.sglang.parser.template_manager import TemplateManager
print('OK')
" 2>&1 | head -50
```

- [ ] **Step 4: 修复 parser/function_call 链路的 import 错误**

同 Step 2 方法。

- [ ] **Step 5: 验证所有保留模块可 import**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
import importlib
modules = [
    'llm_router_utils.sglang',
    'llm_router_utils.sglang.srt',
    'llm_router_utils.sglang.srt.parser.conversation',
    'llm_router_utils.sglang.srt.parser.reasoning_parser',
    'llm_router_utils.sglang.srt.parser.template_manager',
    'llm_router_utils.sglang.srt.parser.template_detection',
    'llm_router_utils.sglang.srt.parser.harmony_parser',
    'llm_router_utils.sglang.srt.parser.inkling_renderer',
    'llm_router_utils.sglang.srt.parser.inkling_tokenizer',
    'llm_router_utils.sglang.srt.parser.jinja_template_utils',
    'llm_router_utils.sglang.srt.parser.code_completion_parser',
    'llm_router_utils.sglang.srt.function_call.function_call_parser',
    'llm_router_utils.sglang.srt.function_call.core_types',
    'llm_router_utils.sglang.srt.function_call.base_format_detector',
    'llm_router_utils.sglang.srt.function_call.utils',
    'llm_router_utils.sglang.srt.function_call.json_array_parser',
    'llm_router_utils.sglang.srt.entrypoints.openai.protocol',
    'llm_router_utils.sglang.srt.entrypoints.openai.serving_base',
    'llm_router_utils.sglang.srt.entrypoints.openai.serving_chat',
    'llm_router_utils.sglang.srt.managers.tokenizer_manager',
    'llm_router_utils.sglang.srt.configs.model_config',
    'llm_router_utils.sglang.srt.server_args',
    'llm_router_utils.sglang.srt.environ',
    'llm_router_utils.sglang.srt.utils',
    'llm_router_utils.sglang.utils',
]
for m in modules:
    try:
        importlib.import_module(m)
        print(f'OK {m}')
    except Exception as e:
        print(f'FAIL {m}: {e}')
" 2>&1
```

Expected: 所有模块 `OK`。如有 `FAIL`，回到 Step 2 修复。

- [ ] **Step 6: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "fix: resolve import errors across _process_messages call chain"
```

---

## Task 16: 迁移单元测试

**Files:**
- Create: `test/__init__.py`
- Create: `test/unit/__init__.py`
- Create: `test/unit/parser/__init__.py`
- Create: `test/unit/function_call/__init__.py`
- Copy: `test/unit/parser/*.py` from `sglang_ksogit/test/registered/unit/parser/`
- Copy: `test/unit/function_call/*.py` from `sglang_ksogit/test/registered/unit/function_call/`

- [ ] **Step 1: 创建 test 包 __init__.py**

Create `E:/githome-windows/llm_router_utils/test/__init__.py` (empty):
```bash
mkdir -p /e/githome-windows/llm_router_utils/test/unit/parser /e/githome-windows/llm_router_utils/test/unit/function_call /e/githome-windows/llm_router_utils/test/integration
touch /e/githome-windows/llm_router_utils/test/__init__.py /e/githome-windows/llm_router_utils/test/unit/__init__.py /e/githome-windows/llm_router_utils/test/unit/parser/__init__.py /e/githome-windows/llm_router_utils/test/unit/function_call/__init__.py /e/githome-windows/llm_router_utils/test/integration/__init__.py
```

- [ ] **Step 2: 拷贝测试文件**

Run:
```bash
cp /e/githome-windows/sglang_ksogit/test/registered/unit/parser/*.py /e/githome-windows/llm_router_utils/test/unit/parser/
cp /e/githome-windows/sglang_ksogit/test/registered/unit/function_call/*.py /e/githome-windows/llm_router_utils/test/unit/function_call/
```

- [ ] **Step 3: 全局替换测试中的 import 路径**

Run:
```bash
cd /e/githome-windows/llm_router_utils && find test/ -name "*.py" -exec sed -i 's/from sglang\./from llm_router_utils.sglang./g; s/import sglang\./import llm_router_utils.sglang./g' {} +
```

- [ ] **Step 4: 删除测试中对 sglang.test 的依赖**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -rln "from llm_router_utils.sglang.test\|from llm_router_utils\.sglang\.test" test/ 2>&1
```

对每个匹配文件，用 Edit 工具：
1. 删除 `from llm_router_utils.sglang.test.ci.ci_register import register_cpu_ci` 行
2. 删除文件中所有 `register_cpu_ci(...)` 调用行
3. 将 `from llm_router_utils.sglang.test.test_utils import CustomTestCase` 改为 `import unittest`，并将测试类父类 `CustomTestCase` 改为 `unittest.TestCase`

- [ ] **Step 5: 验证测试文件语法**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "import ast, glob; files = glob.glob('test/**/*.py', recursive=True); [ast.parse(open(f, encoding='utf-8').read()) for f in files]; print(f'Parsed {len(files)} files OK')"
```
Expected: `Parsed N files OK`

- [ ] **Step 6: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add test/ && git commit -m "test: migrate parser/function_call unit tests from sglang"
```

---

## Task 17: 运行单元测试并修复回归

**Files:**
- Modify: 多个文件（根据测试失败原因）

- [ ] **Step 1: 运行 parser 单元测试**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -m pytest test/unit/parser/ -x --tb=short 2>&1 | tail -50
```

- [ ] **Step 2: 修复失败的测试**

常见失败原因：
1. `ImportError` — 测试引用了未迁移的 helper
2. `AttributeError` — 引用的属性在瘦身版中不存在
3. 测试 fixture 用了 sglang 内部启动逻辑

对每个失败：
- 检查测试代码与被测代码
- 如果是 import 缺失，补回缺失符号或修改测试
- 如果是逻辑差异，**优先怀疑迁移 bug**，回原 sglang 对比

- [ ] **Step 3: 运行 function_call 单元测试**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -m pytest test/unit/function_call/ -x --tb=short 2>&1 | tail -50
```

- [ ] **Step 4: 修复失败**

同 Step 2 方法。

- [ ] **Step 5: 运行全部单元测试**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -m pytest test/unit/ --tb=short 2>&1 | tail -30
```
Expected: 所有测试通过（或仅有与模型下载相关的 skip）。

- [ ] **Step 6: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "fix: resolve unit test regressions after migration"
```

---

## Task 18: 写集成测试 test_process_messages.py

**Files:**
- Create: `test/integration/test_process_messages.py`

- [ ] **Step 1: 写集成测试骨架**

Create `E:/githome-windows/llm_router_utils/test/integration/test_process_messages.py`:

```python
"""Integration tests for OpenAIServingChat._process_messages.

These tests verify that _process_messages produces correct MessageProcessingResult
across typical scenarios: plain text, with tools, with reasoning, jinja vs conversation
template, and input_ids shortcut.
"""
import json
import os
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from llm_router_utils.sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionMessage,
    Tool,
    ToolChoice,
)
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs


def _make_fake_tokenizer(chat_template: Optional[str] = None) -> MagicMock:
    """Create a fake tokenizer that mimics HF tokenizer interface."""
    tok = MagicMock()
    tok.bos_token_id = None
    tok.eos_token_id = 1
    tok.encode.return_value = [1, 2, 3]
    tok.decode.return_value = "decoded"
    tok.apply_chat_template.return_value = [1, 2, 3]
    if chat_template is not None:
        tok.chat_template = chat_template
    else:
        tok.chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
    return tok


def _make_fake_tokenizer_manager(
    server_args: ServerArgs,
    chat_template: Optional[str] = None,
) -> TokenizerManager:
    """Build a TokenizerManager with mocked tokenizer (no model download)."""
    # Bypass real __init__ to avoid network/disk access
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = server_args
    tm.port_args = PortArgs()
    tm.tokenizer = _make_fake_tokenizer(chat_template)
    tm.processor = None
    tm.model_config = MagicMock()
    tm.model_config.hf_config = None
    tm.model_config.is_multimodal = False
    tm.model_config.get_default_sampling_params.return_value = {}
    return tm


class TestProcessMessagesPlainText(unittest.TestCase):
    """Scenario: plain text chat, no tools, no reasoning."""

    def test_plain_text_returns_prompt(self):
        server_args = ServerArgs(model_path="fake-model")
        tm = _make_fake_tokenizer_manager(server_args)
        tpl = TemplateManager()
        tpl.initialize_templates(
            tokenizer_manager=tm,
            model_path="fake-model",
            chat_template=None,
        )
        serving = OpenAIServingChat(tm, tpl)
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[
                ChatCompletionMessage(role="user", content="Hello"),
                ChatCompletionMessage(role="assistant", content="Hi there"),
            ],
        )
        result = serving._process_messages(request, is_multimodal=False)
        self.assertIsNotNone(result)
        self.assertIn("user", result.prompt)
        self.assertIn("Hello", result.prompt)


class TestProcessMessagesWithTools(unittest.TestCase):
    """Scenario: chat with tools — tool_call_constraint should be set."""

    def test_tools_set_tool_call_constraint(self):
        server_args = ServerArgs(
            model_path="fake-model",
            tool_call_parser="hermes",
        )
        tm = _make_fake_tokenizer_manager(server_args)
        tpl = TemplateManager()
        tpl.initialize_templates(
            tokenizer_manager=tm,
            model_path="fake-model",
            chat_template=None,
        )
        serving = OpenAIServingChat(tm, tpl)

        tool = Tool(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        )
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[ChatCompletionMessage(role="user", content="What's the weather?")],
            tools=[tool],
            tool_choice="auto",
        )
        result = serving._process_messages(request, is_multimodal=False)
        self.assertIsNotNone(result.tool_call_constraint)


class TestProcessMessagesInputIds(unittest.TestCase):
    """Scenario: input_ids provided — should skip template tokenization."""

    def test_input_ids_short_circuits(self):
        server_args = ServerArgs(model_path="fake-model")
        tm = _make_fake_tokenizer_manager(server_args)
        tpl = TemplateManager()
        tpl.initialize_templates(
            tokenizer_manager=tm,
            model_path="fake-model",
            chat_template=None,
        )
        serving = OpenAIServingChat(tm, tpl)

        request = ChatCompletionRequest(
            model="fake-model",
            messages=[ChatCompletionMessage(role="user", content="ignored")],
            input_ids=[1, 2, 3, 4],
        )
        result = serving._process_messages(request, is_multimodal=False)
        self.assertEqual(result.prompt, "")
        self.assertEqual(result.prompt_ids, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行集成测试**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -m pytest test/integration/test_process_messages.py -x --tb=short 2>&1 | tail -50
```

- [ ] **Step 3: 修复失败**

如果测试失败，根据失败原因修复：
1. 如果 `ChatCompletionRequest` 字段与原版不一致（如 `input_ids` 字段不存在），回查原版 protocol.py 是否完整保留
2. 如果 `_process_messages` 调用链中某方法被错误删除，回 Task 13 补回
3. 如果 fixture 与 `_process_messages` 实际期望不符，调整 fixture

- [ ] **Step 4: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add test/integration/ && git commit -m "test: add _process_messages integration tests"
```

---

## Task 19: 最终验证与 pyproject 依赖收敛

**Files:**
- Modify: `pyproject.toml`（根据实际依赖收敛）

- [ ] **Step 1: 检查实际 import 的第三方包**

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -rh "^import \|^from " src/llm_router_utils/sglang/ | grep -v "^from llm_router_utils\|^import llm_router_utils\|^from \.\|^from typing\|^from __future__" | sort -u | head -40
```

记录实际用到的第三方包。

- [ ] **Step 2: 移除未使用的依赖**

对照 pyproject.toml 的 dependencies，移除实际未引用的包。常见可移除候选：
- `outlines` — 若 function_call/utils.py 不引用
- `xgrammar` — 若 function_call 链路不引用
- `llguidance` — 若 function_call 链路不引用

Run:
```bash
cd /e/githome-windows/llm_router_utils && grep -rn "import outlines\|import xgrammar\|import llguidance" src/llm_router_utils/sglang/ 2>&1 | head -10
```

如果无匹配，从 pyproject.toml 移除。

- [ ] **Step 3: 验证安装可执行**

Run:
```bash
cd /e/githome-windows/llm_router_utils && pip install -e . 2>&1 | tail -10
```
Expected: 安装成功。

- [ ] **Step 4: 验证全部测试通过**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -m pytest test/ --tb=short 2>&1 | tail -30
```
Expected: 所有测试通过或仅有与模型下载相关的 skip。

- [ ] **Step 5: 验证 torch 未被引入**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
import sys
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
assert 'torch' not in sys.modules, 'torch should not be imported!'
print('OK — torch not loaded')
"
```
Expected: `OK — torch not loaded`

- [ ] **Step 6: Commit**

```bash
cd /e/githome-windows/llm_router_utils && git add -A && git commit -m "chore: converge pyproject dependencies, verify torch-free import"
```

- [ ] **Step 7: 最终验证 — 调用形态示例**

Run:
```bash
cd /e/githome-windows/llm_router_utils && python -c "
from llm_router_utils.sglang.srt.configs.model_config import ModelConfig
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from llm_router_utils.sglang.srt.server_args import ServerArgs, PortArgs
print('All imports OK — call-shape verified')
"
```
Expected: `All imports OK — call-shape verified`

---

## Self-Review

**1. Spec coverage:**

- ✅ 仓库名 `llm-router-utils` + 子模块 `sglang` — Task 1, 2
- ✅ src layout — Task 1
- ✅ 全量拷贝 + 删减 — Task 2, 3, 4
- ✅ 反向依赖闭包裁剪 — Task 6-13
- ✅ TokenizerLike Protocol — Task 12
- ✅ serving_chat 保留 `_process_messages` 链 — Task 13
- ✅ TokenizerManager 瘦身 — Task 11
- ✅ ServerArgs/ModelConfig 瘦身 — Task 9, 10
- ✅ environ.py 删 cuda_coredump — Task 7
- ✅ utils/common.py 瘦身 — Task 6
- ✅ utils.py 重写 — Task 8
- ✅ import 路径替换 — Task 5
- ✅ 测试迁移 — Task 16, 17
- ✅ 集成测试 — Task 18
- ✅ 依赖收敛 — Task 19
- ✅ torch 不被引入 — Task 19 Step 5

**2. Placeholder scan:** 已检查，无 TBD/TODO/placeholder。Task 8 Step 2-3 要求从原文件拷贝函数实现，这是必要的人工步骤，已说明来源。

**3. Type consistency:**
- `ServerArgs` 字段在 Task 9 与 Task 18 一致
- `PortArgs` 在 Task 9 与 Task 11 一致
- `TokenizerManager` 属性（`.tokenizer`/`.processor`/`.model_config`/`.server_args`）在 Task 11、Task 12 Protocol、Task 18 fixture 一致
- `ModelConfig` 方法 `get_default_sampling_params` 在 Task 10 与 Task 18 一致

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-12-llm-router-utils-extraction.md`. Executing inline with `superpowers:executing-plans` per user request.
