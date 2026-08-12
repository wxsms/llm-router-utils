# llm-router-utils 设计文档

**日期：** 2026-08-12
**状态：** 设计已确认，待实现
**作者：** brainstorming session

---

## 1. 仓库名与定位

**仓库名：** `llm-router-utils`

**定位：** 一个面向自定义 router 服务/轻量级 LLM 应用的 Python 库，从 sglang 抽取"前端处理"能力（chat template 渲染、reasoning parser、tool call parser、OpenAI 请求消息处理流水），丢弃所有推理调度相关代码。后续可在 `llm_router_utils/` 下增加 `vllm/` 等其它引擎的适配子模块。

**包路径：** `from llm_router_utils.sglang.parser.reasoning_parser import ReasoningParser` 等。`sglang` 作为包路径的一段保留，便于未来按引擎区分子模块。

**仓库根：** `E:\githome-windows\llm_router_utils\`（与 `sglang_ksogit\` 同级）

**与上游 sglang 的关系：** 一次性迁移，不保留 git 历史，不维护自动同步脚本（后续如有需要再补）。

---

## 2. 目录结构

仓库布局（src layout）：

```
llm_router_utils/
├── .git/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── llm_router_utils/
│       ├── __init__.py            # 顶层导出
│       ├── sglang/                # 从 sglang 抽取的子模块
│       │   ├── __init__.py
│       │   ├── _version.py
│       │   ├── srt/
│       │   │   ├── __init__.py
│       │   │   ├── environ.py                 # 瘦身后（去掉 cuda_coredump 等推理相关）
│       │   │   ├── server_args.py             # 瘦身后（保留 dataclass + __post_init__ + from_cli 的相关字段）
│       │   │   ├── constants.py               # 若 _process_messages 链路用到则保留
│       │   │   ├── configs/
│       │   │   │   └── model_config.py        # 瘦身后（保留 hf_config / is_multimodal / get_default_sampling_params）
│       │   │   ├── parser/                    # 完整迁移（9 个文件，5,229 行）
│       │   │   │   ├── __init__.py
│       │   │   │   ├── code_completion_parser.py
│       │   │   │   ├── conversation.py
│       │   │   │   ├── harmony_parser.py
│       │   │   │   ├── inkling_renderer.py
│       │   │   │   ├── inkling_tokenizer.py
│       │   │   │   ├── jinja_template_utils.py
│       │   │   │   ├── reasoning_parser.py
│       │   │   │   ├── template_detection.py
│       │   │   │   └── template_manager.py    # TokenizerManager 类型注解改 Protocol
│       │   │   ├── function_call/             # 完整迁移（34 个文件，10,836 行）
│       │   │   │   ├── __init__.py
│       │   │   │   ├── ... (全部 detector)
│       │   │   │   ├── function_call_parser.py
│       │   │   │   ├── core_types.py
│       │   │   │   ├── base_format_detector.py
│       │   │   │   ├── json_array_parser.py
│       │   │   │   └── utils.py
│       │   │   ├── tokenizer/
│       │   │   │   └── tiktoken_tokenizer.py  # 完整迁移
│       │   │   ├── managers/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── tokenizer_manager.py    # 瘦身版（保留 __init__/init_model_config/init_tokenizer_and_processor）
│       │   │   │   ├── io_struct.py            # 瘦身版（保留 GenerateReqInput/EmbeddingReqInput 等被引用的）
│       │   │   │   ├── embed_types.py          # 若被引用则保留
│       │   │   │   └── tokenizer_control_mixin.py  # 若 TokenizerManager 继承链用到则保留瘦身版
│       │   │   ├── entrypoints/
│       │   │   │   └── openai/
│       │   │   │       ├── __init__.py
│       │   │   │       ├── protocol.py         # 完整保留（1,879 行）
│       │   │   │       ├── serving_base.py     # 完整迁移（306 行）
│       │   │   │       ├── serving_chat.py     # 裁剪版（保留 _process_messages 及其调用链，删 _generate_stream_content 等）
│       │   │   │       ├── chat_encoding.py
│       │   │   │       ├── encoding_dsv32.py
│       │   │   │       ├── encoding_dsv4.py
│       │   │   │       ├── sse_utils.py
│       │   │   │       ├── usage_processor.py
│       │   │   │       ├── utils.py
│       │   │   │       └── harmony_utils.py    # 若 _process_messages 链路用到则保留
│       │   │   └── utils/                      # 瘦身版（仅保留 _process_messages 链路用到的工具函数）
│       │   │       ├── __init__.py
│       │   │       └── common.py              # 由原 srt/utils/common.py 裁剪
│       │   └── utils.py                       # 由原 sglang/utils.py 裁剪（保留 convert_json_schema_to_str 等）
├── test/
│   ├── __init__.py
│   ├── unit/
│   │   ├── parser/                # 从 test/registered/unit/parser/ 迁移
│   │   │   ├── test_code_completion_parser.py
│   │   │   ├── test_conversation.py
│   │   │   ├── test_harmony_parser.py
│   │   │   ├── test_inkling_renderer.py
│   │   │   ├── test_jinja_template_utils.py
│   │   │   ├── test_reasoning_content_without_parser.py
│   │   │   ├── test_reasoning_parser.py
│   │   │   └── test_template_manager.py
│   │   └── function_call/         # 从 test/registered/unit/function_call/ 迁移
│   │       ├── test_function_call_parser.py
│   │       ├── test_hermes_detector.py
│   │       ├── test_hunyuan_detector.py
│   │       ├── test_json_schema_constraint.py
│   │       ├── test_llama32_detector.py
│   │       ├── test_minicpm5_detector.py
│   │       ├── test_minimax_m3_detector.py
│   │       ├── test_mistral_detector.py
│   │       ├── test_normalize_json_schema_types.py
│   │       ├── test_parallel_tool_calls.py
│   │       ├── test_poolside_v1_detector.py
│   │       └── test_unknown_tool_name.py
│   └── integration/
│       └── test_process_messages.py   # 新写，针对 _process_messages 在常见场景下的端到端正确性
├── docs/
│   └── superpowers/specs/
│       └── 2026-08-12-llm-router-utils-design.md
└── scripts/
    └── (暂无)
```

**关键说明：**

1. **src layout**：`src/llm_router_utils/sglang/...` 的包路径为 `llm_router_utils.sglang.srt.parser.reasoning_parser`，与"sglang 作为包路径一段"一致。
2. **test 不放在 src 里**：遵循 Python 主流约定，测试在仓库根 `test/` 下。
3. **瘦身文件标注**：标注"瘦身版"的文件会在迁移时按 `_process_messages` 调用链反向裁剪，保留必需部分。
4. **`_mps_stub.py` / `_triton_stub.py`** 等 macOS stub 文件不迁移（与推理相关）。
5. **multimodal_gen/、kernels/、jit_kernel/、bench_*/、eval/、cli/、lang/**：全部不迁移（推理或前端 DSL 相关，与 parser/render 无关）。

---

## 3. 迁移策略与裁剪规则

**总策略：** 先整体拷贝 `sglang_ksogit/python/sglang/` 到 `src/llm_router_utils/sglang/`，然后按"自顶向下 + 反向依赖闭包"两阶段裁剪。

### 阶段 A：全量拷贝

```
cp -r /e/githome-windows/sglang_ksogit/python/sglang/  /e/githome-windows/llm_router_utils/src/llm_router_utils/sglang/
```

拷贝后，对每个 `.py` 文件做一次"导入语法是否正确"的冒烟检查（仅 `ast.parse`，不实际 import），确保后续裁剪有干净起点。

### 阶段 B：删除明显无关的子树

直接整目录删除（删除前对每个待删文件做一次"是否被保留代码 import"的 grep 校验，若被引用则保留并标为"待裁剪"）：

| 路径 | 删除理由 |
|---|---|
| `sglang/benchmark/` | 推理基准测试 |
| `sglang/eval/` | 推理评测 |
| `sglang/cli/` | sglang 启动 CLI（推理相关） |
| `sglang/lang/` | sglang 前端 DSL（与 router 无关） |
| `sglang/multimodal_gen/` | 多模态生成（推理） |
| `sglang/jit_kernel/` | JIT kernel（推理） |
| `sglang/kernels/` | CUDA/Triton kernels |
| `sglang/auto_benchmark*.py`、`bench_*.py`、`launch_server.py`、`check_env.py`、`compile_deep_gemm.py`、`profiler.py` | 推理启动与基准 |
| `sglang/srt/layers/`、`model_executor/`、`model_loader/`、`models/` | 模型实现 |
| `sglang/srt/sampling/`、`constrained/`、`speculative/` | 采样/约束解码/推测解码 |
| `sglang/srt/distributed/`、`disaggregation/`、`connector/`、`eplb/`、`elastic_ep/`、`dllm/`、`multiplex/` | 分布式 |
| `sglang/srt/mem_cache/`、`kv_canary/`、`state_capturer/`、`checkpoint_engine/` | KV cache / 检查点 |
| `sglang/srt/lora/` | LoRA |
| `sglang/srt/managers/` 下调度相关（scheduler*/tp_worker*/detokenizer_manager/data_parallel_controller/schedule_batch/schedule_policy/communicator/hisparse_coordinator/cache_controller/overlap_utils/prefill_delayer/min_free_slots_delayer/multi_tokenizer_mixin/multimodal_processor/mm_utils/load_snapshot/disagg_service/utils/async_dynamic_batch_tokenizer/configure_logging/scheduler_input_blocker/tokenizer_control_mixin/tokenizer_manager_score_mixin） | 推理调度 |
| `sglang/srt/observability/`、`compilation/`、`batch_invariant_ops/`、`batch_overlap/` | 推理可观测/编译 |
| `sglang/srt/entrypoints/` 下推理服务（anthropic/ollama/search/realtime/grpc*/http_server*/engine*/sidecar/warmup/v1_loads/ssl_utils/http_request_decompression/request_headers/tool/context/elastic_ep） | 推理 HTTP 服务 |
| `sglang/srt/platforms/`、`hardware_backend/` | 硬件后端 |
| `sglang/srt/plugins/`、`session/`、`arg_groups/` | 推理相关 |
| `sglang/srt/utils/` 下推理相关文件（cuda_ipc_transport_utils/cudacore_pyspy_dump_utils/custom_op/device_timer/host_shared_memory/nvtx_*/profile_*/torch_memory_saver_adapter/torch_npu_patch_utils/rpd_utils/slow_rank_detector/stale_shm_cleanup/tensor_bridge/weight_checker*/async_probe/bench_utils/gauge_histogram/model_file_verifier/network/numa_utils/poll_based_barrier/aio_rwlock/aiter/http_middleware_patch/phase_checker/watchdog/offloader/patch_torch/patch_tokenizer/video_decoder/runai_utils/hf_transformers_patches/hf_transformers_utils(若 _process_messages 不用)/scheduler_status_logger/multi_stream_utils/msgspec_utils/json_response/field_validators/common(待裁剪)/auth/log_utils/request_logger） | 推理相关 |

**注意：** `parser/` 与 `function_call/` 完整保留，不在本阶段删除。

### 阶段 C：反向依赖闭包裁剪

以 `_process_messages` 为根，反向追踪实际调用链上用到的符号，对"待裁剪"文件做精细裁剪。

**核心保留集合（`_process_messages` 调用链）：**

- **`serving_chat.py`**：
  - 保留：`__init__`、`_resolve_chat_encoding_spec`、`_encode_messages`、`_patch_reasoning_skip_special_tokens`、`_get_reasoning_from_request`、`_apply_jinja_template`、`_apply_conversation_template`、`_process_messages`
  - 删除：`_generate_stream_content`、`_handle_streaming_request`、`_handle_non_streaming_request`、`_generate_chat_stream`、`_process_tool_call_stream`、`_process_tool_call_non_stream` 等所有调度方法
- **`serving_base.py`**：完整保留（306 行）
- **`protocol.py`**：完整保留（1,879 行）
- **`parser/*`、`function_call/*`**：完整保留
- **`tokenizer/tiktoken_tokenizer.py`**：完整保留
- **`entrypoints/openai/`**：保留 `chat_encoding.py`、`encoding_dsv32.py`、`encoding_dsv4.py`、`sse_utils.py`、`usage_processor.py`、`utils.py`、`__init__.py`
- **`managers/tokenizer_manager.py`**：仅保留
  - `__init__(server_args, port_args)`
  - `init_model_config()`
  - `init_tokenizer_and_processor()`
  - 属性：`.tokenizer`、`.processor`、`.model_config`、`.server_args`
  - 删除所有调度相关方法（`_dispatch_to_scheduler`、`generate_request`、`create_abort_task`、`_handle_abort_finish_reason`、`init_ipc_channels`、`init_running_status`、`init_request_logging_and_dumping`、`init_weight_update`、`init_lora`、`init_disaggregation`、`init_metric_collector_watchdog`、`init_request_dispatcher`、所有 `_validate_*`、`_prepare_tokenizer_input`、`_extract_tokenizer_results` 等）
- **`managers/io_struct.py`**：仅保留 `GenerateReqInput`（被 serving_chat 引用）；若 `GenerateReqInput` 仅在已删方法中被构造，则可整体删除该 import
- **`configs/model_config.py`**：仅保留 `ModelConfig` 类的 `__init__`、`hf_config`、`is_multimodal`、`get_default_sampling_params()`、`context_length`
- **`server_args.py`**：保留 `ServerArgs` dataclass，仅保留字段：`model_path`、`chat_template`、`tool_call_parser`、`reasoning_parser`、`context_length`、`served_model_name`、`default_chat_template_kwargs`、`enable_cache_report`、`incremental_streaming_output`、`stream_response_default_include_usage`、`allow_auto_truncate`、`skip_tokenizer_init`（init_tokenizer 用）；保留 `__post_init__` 中与这些字段相关的逻辑；保留 `PortArgs` dataclass（最小版）；保留 `from_cli` 的简化版（仅解析上述字段）
- **`environ.py`**：保留 `EnvField` 体系、`ToolStrictLevel`、`Envs` 类中 `_process_messages` 链路用到的环境变量字段（如 `SGLANG_TOOL_STRICT_LEVEL`、`SGLANG_DEFAULT_THINKING` 等），删除末尾 `import sglang.srt.debug_utils.cuda_coredump`
- **`utils/common.py`**：仅保留 `ImageData`、`VideoData`、`read_system_prompt_from_file`、`find_local_repo_dir` 等被 parser 用的函数；删除所有 torch/CUDA/PIL/numpy 相关函数
- **`utils.py`**：仅保留 `convert_json_schema_to_str`、`LazyImport`（若仍需要）等被引用的函数

### 阶段 D：修复导入与运行时

1. **全局替换 import 路径**：所有 `from sglang.` 改为 `from llm_router_utils.sglang.`，`from sglang.srt.` 改为 `from llm_router_utils.sglang.srt.`。
2. **TokenizerManager Protocol**：在 `template_manager.py` 中新增 `TokenizerLike` Protocol（含 `.tokenizer`/`.processor`/`.model_config`/`.server_args` 属性），将原代码中所有 `tokenizer_manager: TokenizerManager` 类型注解改为 `TokenizerLike`，但保留 `TokenizerManager` 实际类作为默认实现。
3. **删除推理专属 import**：清理所有已删模块的 import 行，避免 `ImportError`。
4. **顶层 `__init__.py`**：原 `sglang/__init__.py` 顶部有一段 `_mps_stub` / `_triton_stub` / `apply_all_hf_patches` 的早期 import，全部删除；只保留 `__version__` 导出与对 parser/function_call 的便利 re-export。
5. **`srt/utils/__init__.py`**：原内容是 `from sglang.srt.utils.common import *`，瘦身后改为显式导出 `ImageData`、`VideoData`、`read_system_prompt_from_file` 等。

### 阶段 E：测试迁移与清理

1. 将 `sglang_ksogit/test/registered/unit/parser/` 与 `test/registered/unit/function_call/` 整目录拷贝到 `llm_router_utils/test/unit/`。
2. 全局替换 import 路径（同阶段 D）。
3. 删除测试中对 `sglang.test.ci.ci_register.register_cpu_ci` 与 `sglang.test.test_utils.CustomTestCase` 的依赖，改为：
   - `register_cpu_ci` 调用直接删除（它是 CI 注册装饰器，与单测逻辑无关）。
   - `CustomTestCase` 改为 `unittest.TestCase`。
4. 删除测试中 `setUp` 里启动 sglang server 的代码（如有）。
5. 新写 `test/integration/test_process_messages.py`，覆盖几个端到端场景：
   - 纯文本 chat（无 tools、无 reasoning）
   - 带 tools 的 chat（hermes/qwen3_coder/glm4 等典型 detector）
   - 带 reasoning parser 的 chat（deepseek/qwq 风格）
   - chat_template 走 jinja vs 走 conversation 自定义模板
   - input_ids 直接传入的场景

### 阶段 F：依赖与 pyproject

`pyproject.toml` 关键配置：

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-router-utils"
dynamic = ["version"]
requires-python = ">=3.10"
dependencies = [
  "pydantic",
  "jinja2",
  "partial_json_parser",
  "orjson",
  "typing_extensions",
  "tiktoken",
  "transformers",   # AutoTokenizer 加载
  "openai",         # protocol.py 中 openai.types.responses 引用
  "outlines",       # 若 function_call/utils.py 仍依赖
  "xgrammar",       # 若 function_call 链路需要
  "llguidance",     # 同上
  "mistral_common", # mistral_detector 需要
  "packaging",
]

[project.optional-dependencies]
test = ["pytest", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]
```

**依赖最小化策略：** 阶段 C 完成后，对每个 `import` 做一次"是否真的需要"的扫描，若某个第三方包仅被已删代码引用，则从 dependencies 中移除（如 `outlines`/`xgrammar`/`llguidance` 若 `_process_messages` 链路不需要就移除）。

### 验证标准（每阶段完成后）

| 阶段 | 验证命令 | 期望 |
|---|---|---|
| B | `python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('src/**/*.py', recursive=True)]"` | 全部文件语法正确 |
| C | `python -c "from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat"` | 无 ImportError |
| D | `python -c "from llm_router_utils.sglang.parser.reasoning_parser import ReasoningParser; from llm_router_utils.sglang.function_call.function_call_parser import FunctionCallParser"` | 无 ImportError |
| E | `pytest test/unit/ -x` | 全部单元测试通过（与原版行为一致） |
| F | `pytest test/integration/test_process_messages.py -x` | 集成测试通过 |

---

## 4. 正确性保证与风险

### 正确性保证策略

1. **逐文件行为对比**：对每个保留的文件，迁移后用 `diff` 对比原版，确认仅有的差异是：
   - import 路径替换（`sglang.` → `llm_router_utils.sglang.`）
   - 显式标注的裁剪删减
   - Protocol 替换（TokenizerManager 类型注解）

   不应有的差异：逻辑改动、变量重命名、函数签名变更。

2. **原版测试作为黄金基准**：迁移过来的 `test/unit/parser/` 与 `test/unit/function_call/` 测试用例**一字不改地保留断言**（仅改 import 路径与 `CustomTestCase` → `TestCase`）。这些测试在原 sglang 仓库里是绿的，迁移后必须继续绿——这是"与原版无偏差"的最硬证据。

3. **跨版本快照对比**（针对 `_process_messages` 集成测试）：构造一批典型 `ChatCompletionRequest` fixture，分别在原 sglang 仓库与迁移后的 `llm_router_utils` 上调用 `_process_messages`，对比返回的 `MessageProcessingResult` 字段（prompt、prompt_ids、stop、tool_call_constraint 等）是否完全一致。任何字段差异都视为迁移 bug。

4. **不引入新逻辑**：迁移期间不重构、不优化、不"顺手改进"。即使看到原代码的明显瑕疵（如重复代码、命名不佳），也保持原样，留给后续独立 PR。这条纪律是保证"正确性与原版无偏差"的前提。

### 已识别风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| **R1: `_process_messages` 隐式依赖未被 grep 抓到** | 运行时 ImportError 或 AttributeError | 阶段 C 完成后立即跑集成测试；每个裁剪提交单独跑一次 `pytest test/unit/ -x`；遇到缺失符号时回退裁剪，补回该符号再删 |
| **R2: `protocol.py` 完整保留但顶部 `from openai.types.responses import ...` 引入重依赖** | 安装时拖入 openai SDK（已确认可接受） | 已在选定的依赖范围内；若后续发现仅用到几个类型，可考虑内联定义以去除 openai 依赖，但当前不做 |
| **R3: `ModelConfig.hf_config` 来自 transformers `AutoConfig.from_pretrained`，加载 model_path 时会真实下载/读取模型配置** | 测试时若不指定本地 model_path，会触发网络请求 | 测试 fixture 使用本地小模型或 mock `ModelConfig`；集成测试里用 `prefer_local_files=True` 类参数避免联网 |
| **R4: `srt/utils/common.py` 瘦身可能漏删 torch 间接依赖** | `import llm_router_utils.sglang.srt.utils.common` 时仍触发 `import torch` | 阶段 C 完成后用 `python -c "import llm_router_utils.sglang.srt.utils.common; import sys; assert 'torch' not in sys.modules"` 显式验证 |
| **R5: `environ.py` 末尾 `import sglang.srt.debug_utils.cuda_coredump` 漏删** | import 链触发 CUDA 相关模块加载失败 | 阶段 C 显式删除该行，并 grep 全仓库 `cuda_coredump` 确认无残留 |
| **R6: `TokenizerControlMixin` / `TokenizerManagerScoreMixin` 是 TokenizerManager 的父类，瘦身时若删父类方法会导致子类 `super()` 调用断裂** | 实例化 TokenizerManager 时 AttributeError | 评估后倾向于：两个 mixin 文件若被引用则整体保留（即使部分方法用不到），不细删方法级；若整体保留代价过大则改为 TokenizerManager 不再继承它们（修改 `class TokenizerManager:` 而非 `class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):`）—— 此决策在阶段 C 触达该文件时再次确认 |
| **R7: parser/function_call 中某些 detector 可能 import 已删模块（如 `constrained.*`）** | import 失败 | 阶段 C 每删一个模块后跑一次 `python -c "import llm_router_utils.sglang.function_call"` 全量 import 冒烟 |
| **R8: 测试中 `sglang.test.test_utils` 的其他工具函数被引用** | 测试运行时 ImportError | 阶段 E 逐个测试文件处理，把对 `test_utils` 的引用替换为本地 helper 或删除 |
| **R9: `template_manager.py` 的 Protocol 替换可能影响原版测试 `test_template_manager.py` 的断言** | 测试失败 | Protocol 仅替换类型注解，不改变运行时行为；若测试里 `isinstance(tokenizer_manager, TokenizerManager)` 之类断言则保持 TokenizerManager 类可用；预期不影响 |
| **R10: `function_call/utils.py` 可能依赖 `outlines`/`xgrammar`/`llguidance` 等约束解码库** | 依赖扩大 | 阶段 F 显式 grep `import outlines`/`import xgrammar`/`import llguidance`，若仅在已删代码中引用则从 dependencies 移除 |

### 不在本次范围

明确不做的事，避免范围蔓延：

- 不迁移任何推理调度代码（scheduler / detokenizer / data_parallel 等）
- 不迁移任何 HTTP 服务代码（http_server / fastapi app / SSE 流式协议实现）
- 不迁移 sglang 前端 DSL（`sglang.lang.*`）
- 不迁移 CLI 启动逻辑
- 不写自动同步上游脚本
- 不重构、不优化原代码（即使看到改进点）
- 不支持 LoRA / 多模态推理 / 推测解码（这些能力的 parser 部分若被 `_process_messages` 链路引用则保留，但不迁移它们的运行时实现）

---

## 5. 调用形态示例

迁移完成后，调用方使用方式：

```python
from llm_router_utils.sglang.srt.configs.model_config import ModelConfig
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from llm_router_utils.sglang.srt.server_args import ServerArgs, PortArgs

server_args = ServerArgs(model_path="Qwen/Qwen3-32B", tool_call_parser="qwen3_coder")
port_args = PortArgs()  # 最小版，无实际端口
tokenizer_manager = TokenizerManager(server_args, port_args)
template_manager = TemplateManager()
template_manager.initialize_templates(
    tokenizer_manager=tokenizer_manager,
    model_path=server_args.model_path,
    chat_template=server_args.chat_template,
)
openai_serving_chat = OpenAIServingChat(tokenizer_manager, template_manager)

# 后续 router 服务调用：
result = openai_serving_chat._process_messages(request, is_multimodal=False)
# result.prompt / result.prompt_ids / result.stop / result.tool_call_constraint
```
