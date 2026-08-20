# 升级指南：同步上游 sglang 版本

本仓库从 sglang 前端消息处理层裁剪而来。当上游 sglang 发布新版本时，需要把本仓库保留的文件同步到新版本。本文档描述完整的升级流程、裁剪规则与常见踩坑点。

## 前置准备

1. **克隆上游 sglang 仓库**（如尚未有）：
   ```bash
   git clone https://github.com/sgl-project/sglang.git /path/to/sglang
   cd /path/to/sglang
   git checkout release/v0.5.X   # 目标版本
   ```

2. **从 master 创建升级分支**：
   ```bash
   cd /path/to/llm_router_utils
   git checkout master
   git checkout -b upgrade/sglang-0.5.X
   ```

3. **确认基线 tag**：本仓库当前对应的 sglang 版本记录在 `README.md` 的 "Upstream source" 行，以及下文的版本对应表。

## 第一步：确定变更范围

上游仓库路径记为 `$UPSTREAM`（例如 `E:/githome-windows/sglang_ksogit/python`）。**注意**：`git diff` 时用 `sglang/` 前缀（不带 `python/`），否则返回空。

```bash
cd $UPSTREAM
# 列出 srt 目录下两个版本间所有变更文件
git diff --name-status v0.5.OLD v0.5.NEW -- sglang/srt/ > /tmp/changed.txt

# 列出本仓库保留的文件
cd /path/to/llm_router_utils
find src/llm_router_utils/sglang -name "*.py" | sed 's|src/llm_router_utils/sglang/||' | sort > /tmp/kept.txt
```

求交集：对 `/tmp/changed.txt` 中每个 `M`/`A` 文件，检查其路径（去掉 `python/sglang/` 前缀）是否在 `/tmp/kept.txt` 中。交集即为需要更新的文件。

```bash
cd $UPSTREAM
while read status path; do
  rel=$(echo "$path" | sed 's|^python/sglang/||')
  if grep -qx "$rel" /tmp/kept.txt; then
    echo "$status $rel"
  fi
done < /tmp/changed.txt
```

**新增文件**（`A`）的处理：只纳入 function_call 检测器（本仓库保留全部检测器）。其他新增文件（推理引擎、config spec、observability 等）不纳入。

## 第二步：应用变更

### 完整保留的文件

对于与上游 0.5.OLD 字节一致（仅 import 改写差异）的文件，直接用上游新版本替换：

```bash
cd $UPSTREAM
git show v0.5.NEW:python/sglang/srt/<path> \
  | sed -e 's|from sglang\.|from llm_router_utils.sglang.|g' \
        -e 's|import sglang\.|import llm_router_utils.sglang.|g' \
  > /path/to/llm_router_utils/src/llm_router_utils/sglang/srt/<path>
```

典型完整保留文件：`parser/reasoning_parser.py`、`function_call/` 下各检测器、`environ.py`。

### Slimmed 文件

对于已裁剪的文件（如 `configs/model_config.py`、`managers/tokenizer_manager.py`、`server_args.py`、`utils/common.py`、`entrypoints/openai/serving_chat.py`），**不能**直接替换。需手动对照上游 diff，只把保留部分的逻辑变更应用上来：

```bash
cd $UPSTREAM
git diff v0.5.OLD v0.5.NEW -- sglang/srt/<path>
git show v0.5.NEW:python/sglang/srt/<path>   # 完整内容对照
```

用编辑工具（`replace_string_in_file`）逐处应用。跳过上游 diff 中涉及本仓库已删除方法的变更。

## 第三步：裁剪与适配规则（硬不变量）

以下规则来自 `CLAUDE.md`，**必须严格遵守**：

### 1. Import 改写

- `from sglang.xxx` → `from llm_router_utils.sglang.xxx`
- `import sglang.xxx` → `import llm_router_utils.sglang.xxx`
- **注意**：`sed` 只替换 `from sglang.`/`import sglang.`，不替换字符串字面量里的 `sglang.`（如 `ModuleType("sglang.srt...")` 需单独处理）。

### 2. 禁止直接 import torch

`src/` 下不能出现 `import torch` 或 `from torch import ...`。torch 只能通过 xgrammar 的 tvm_ffi 间接引入。若上游 diff 新增了 torch import 且仅用于推理引擎路径，删除该 import。

### 3. xgrammar import 必须 try/except

所有 xgrammar 相关 import 必须包在 `try/except ImportError` 里，失败时 fallback 到 `typing.Any`：

```python
try:
    from xgrammar import StructuralTag
except ImportError:
    StructuralTag = Any
```

若上游新增 xgrammar 子模块 import（如 `from xgrammar.structural_tag import ...`），需为每个符号提供 `Any` fallback。

### 4. 不功能重写保留代码

只删除/裁剪，不重写逻辑。本仓库文件的逻辑变更必须与上游对齐。若发现上游 bug，应上游先修再 re-import，不在本仓库单独修。

### 5. 迁移测试是回归基线

测试断言必须与上游字节一致，只适配：
- 移除 `from sglang.test.ci.ci_register import register_cpu_ci` 及 `register_cpu_ci(...)` 调用
- `CustomTestCase` → `unittest.TestCase`
- `from sglang.srt.utils.hf_transformers_utils import get_tokenizer` → `from transformers import AutoTokenizer as _AT; get_tokenizer = lambda name, **kw: _AT.from_pretrained(name, **kw)`
- `sglang.` → `llm_router_utils.sglang.`

## 第四步：Windows 测试适配

`tempfile.NamedTemporaryFile` 在 Windows 上默认独占锁定，无法被同名重新打开。涉及读取临时 jinja 模板的测试需改用 `mkstemp`：

```python
import os, tempfile
fd, path = tempfile.mkstemp(suffix=".jinja")
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(template_content)
    # ... 使用 path ...
finally:
    os.unlink(path)
```

## 第五步：验证

```bash
cd /path/to/llm_router_utils
PYTHONPATH=src python -m pytest test/ -q
```

所有测试必须通过。若上游改了实现（如 inkling_detector 重写），对应测试需同步更新到上游新版本。

## 第六步：更新版本记录

1. 更新 `README.md` 的 "Upstream source" 行为新版本。
2. 在 `README.md` 的版本对应表追加一行。
3. 提交并打 tag。

## 常见踩坑

- **`git diff` 路径前缀**：必须用 `sglang/` 而非 `python/sglang/`，否则返回空结果。
- **subagent 只读**：`Explore` subagent 无法编辑文件，需自己应用变更。
- **grep 行号过期**：文件编辑后 `grep_search` 行号会失效，用 `grep -n` 重新定位。
- **特殊字符**：处理含 XML 标签（`<...>`）的代码时，工具调用可能被误解析，用 `create_file` 写脚本或 base64 编码处理。
- **CRLF**：Windows 检出会导致 CRLF，与上游 LF 对比时用 `diff --strip-trailing-cr`。
- **slimmed 文件的"无需变更"判断**：上游变更若全落在已剥离的推理引擎代码（调度、dispatch、metrics、cuda graph 等），则本仓库 slimmed 版本无需变更。需逐文件确认。

## 版本对应表

见 `README.md` 中的 "版本对应关系" 章节。
