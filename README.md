# llm-router-utils

Lightweight extraction of sglang's reasoning parser, tool-call parser, and chat template rendering, for use in custom router services and lightweight LLM applications.

This library does **not** include any inference engine code. It only provides the "frontend" message processing pipeline: `OpenAIServingChat._process_messages` and its dependencies.

**Upstream source:** sglang [release/v0.5.16](https://github.com/sgl-project/sglang/tree/release/v0.5.16).

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
