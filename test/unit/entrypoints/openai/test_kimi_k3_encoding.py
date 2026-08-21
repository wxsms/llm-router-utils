"""Unit tests for the KimiK3 encoding path in OpenAIServingChat.

Migrated from upstream sglang v0.5.17
test/registered/unit/entrypoints/openai/test_serving_chat.py
(test_kimi_k3_encoder_receives_wire_request_fields,
 test_kimi_k3_neutralizes_text_only_assistant_history).

These cover the _prepare_kimi_k3_messages + _encode_messages kimi_k3 branch:
image-placeholder neutralization, message-level tools injection, and
template-kwargs setup (image_prompts/tool_choice/response_format).
"""
import unittest
from unittest.mock import MagicMock

from llm_router_utils.sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
)
from llm_router_utils.sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs


def _make_fake_tokenizer_manager(server_args: ServerArgs) -> TokenizerManager:
    """Build a TokenizerManager with mocked tokenizer (no model download)."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = server_args
    tm.port_args = PortArgs()
    tok = MagicMock()
    tok.bos_token_id = None
    tok.eos_token_id = 1
    tok.apply_chat_template.return_value = [7, 8, 9]
    tok.chat_template = (
        "{% for message in messages %}{{ message.role }}: "
        "{{ message.content }}\n{% endfor %}"
    )
    tm.tokenizer = tok
    tm.processor = None
    tm.model_config = MagicMock()
    tm.model_config.hf_config = MagicMock()
    tm.model_config.hf_config.architectures = []
    tm.model_config.hf_config.model_type = "llama"
    tm.model_config.is_multimodal = False
    tm.model_config.get_default_sampling_params.return_value = {}
    return tm


def _make_serving(chat_encoding_spec: str = "kimi_k3", is_multimodal: bool = False):
    server_args = ServerArgs(model_path="x")
    tm = _make_fake_tokenizer_manager(server_args)
    tm.model_config.is_multimodal = is_multimodal
    tpl = TemplateManager()
    tpl.initialize_templates(
        tokenizer_manager=tm,
        model_path="x",
        chat_template=None,
    )
    # initialize_templates(chat_template=None) leaves chat_template_name None,
    # so _process_messages routes to _apply_jinja_template -> _encode_messages.
    serving = OpenAIServingChat(tm, tpl)
    serving.chat_encoding_spec = chat_encoding_spec
    return serving, tm


class TestKimiK3Encoding(unittest.TestCase):
    def test_kimi_k3_encoder_receives_wire_request_fields(self):
        serving, tm = _make_serving(chat_encoding_spec="kimi_k3", is_multimodal=True)
        tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"},
            },
        }
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "developer",
                    "content": "<|kimi_image_placeholder|>",
                    "tools": [tool],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Explain <|kimi_image_placeholder|>",
                        },
                        {"type": "image_url", "image_url": {"url": "image-1"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect <|kimi_image_placeholder|>",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "inspect",
                                "arguments": {
                                    "source": "<|kimi_image_placeholder|>",
                                    "nested": ["<|kimi_image_placeholder|>"],
                                },
                            },
                        }
                    ],
                },
            ],
            tools=[tool],
            tool_choice="required",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": False,
                },
            },
        )

        result = serving._process_messages(request, is_multimodal=True)

        call = tm.tokenizer.apply_chat_template.call_args
        rendered_messages = call.args[0]
        self.assertEqual(rendered_messages[0]["role"], "system")
        self.assertEqual(
            rendered_messages[0]["content"], "<| kimi_image_placeholder |>"
        )
        self.assertNotIn("strict", rendered_messages[0]["tools"][0]["function"])
        self.assertEqual(
            rendered_messages[1]["content"][0]["text"],
            "Explain <| kimi_image_placeholder |>",
        )
        self.assertEqual(
            rendered_messages[2]["reasoning_content"],
            "Inspect <| kimi_image_placeholder |>",
        )
        self.assertEqual(
            rendered_messages[2]["tool_calls"][0]["function"]["arguments"],
            {
                "source": "<| kimi_image_placeholder |>",
                "nested": ["<| kimi_image_placeholder |>"],
            },
        )
        self.assertEqual(call.kwargs["image_prompts"], ["<|media_pad|>"])
        self.assertEqual(call.kwargs["tool_choice"], "required")
        self.assertNotIn("strict", call.kwargs["tools"][0]["function"])
        self.assertEqual(
            call.kwargs["response_format"]["json_schema"]["schema"],
            {"type": "object"},
        )
        self.assertNotIn("schema_", call.kwargs["response_format"]["json_schema"])
        self.assertEqual(result.prompt_ids, [7, 8, 9])
        self.assertEqual(result.image_data[0].url, "image-1")

    def test_kimi_k3_neutralizes_text_only_assistant_history(self):
        serving, tm = _make_serving(chat_encoding_spec="kimi_k3", is_multimodal=False)
        tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Run it"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Read <|kimi_image_placeholder|>",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": "not-json <|kimi_image_placeholder|>",
                            },
                        }
                    ],
                },
            ],
        )

        serving._process_messages(request, is_multimodal=False)

        messages = tm.tokenizer.apply_chat_template.call_args.args[0]
        kwargs = tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(
            messages[-1]["reasoning_content"],
            "Read <| kimi_image_placeholder |>",
        )
        self.assertEqual(
            messages[-1]["tool_calls"][0]["function"]["arguments"],
            "not-json <| kimi_image_placeholder |>",
        )
        self.assertNotIn("image_prompts", kwargs)


if __name__ == "__main__":
    unittest.main()
