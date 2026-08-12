from __future__ import annotations

import copy
import logging
import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


class ThinkingMode(str, Enum):
    """Mode for message encoding - chat vs thinking/reasoning."""

    CHAT = "chat"
    THINKING = "thinking"


import jinja2
import orjson

from llm_router_utils.sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv32
from llm_router_utils.sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
    ResponseParserProtocol,
    ToolChoice,
)
from llm_router_utils.sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from llm_router_utils.sglang.srt.environ import envs
from llm_router_utils.sglang.srt.function_call.function_call_parser import FunctionCallParser
from llm_router_utils.sglang.srt.function_call.utils import (
    get_json_schema_constraint,
)
from llm_router_utils.sglang.srt.parser.conversation import generate_chat_conv
from llm_router_utils.sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from llm_router_utils.sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager
    from llm_router_utils.sglang.srt.parser.template_manager import TemplateManager

logger = logging.getLogger(__name__)


def normalize_tool_content(role: str, content):
    """Normalize tool message content from OpenAI array format to plain string.

    OpenAI clients may send tool content as a list of content parts
    (e.g. [{"type":"text","text":"..."}]) but most chat templates expect
    a plain string for tool messages. Only flatten when ALL items are
    pure OpenAI text parts; preserve lists containing non-text-type items
    that some templates intentionally iterate over.
    """
    if role != "tool" or not isinstance(content, list):
        return content
    parts = content
    is_openai_text_parts = all(
        (isinstance(p, dict) and p.get("type") == "text") or isinstance(p, str)
        for p in parts
    )
    if is_openai_text_parts:
        text_parts = [p.get("text", "") if isinstance(p, dict) else p for p in parts]
        return " ".join(text_parts)
    return content


def parse_tool_call_arguments(arguments: str) -> Dict[str, Any]:
    """Parse OpenAI tool call arguments for chat templates."""
    try:
        parsed_arguments = orjson.loads(arguments)
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            "Assistant tool call function.arguments must be valid JSON."
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError(
            "Assistant tool call function.arguments must be a JSON object."
        )

    return parsed_arguments


def normalize_assistant_tool_call_arguments(message: Dict[str, Any]) -> None:
    """Normalize assistant history tool call arguments in-place."""
    if message.get("role") != "assistant" or not isinstance(
        message.get("tool_calls"), list
    ):
        return

    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        if "arguments" in function and isinstance(function["arguments"], str):
            function["arguments"] = parse_tool_call_arguments(function["arguments"])


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # pydantic object or dict type
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): per-item max_dynamic_patch for both image and video
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    _default_sampling_params_logged = False

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
        self.default_chat_template_kwargs = (
            self.tokenizer_manager.server_args.default_chat_template_kwargs or {}
        )
        self._reasoning_detector = None
        if self.reasoning_parser:
            try:
                rp = ReasoningParser(
                    model_type=self.reasoning_parser,
                    stream_reasoning=True,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                self._reasoning_detector = rp.detector
            except ValueError as e:
                logger.warning(
                    "Failed to initialize reasoning detector for parser '%s': %s",
                    self.reasoning_parser,
                    e,
                )

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

        # Check if the model is a GPT-OSS model
        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )
        self.is_gemma4 = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type
            in ("gemma4", "gemma4_unified")
        )

        # Which Python-based chat encoder (if any) bypasses apply_chat_template.
        # Values: "dsv32", "dsv4", or custom values set by subclass. None for default.
        self.chat_encoding_spec = self._resolve_chat_encoding_spec()

        # Resolve the env-configured Inkling effort default once: the env var is
        # frozen for the server's lifetime, and a misconfigured value should
        # fail at boot, not 400 every request.
        self._inkling_default_reasoning_effort: Optional[float] = (
            self._get_inkling_default_reasoning_effort()
            if self.chat_encoding_spec == "inkling"
            else None
        )

        # Per-request response parser for custom decoding (set by _encode_messages)
        self._response_parser: Optional[ResponseParserProtocol] = None

        # Probe whether ``encode("")`` returns specials. If it does, we must
        # keep ``add_special_tokens=False`` at the chat-template encode site
        # to avoid double BOS; otherwise the kwarg is a no-op and dropping it
        # lets slow tokenizers (e.g. Kimi's TikTokenTokenizer) stay on the
        # fast internal path.
        try:
            self._tokenizer_auto_adds_specials = (
                len(self.tokenizer_manager.tokenizer.encode("")) > 0
            )
        except Exception:
            self._tokenizer_auto_adds_specials = True

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle continue_final_message feature: separate final assistant message.

        If continue_final_message is enabled and the last message is from assistant,
        extract its content and remove it from the message list.
        If continue_final_message is False and the last message is from assistant,
        convert it to a user message to ensure the last message is always from user.

        Only processes text-based content (strings), ignoring multimodal content (lists).

        Args:
            messages: List of message dictionaries
            request: ChatCompletionRequest with continue_final_message flag

        Returns:
            Tuple of (processed_messages, assistant_prefix)
            - processed_messages: Messages with last assistant message handled appropriately
            - assistant_prefix: Content of the last assistant message (string only), or None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # Only process string content, ignore multimodal content (lists)
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # Extract content and remove the assistant message
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # Convert the last assistant message to user message
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        Append assistant prefix to prompt_ids.

        Args:
            prompt_ids: Current prompt token IDs
            assistant_prefix: Assistant message content to append

        Returns:
            Updated prompt_ids with assistant prefix appended
        """
        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _resolve_chat_encoding_spec(self) -> Optional[str]:
        """Determine which chat encoding spec to use.

        Override in subclass to add custom encoding specs.
        """
        from llm_router_utils.sglang.srt.entrypoints.openai.chat_encoding import (
            resolve_chat_encoding_spec,
        )

        return resolve_chat_encoding_spec(
            hf_config=self.tokenizer_manager.model_config.hf_config,
            tokenizer=self.tokenizer_manager.tokenizer,
            tool_call_parser=self.tool_call_parser,
        )

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _encode_messages(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
        thinking_mode: ThinkingMode,
        tools: Optional[List[Dict]] = None,
    ) -> Optional[List[int]]:
        """Encode messages for custom chat_encoding_spec values.

        Returns prompt_ids if handled, None to use default encoding.
        """
        if self.chat_encoding_spec == "inkling":
            # Inkling: render messages -> input_ids with framing tokens + ONE placeholder per
            # media (encoding/expansion happens later in InklingMultimodalProcessor). The
            # server's tokenizer is the base tiktoken backend; wrap it so encode_special
            # supplies the framing-token overlay.
            from llm_router_utils.sglang.srt.parser.inkling_renderer import render_inkling_messages
            from llm_router_utils.sglang.srt.parser.inkling_tokenizer import (
                CONTENT_TEXT,
                MESSAGE_MODEL,
                InklingTokenizer,
            )

            inkling_tokenizer = InklingTokenizer(
                tokenizer=self.tokenizer_manager.tokenizer
            )
            reasoning_effort = self._parse_inkling_reasoning_effort(
                request.reasoning_effort
            )
            if reasoning_effort is None:
                reasoning_effort = self._inkling_default_reasoning_effort
            assistant_prefix = self._pop_inkling_assistant_prefix(messages, request)
            prompt_ids = render_inkling_messages(
                messages,
                inkling_tokenizer,
                add_generation_prompt=False,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
            if assistant_prefix is not None:
                # Continue the final assistant message inside an OPEN model text
                # block: header + payload, no <|end_message|> and no
                # <|content_model_end_sampling|>, so the model resumes the turn.
                prompt_ids += [
                    inkling_tokenizer.encode_special(MESSAGE_MODEL),
                    inkling_tokenizer.encode_special(CONTENT_TEXT),
                    *inkling_tokenizer.encode_text(assistant_prefix),
                ]
            return prompt_ids
        return None

    @staticmethod
    def _pop_inkling_assistant_prefix(
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> Optional[str]:
        """Extract the trailing assistant text for ``continue_final_message``.

        Only a plain-string assistant message with no tool calls and no
        reasoning content can be continued; anything else renders as a closed
        historical turn. Mutates ``messages`` in place (callers pass a copy).
        """
        if not request.continue_final_message or not messages:
            return None
        last = messages[-1]
        if (
            last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or last.get("tool_calls")
            or last.get("reasoning_content")
        ):
            return None
        messages.pop()
        return last["content"]

    @staticmethod
    def _parse_inkling_reasoning_effort(
        value: Optional[Union[str, float]],
    ) -> Optional[float]:
        """Convert an OpenAI-style reasoning_effort to an Inkling float."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Inkling reasoning_effort must not be a boolean")
        if isinstance(value, (int, float)):
            parsed = float(value)
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
                raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
            return parsed
        _EFFORT_MAP = {
            "none": 0.0,
            "minimal": 0.1,
            "low": 0.2,
            "medium": 0.7,
            "high": 0.9,
            "xhigh": 0.99,
            "max": 0.99,
        }
        if value in _EFFORT_MAP:
            return _EFFORT_MAP[value]
        try:
            parsed = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError("Inkling reasoning_effort must be in [0.0, 0.99]")
        return parsed

    @staticmethod
    def _get_inkling_default_reasoning_effort() -> float:
        """Read the default Inkling reasoning effort from the environment."""
        from llm_router_utils.sglang.srt.environ import envs

        val = envs.SGLANG_INKLING_DEFAULT_REASONING_EFFORT.get()
        if not val:
            return 0.9
        try:
            parsed = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be numeric"
            ) from exc
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
            raise ValueError(
                "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be in [0.0, 0.99]"
            )
        return parsed

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        if self.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            for k, v in self.default_chat_template_kwargs.items():
                ctk.setdefault(k, v)
            request.chat_template_kwargs = ctk
            effort = ctk.get("reasoning_effort")
            if effort is not None and request.reasoning_effort is None:
                request.reasoning_effort = effort

        # GptOss model needs to keep special tokens for harmony parsing
        if self.is_gpt_oss or self.is_gemma4:
            request.skip_special_tokens = False

        self._patch_reasoning_skip_special_tokens(request)

        thinking_mode = self._get_reasoning_from_request(request)
        # SGLang's ReasonerGrammarBackend owns the reasoning prefix
        # when --reasoning-parser is configured, so builtin xgrammar
        # tags must describe only the post-reasoning tool-call suffix.
        xgrammar_reasoning = thinking_mode and (
            self.tokenizer_manager.server_args.reasoning_parser is None
        )
        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(
                    request.tools,
                    self.tool_call_parser,
                    tokenizer=self.tokenizer_manager.tokenizer,
                )
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    thinking_mode=xgrammar_reasoning,
                )
            # Fallback: use generic JSON schema for required/named tool choice
            # only when no parser-specific constraint was set
            if tool_call_constraint is None and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ToolChoice)
            ):
                json_schema = get_json_schema_constraint(
                    request.tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        # When input_ids are provided, skip template tokenization entirely;
        # only stop tokens and tool_call_constraint are needed.
        if request.input_ids is not None:
            result = MessageProcessingResult(
                prompt="",
                prompt_ids=request.input_ids,
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=request.stop or [],
            )
        elif self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        # Try custom encoding first (override in subclass for custom renderers)
        thinking_requested = (request.chat_template_kwargs or {}).get(
            "thinking", envs.SGLANG_DEFAULT_THINKING.get()
        )
        thinking_mode = (
            ThinkingMode.THINKING if thinking_requested else ThinkingMode.CHAT
        )
        messages = [msg.model_dump() for msg in request.messages]
        for message in messages:
            normalize_assistant_tool_call_arguments(message)

        prompt_ids = self._encode_messages(
            copy.deepcopy(messages),
            request,
            thinking_mode,
            tools=tools,
        )

        if prompt_ids is not None:
            # Custom encoding produced prompt_ids. Text-only encoders (dsv4/dsv32) need
            # nothing more; Inkling is the only multimodal custom encoder and still needs the
            # image/audio media harvested from the messages for the MM processor.
            if self.chat_encoding_spec == "inkling":
                for message in request.messages:
                    msg_dict = message.model_dump()
                    if msg_dict.get("content") is None:
                        msg_dict["content"] = ""
                    process_content_for_template_format(
                        msg_dict,
                        "openai",
                        image_data,
                        video_data,
                        audio_data,
                        modalities,
                    )
        elif self.chat_encoding_spec is not None:
            # dsv4/dsv32 encoding path
            messages = copy.deepcopy(messages)

            # dsv4/dsv32 are text-only and consume string content; flatten
            # OpenAI parts-list content here so the encoder sees a plain string.
            for i, msg in enumerate(messages):
                if isinstance(msg.get("content"), list):
                    messages[i] = process_content_for_template_format(
                        msg, "string", [], [], [], []
                    )

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.chat_encoding_spec == "dsv32",
                )
                msg.update(processed_msg)

            # Handle continue_final_message: separate final assistant message
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # insert an empty system prompt to help render tool system prompt
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]

            # Default encoding (dsv4/dsv32)
            if self.chat_encoding_spec == "dsv4":
                # V4 encoder only accepts "max" / "high" / None.
                # OpenAI protocol defaults to "medium" which V4 rejects; drop it.
                # Fallback: if request didn't set it, try env SGLANG_DSV4_REASONING_EFFORT.
                effort_source = request.reasoning_effort
                if effort_source is None:
                    env_val = envs.SGLANG_DSV4_REASONING_EFFORT.get()
                    if env_val:
                        effort_source = env_val
                v4_reasoning_effort = (
                    effort_source if effort_source in ("max", "high") else None
                )
                if request.task is not None:
                    encoding_dsv4.attach_task_to_last_user_message(
                        messages, request.task
                    )
                real_input = encoding_dsv4.encode_messages(
                    messages,
                    thinking_mode=thinking_mode,
                    reasoning_effort=v4_reasoning_effort,
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)
            else:
                real_input = encoding_dsv32.encode_messages(
                    messages, thinking_mode=thinking_mode
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for msg_dict in copy.deepcopy(messages):
                if msg_dict.get("content") is None:
                    msg_dict["content"] = ""

                # Process content based on detected template format
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                processed_msg["content"] = normalize_tool_content(
                    processed_msg["role"], processed_msg.get("content")
                )

                openai_compatible_messages.append(processed_msg)

            # Handle continue_final_message: separate final assistant message
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            rc = self.template_manager.reasoning_config
            if rc is not None and rc.effort_kwarg is not None:
                if request.reasoning_effort == "low":
                    extra_template_kwargs.setdefault(rc.effort_kwarg, True)
                elif request.reasoning_effort in ("medium", "high", "max"):
                    logger.warning(
                        "Model '%s' supports only 'low' reasoning effort; "
                        "requested '%s' treated as default thinking",
                        self.tokenizer_manager.server_args.served_model_name,
                        request.reasoning_effort,
                    )

            # Split apply_chat_template(tokenize=True) into render + encode so we
            # can skip add_special_tokens=False on tokenizers that don't auto-add
            # specials (Kimi-like, OpenAI-chat analogue of #25265). Chat
            # templates already include role/special tokens, so the encode must
            # avoid double BOS on tokenizers that would add it.
            encode_kwargs = (
                {"add_special_tokens": False}
                if self._tokenizer_auto_adds_specials
                else {}
            )
            try:
                rendered_prompt = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )
            except Exception:
                # If the first attempt fails, try with flat function-only format.
                # Some templates (e.g. Mistral) expect tools without the OpenAI wrapper.
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    rendered_prompt = (
                        self.tokenizer_manager.tokenizer.apply_chat_template(
                            openai_compatible_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            tools=tools,
                            return_dict=False,
                            **extra_template_kwargs,
                        )
                    )
                    prompt_ids = self.tokenizer_manager.tokenizer.encode(
                        rendered_prompt, **encode_kwargs
                    )
                except (jinja2.TemplateError, TypeError) as template_error:
                    # Template errors (e.g., from raise_exception in Jinja templates)
                    # and TypeError (e.g., tojson filter on Jinja2 Undefined variables)
                    # should be treated as client errors (400 BadRequest)
                    raise ValueError(str(template_error)) from template_error

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if self._get_reasoning_from_request(request) and (
                self._reasoning_detector is None
                or not self._reasoning_detector.thinks_internally
            ):
                # Models with thinks_internally=True think without a leading <think> token
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _patch_reasoning_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:
        """Keep parser-specific reasoning markers in the decoded text.

        Some reasoning parsers rely on special-token delimiters that would be
        removed during detokenization when ``skip_special_tokens=True``.
        """
        if self.reasoning_parser == "apertus2509":
            request.skip_special_tokens = False

        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False
        elif self.reasoning_parser == "inkling":
            request.skip_special_tokens = False

    def wrap_reasoning_history(self, reasoning_text: str) -> str:
        """Wrap prior-turn reasoning in the detector's own start/end tokens.

        Pulling the delimiters from the detector keeps adapters in lockstep
        with any future parser that ships non-``<think>`` markers - Mistral's
        ``[THINK]``, Gemma4's ``think_start_self_label = "thought\\n"``, etc.
        Falling back to a plain string is unsafe: it would let prior
        thinking text reach a non-reasoning model as ordinary assistant
        content, so the caller must surface this state, not paper over it.
        """
        if self._reasoning_detector is None:
            raise ValueError(
                "Cannot rewrap thinking history: no reasoning detector is "
                "configured for this model"
            )
        d = self._reasoning_detector
        return (
            f"{d.think_start_token}{d.think_start_self_label}"
            f"{reasoning_text}\n{d.think_end_token}"
        )

    def _reasoning_default_mode(self) -> Optional[str]:
        if self._reasoning_detector is None:
            return None
        return self._reasoning_detector.reasoning_default

    def _get_reasoning_toggle_param(self) -> Optional[str]:
        """Resolve the chat-template kwarg that toggles reasoning, if any."""
        config = self.template_manager.reasoning_config
        if config is not None:
            return config.toggle_param

        mode = self._reasoning_default_mode()
        if mode in ("thinking", "enable_thinking"):
            return mode
        if mode in ("explicit_thinking", "explicit_enable_thinking"):
            return mode.replace("explicit_", "")
        return None

    def apply_reasoning_enabled(
        self, request: ChatCompletionRequest, enabled: bool
    ) -> None:
        """Force the request into the requested reasoning-on/off mode.

        Mirrors the read-side logic in ``_get_reasoning_from_request``;
        the two must stay in sync. Always-on models cannot be disabled,
        so explicit ``enabled=False`` raises rather than silently leaving
        reasoning on.
        """
        if not self.reasoning_parser:
            if enabled:
                raise ValueError(
                    "Anthropic thinking is not supported for models without "
                    "a reasoning parser"
                )
            return

        if self.reasoning_parser == "hunyuan":
            request.reasoning_effort = "medium" if enabled else "no_think"
            return

        config = self.template_manager.reasoning_config
        is_mistral = (config is not None and config.special_case == "mistral") or (
            config is None and self._reasoning_default_mode() == "mistral"
        )
        if is_mistral:
            request.reasoning_effort = "medium" if enabled else "none"
            return

        is_always_on = (config is not None and config.special_case == "always") or (
            config is None and self._reasoning_default_mode() == "always"
        )
        if is_always_on:
            if not enabled:
                raise ValueError(
                    f"Reasoning parser '{self.reasoning_parser}' is always-on "
                    f"and cannot be disabled via Anthropic thinking"
                )
            return

        toggle_param = self._get_reasoning_toggle_param()
        # The read side (``_get_reasoning_from_request``) returns False
        # whenever ``config.toggle_param is None`` OR
        # ``config.default_enabled is None``. The write side must mirror
        # both conditions: if ``default_enabled`` is unset we cannot
        # actually honor an ``enabled=True`` request even when the toggle
        # name itself is resolvable, so writing the kwarg would set up the
        # template to emit reasoning tokens while the parser ignores them
        # (literal ``<think>`` markers leak into the assistant text).
        config = self.template_manager.reasoning_config
        read_side_supported = toggle_param is not None and (
            config is None or config.default_enabled is not None
        )
        if not read_side_supported:
            if not enabled:
                return
            raise ValueError(
                f"Anthropic thinking is not supported for reasoning parser "
                f"'{self.reasoning_parser}'"
            )

        chat_template_kwargs = dict(request.chat_template_kwargs or {})
        chat_template_kwargs[toggle_param] = enabled
        request.chat_template_kwargs = chat_template_kwargs

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """Determine whether reasoning mode should be enabled for this request.

        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False

        if self.reasoning_parser == "minimax-m3":
            # M3 template prefills <mm:think> for thinking_mode=enabled, so it never
            # appears in output and reasoning must be forced. Mirrors reasoning_parser.py.
            return (request.chat_template_kwargs or {}).get(
                "thinking_mode"
            ) == "enabled"

        if self.reasoning_parser == "hunyuan":
            # Hy3-preview template emits no <think> when reasoning_effort is
            # "no_think" / "none" / unset; forcing reasoning would route all
            # output into reasoning_content.
            return request.reasoning_effort not in (None, "none", "no_think")

        config = self.template_manager.reasoning_config
        if config is None:
            # Fallback to parser-level defaults when template toggle config
            # cannot be inferred (e.g., parser-only <think> templates).
            mode = (
                self._reasoning_detector.reasoning_default
                if self._reasoning_detector is not None
                else None
            )
            if mode is None:
                return False
            if mode == "always":
                return True
            if mode == "mistral":
                return (
                    request.reasoning_effort is not None
                    and request.reasoning_effort != "none"
                )
            if mode in ("thinking", "enable_thinking"):
                return (
                    not request.chat_template_kwargs
                    or request.chat_template_kwargs.get(mode) is not False
                )
            if mode in ("explicit_thinking", "explicit_enable_thinking"):
                toggle = mode.replace("explicit_", "")
                return (
                    request.chat_template_kwargs is not None
                    and request.chat_template_kwargs.get(toggle) is True
                )
            logger.warning(
                "Unknown reasoning_default mode '%s', defaulting to reasoning disabled",
                mode,
            )
            return False

        if config.special_case == "always":
            return True

        if config.special_case == "mistral":
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )

        if config.toggle_param is None or config.default_enabled is None:
            return False

        if config.default_enabled:
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get(config.toggle_param) is not False
            )
        return (
            request.chat_template_kwargs is not None
            and request.chat_template_kwargs.get(config.toggle_param) is True
        )
