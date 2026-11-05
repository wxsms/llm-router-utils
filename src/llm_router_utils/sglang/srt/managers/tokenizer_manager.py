# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Derivative work: slimmed for llm-router-utils. See DEVELOPMENT.md.
# Original copyright notice retained per Apache 2.0 §4(b)/§4(c).
# ==============================================================================
"""Lightweight TokenizerManager extracted from sglang.srt.managers.tokenizer_manager.

Only the initializer and the attributes referenced by the _process_messages
call chain are kept. All scheduling/dispatch methods are removed.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from llm_router_utils.sglang.srt.configs.model_config import ModelConfig
from llm_router_utils.sglang.srt.runtime_context import set_runtime_context
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

# Fields the manager owns rather than reading from the resolved config tree.
_MANAGER_OWNED_FIELDS = ("model_path", "served_model_name")

#: Fields that exist upstream only as a *declaration* (no ServerArgs attribute).
#: ``serving_hook.handle_grammar_backend`` declares ``"xgrammar"`` when the
#: structured-output backend is enabled; the slimmed ServerArgs omits the field
#: entirely, so without this ``config_value("grammar_backend")`` is None and
#: ``serving_chat._template_turn_grammar()`` is permanently off.
_DECLARED_ONLY_DEFAULTS = {"grammar_backend": "xgrammar"}

#: The sentinel an operator passes to request auto-detection.
_AUTO_SENTINEL = "auto"

#: Fields where the sentinel is a request for detection, never a usable value.
_PARSER_FIELDS = ("reasoning_parser", "tool_call_parser")


class TokenizerManager:
    """Minimal TokenizerManager that loads a tokenizer + model config.

    Mirrors the public surface used by OpenAIServingChat._process_messages:
    exposes ``.tokenizer``, ``.processor``, ``.model_config``, ``.model_path``,
    ``.server_args``.
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args
        self.model_config: Optional[ModelConfig] = None
        self.tokenizer: Any = None
        self.processor: Any = None

        self.init_model_config()
        self.resolve_auto_parsers()
        if not server_args.skip_tokenizer_init:
            self.init_tokenizer_and_processor()

    def resolve_auto_parsers(self) -> None:
        """Resolve ``--*-parser auto`` into concrete parser names.

        Upstream does this once in ``entrypoints.engine`` before the tokenizer
        is built; that module does not exist here, so the resolution never
        happened and every ``auto`` field stayed the literal sentinel. A
        downstream ``ToolCallParserEnum.get("auto")`` is ``None``, which
        silently accepts tool calls and then never parses them.

        Runs before ``init_tokenizer_and_processor`` because detection loads a
        tokenizer of its own, and is best-effort: detection needs a reachable
        checkpoint, and a manager built from an unreachable path must still
        construct (the sentinel is then reported loudly by ``config_value``).
        """
        from llm_router_utils.sglang.srt.parser.template_detection import (
            resolve_auto_parsers,
        )

        try:
            resolve_auto_parsers(self.server_args)
        except Exception as exc:  # noqa: BLE001 - detection is best-effort
            logger.warning("Failed to resolve auto parsers: %s", exc)

    def config_value(self, name: str):
        """The value in effect for one config field.

        Upstream reads ``get_context().config_leaf(name)``, i.e. the *resolved*
        leaf, so a declaration made by detection wins over the raw input. The
        declaration stash is this repo's equivalent, so it must be consulted
        first: reading the dataclass field alone silently returned ``"auto"``
        for every ``--*-parser auto`` field and disabled tool-call parsing.

        ``grammar_backend`` only ever exists as a declaration upstream
        (``serving_hook.handle_grammar_backend``) and is absent from the slimmed
        ``ServerArgs``, hence the fallback table.
        """
        if name in _MANAGER_OWNED_FIELDS:
            return getattr(self, name)
        from llm_router_utils.sglang.srt.arg_groups.overrides import (
            resolution_result,
        )

        value = resolution_result(
            self.server_args, name, _DECLARED_ONLY_DEFAULTS.get(name)
        )
        if value == _AUTO_SENTINEL and name in _PARSER_FIELDS:
            # Returning "auto" here is not a value: ToolCallParserEnum.get()
            # yields None (tool calls accepted, then silently never parsed) and
            # ReasoningParser raises "Unsupported model type: auto". Failing
            # here names the real problem instead.
            raise ValueError(
                f"{name}={_AUTO_SENTINEL!r} was never resolved: automatic parser "
                "detection found no match for this checkpoint. Pass an explicit "
                f"--{name.replace('_', '-')} or use a checkpoint with a readable "
                "chat template."
            )
        return value

    def init_model_config(self) -> None:
        self.model_path = self.server_args.model_path
        # Upstream sets this on the manager (tokenizer_manager.py, from the
        # published serving config) and serving_chat reads it directly when it
        # renames the model for a reasoning-effort rewrite. ServerArgs defaults
        # it to model_path in __post_init__, so this is never None.
        self.served_model_name = self.server_args.served_model_name
        self.model_config = ModelConfig(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            revision=self.server_args.revision,
            context_length=self.server_args.context_length,
        )
        # Mirrors upstream TokenizerManager: read from the model config so the
        # dsv41 encoder can replace its image placeholder with the model's
        # image token.
        self.image_token_id = self.model_config.image_token_id
        # Upstream publishes the resolved config to a process-wide runtime
        # context before serving; that is what backs serving_chat's
        # get_model()/get_serving() reads. Register it so those accessors do
        # not fail closed for the dsv4/dsv41/inkling encoding adapters.
        #
        # NOTE: this is a process-level default. With several managers in one
        # process it is last-writer-wins, so a multi-model router that needs
        # per-instance isolation must wrap request handling in
        # ``use_runtime_context(...)``. See runtime_context's module docstring.
        set_runtime_context(
            server_args=self.server_args, model_config=self.model_config
        )

    def init_tokenizer_and_processor(self) -> None:
        from llm_router_utils.sglang.srt.utils.hf_transformers.tokenizer import (
            get_tokenizer,
        )

        tokenizer_path = self.server_args.tokenizer_path or self.server_args.model_path
        try:
            self.tokenizer = get_tokenizer(
                tokenizer_path,
                tokenizer_mode=self.server_args.tokenizer_mode,
                trust_remote_code=self.server_args.trust_remote_code,
                tokenizer_revision=getattr(self.server_args, "revision", None),
                tokenizer_backend=getattr(
                    self.server_args, "tokenizer_backend", "huggingface"
                ),
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
