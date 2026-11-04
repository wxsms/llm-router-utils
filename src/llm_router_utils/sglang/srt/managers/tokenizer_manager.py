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
from llm_router_utils.sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

# Fields the manager owns rather than reading from the resolved config tree.
_MANAGER_OWNED_FIELDS = ("model_path", "served_model_name")


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
        if not server_args.skip_tokenizer_init:
            self.init_tokenizer_and_processor()

    def config_value(self, name: str):
        """The value in effect for one config field.

        Upstream reads through the published runtime context
        (``get_context().config_leaf(name)``); this repo has no publication
        layer, so the slimmed ``ServerArgs`` dataclass *is* the resolved config.
        """
        if name in _MANAGER_OWNED_FIELDS:
            return getattr(self, name)
        if hasattr(self.server_args, name):
            return getattr(self.server_args, name)
        from llm_router_utils.sglang.srt.arg_groups.overrides import (
            resolution_result,
        )

        return resolution_result(self.server_args, name)

    def init_model_config(self) -> None:
        self.model_path = self.server_args.model_path
        self.model_config = ModelConfig(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            context_length=self.server_args.context_length,
        )
        # Mirrors upstream TokenizerManager: read from the model config so the
        # dsv41 encoder can replace its image placeholder with the model's
        # image token.
        self.image_token_id = self.model_config.image_token_id

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
