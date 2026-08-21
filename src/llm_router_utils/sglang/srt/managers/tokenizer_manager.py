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
# Derivative work: slimmed for llm-router-utils. See HOW_TO_UPGRADE.md.
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
