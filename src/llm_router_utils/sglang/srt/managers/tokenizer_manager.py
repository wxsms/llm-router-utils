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
