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

        # Load HF config via upstream's get_config (byte-equivalent to sglang)
        from llm_router_utils.sglang.srt.utils.hf_transformers.config import get_config
        try:
            self.hf_config = get_config(
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
