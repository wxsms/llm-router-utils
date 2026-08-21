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
"""Lightweight io_struct — placeholder for GenerateReqInput/EmbeddingReqInput.

The original io_struct.py is 2282 lines with many inference-specific fields.
This slim version exists only so that imports in serving_base.py and
serving_chat.py keep working. Since llm-router-utils does not perform
inference, these classes are not actively constructed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class GenerateReqInput:
    """Minimal GenerateReqInput — only fields still referenced by retained code."""

    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    session_id: Optional[str] = field(default=None, kw_only=True)
    text: Optional[Union[List[str], str]] = None
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    image_data: Any = None
    video_data: Any = None
    audio_data: Any = None
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    return_logprob: Optional[Union[List[bool], bool]] = None
    logprob_start_len: Optional[Union[List[int], int]] = None
    top_logprobs_num: Optional[Union[List[int], int]] = None
    stream: bool = False
    modalities: Optional[List[str]] = None
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_host: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[Optional[int]], int]] = None
    routed_dp_rank: Optional[int] = None
    disagg_prefill_dp_rank: Optional[int] = None
    return_hidden_states: Union[List[bool], bool] = False
    return_routed_experts: bool = False
    routed_experts_start_len: int = 0
    extra_key: Optional[Union[List[str], str]] = None
    require_reasoning: bool = False
    priority: Optional[int] = None
    conversation_id: Optional[str] = None
    # Inference-time fields kept as Any for compatibility
    received_time: Any = None


@dataclass
class EmbeddingReqInput:
    """Minimal EmbeddingReqInput — only fields still referenced by retained code."""

    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    text: Optional[Union[List[List[str]], List[str], str]] = None
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    image_data: Any = None
    video_data: Any = None
    audio_data: Any = None
    sampling_params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    modalities: Optional[List[str]] = None
    is_cross_encoder_request: bool = False
    lora_path: Optional[Union[List[Optional[str]], str]] = None
    routing_key: Optional[str] = None
    received_time: Any = None
