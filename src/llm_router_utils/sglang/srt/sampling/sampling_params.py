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
# Derivative work: slimmed for llm-router-utils.
#
# Upstream declares the full ``SamplingParams`` struct here plus watermark
# validation, which drags in the sampler/watermarking runtime (and torch). This
# repo only renders prompts, so the file keeps the private request-transport
# helpers ``serving_chat`` uses to record how a request was rendered.
"""Sampling parameters for text generation."""

from typing import Any, Dict, List, Optional

# Private transport key: the renderer-selected reasoning terminator.
REQUEST_REASONING_END_TOKEN_IDS_KEY = "__sglang_reasoning_end_token_ids"

# Private transport key: the request's grammar also describes the reasoning, so
# it applies from the first output token.
REQUEST_REASONING_IN_GRAMMAR_KEY = "__sglang_reasoning_in_grammar"


def get_request_reasoning_end_token_ids(
    custom_params: Optional[Dict[str, Any]],
) -> Optional[List[int]]:
    if not isinstance(custom_params, dict):
        return None
    token_ids = custom_params.get(REQUEST_REASONING_END_TOKEN_IDS_KEY)
    return list(token_ids) if token_ids is not None else None


def set_request_reasoning_end_token_ids(
    sampling_params: Dict,
    token_ids: Optional[List[int]],
) -> None:
    """Attach the renderer-selected reasoning terminator to a request."""
    if token_ids is None:
        return
    if not token_ids or any(
        type(token_id) is not int or token_id < 0 for token_id in token_ids
    ):
        raise ValueError("reasoning end token IDs must be non-empty integers")
    custom_params = dict(sampling_params.get("custom_params") or {})
    custom_params[REQUEST_REASONING_END_TOKEN_IDS_KEY] = list(token_ids)
    sampling_params["custom_params"] = custom_params


def get_request_reasoning_in_grammar(
    custom_params: Optional[Dict[str, Any]],
) -> bool:
    return (
        isinstance(custom_params, dict)
        and custom_params.get(REQUEST_REASONING_IN_GRAMMAR_KEY) is True
    )


def set_request_reasoning_in_grammar(sampling_params: Dict, enabled: bool) -> None:
    """Mark a request whose grammar also describes the reasoning."""
    if not enabled:
        return
    custom_params = dict(sampling_params.get("custom_params") or {})
    custom_params[REQUEST_REASONING_IN_GRAMMAR_KEY] = True
    sampling_params["custom_params"] = custom_params
