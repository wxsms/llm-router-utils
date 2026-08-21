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
"""Lightweight embed_types — placeholder for PositionalEmbeds.

The original PositionalEmbeds depends on torch + msgspec. Since llm-router-utils
does not perform inference, positional embed overrides are not supported.
This module exists only so io_struct imports keep working.
"""
from __future__ import annotations

from typing import Any


class PositionalEmbeds:
    """Stub — not supported in llm-router-utils (no torch).

    Defined as a plain class so isinstance() checks fail gracefully
    rather than triggering ImportError.
    """

    def __init__(self, embeds: Any, positions: list):
        raise NotImplementedError(
            "PositionalEmbeds is not supported in llm-router-utils "
            "(requires torch, which is an inference-only dependency)."
        )
