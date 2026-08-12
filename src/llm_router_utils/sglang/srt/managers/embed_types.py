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
