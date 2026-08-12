"""Lightweight OpenAIServingBase — minimal base class for OpenAIServingChat.

The original serving_base.py contains HTTP request handling logic (FastAPI,
StreamingResponse, error responses) which llm-router-utils does not need.
This slim version provides only the ``__init__`` storing ``tokenizer_manager``
so that ``OpenAIServingChat`` can inherit it.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIServingBase:
    """Minimal base class — stores tokenizer_manager only.

    The full sglang version is an ABC with abstract methods for HTTP request
    handling. llm-router-utils does not perform HTTP serving, so those are
    dropped. Subclasses (e.g. OpenAIServingChat) inherit only ``__init__``.
    """

    def __init__(self, tokenizer_manager: "TokenizerManager"):
        self.tokenizer_manager = tokenizer_manager
