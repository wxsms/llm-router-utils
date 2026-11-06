"""Lightweight OpenAIServingBase — the base-class surface OpenAIServingChat uses.

Upstream's ``serving_base.py`` is a 300-line ABC that also owns HTTP request
handling (``handle_request``, the streaming/non-streaming dispatch, FastAPI
``Request`` plumbing). llm-router-utils does not perform HTTP serving, so those
entry points are dropped.

What is **not** dropped is the helper surface a faithfully-retained
``serving_chat.py`` calls into: the error-response builders, the request-id
helpers, and the header/LoRA extractors used by ``_convert_to_internal_request``.
An earlier version kept only ``__init__``, which made ``serving_chat.py`` (a
byte-identical upstream file) raise ``AttributeError`` on those calls.

``handle_request`` and the two ``_handle_*_request`` dispatchers remain absent
because they need a live HTTP layer; every method defined here is self-contained.

Derivative work: slimmed for llm-router-utils. See DEVELOPMENT.md.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse

from llm_router_utils.sglang.srt.entrypoints.openai.protocol import (
    ErrorResponse,
    OpenAIServingRequest,
)

if TYPE_CHECKING:
    from llm_router_utils.sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

#: Observability knobs the header extractors read.  Upstream resolves these
#: through ``get_observability()`` (an arg_groups namespace this repo does not
#: publish); the defaults below are upstream's field defaults, so behaviour is
#: identical until a server config is published, which never happens here.
_DEFAULT_CUSTOM_LABELS_HEADER = "x-custom-labels"


class OpenAIServingBase:
    """Base class carrying the non-HTTP helpers ``OpenAIServingChat`` calls."""

    def __init__(self, tokenizer_manager: "TokenizerManager"):
        self.tokenizer_manager = tokenizer_manager
        # Upstream reads these from the published observability namespace and
        # only when ``server_args`` is a real ServerArgs instance.
        self.allowed_custom_labels: Optional[List[str]] = None
        self._custom_labels_header = _DEFAULT_CUSTOM_LABELS_HEADER

    # -- request ids -------------------------------------------------------
    def _request_id_prefix(self) -> str:
        """Override point; upstream subclasses customise this."""
        return ""

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> Optional[str]:
        """Return a request id if the request did not supply one."""
        if getattr(request, "rid", None):
            return None
        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    # -- model / LoRA ------------------------------------------------------
    def _parse_model_parameter(self, model: str) -> Tuple[str, Optional[str]]:
        """Split ``base:adapter`` into a base model and an optional LoRA path."""
        if getattr(self.tokenizer_manager.server_args, "enable_lora", False):
            # Keep backward-compatibility where it was using the delimiter ":".
            if ":" in model:
                base, lora = model.split(":", 1)
                if lora:
                    return base, lora
            return model, None
        return model, None

    def _resolve_lora_path(
        self, lora_path: Optional[str], model: Optional[str]
    ) -> Optional[str]:
        """Resolve the LoRA adapter path for a request."""
        if model:
            _, lora_path_from_model = self._parse_model_parameter(model)
            if lora_path_from_model:
                return lora_path_from_model
        return lora_path

    # -- error responses ---------------------------------------------------
    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response (upstream signature and return type)."""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response (upstream signature and shape)."""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})

    # -- header extractors -------------------------------------------------
    def extract_custom_labels(self, raw_request) -> Optional[Dict[str, str]]:
        """Read the allow-listed custom metric labels from request headers."""
        if (
            not self.allowed_custom_labels
            or not self._custom_labels_header
        ):
            return None

        header = self._custom_labels_header
        try:
            raw_labels = (
                orjson.loads(raw_request.headers.get(header))
                if raw_request and raw_request.headers.get(header)
                else None
            )
        except json.JSONDecodeError as e:
            logger.exception("Error in request: %s", e)
            raw_labels = None

        if isinstance(raw_labels, dict):
            return {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return None

    def extract_routing_key(self, raw_request) -> Optional[str]:
        if raw_request is None:
            return None
        return raw_request.headers.get("x-smg-routing-key")

    def extract_routed_dp_rank_from_header(
        self, raw_request: Request, body_routed_dp_rank: Optional[int] = None
    ) -> Optional[int]:
        """Read ``X-Data-Parallel-Rank``; it outranks the body's value."""
        if raw_request is None:
            return body_routed_dp_rank

        header_value = raw_request.headers.get("x-data-parallel-rank")
        if header_value is not None:
            try:
                header_dp_rank = int(header_value)
                if (
                    body_routed_dp_rank is not None
                    and header_dp_rank != body_routed_dp_rank
                ):
                    logger.debug(
                        "X-Data-Parallel-Rank header (%s) overrides body "
                        "routed_dp_rank (%s)",
                        header_dp_rank,
                        body_routed_dp_rank,
                    )
                return header_dp_rank
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid X-Data-Parallel-Rank header: must be an integer, "
                        f"got '{header_value}'"
                    ),
                )

        return body_routed_dp_rank
