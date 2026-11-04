# Copyright 2023-2026 SGLang Team
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
# ==============================================================================
# Derivative work: slimmed for llm-router-utils. See DEVELOPMENT.md.
# Original copyright notice retained per Apache 2.0 §4(b)/§4(c).
# ==============================================================================
"""Minimal stand-in for sglang.srt.runtime_context.

Upstream's ``runtime_context`` is a 2,200-line server-args projection layer
(arg_groups namespaces, role-scoped reads, Dynamo-friendly config bags) used by
the scheduler, the model executor and the HTTP server. None of that machinery
survives this repo's extraction: the retained serving layer only needs
read-only access to a handful of *serving* settings and *model* attributes.

This module therefore exposes the same two accessors ``serving_chat`` uses —
:func:`get_serving` and :func:`get_model` — backed by plain attribute proxies
over the ``ServerArgs`` and ``ModelConfig`` this repo already carries.

Both accessors resolve lazily from the process-level context that
``set_runtime_context`` installs, falling back to defaults so that importing the
package never requires a server to be configured.
"""

from __future__ import annotations

from typing import Any, Optional

_SERVER_ARGS: Optional[Any] = None
_MODEL_CONFIG: Optional[Any] = None


def set_runtime_context(
    *, server_args: Optional[Any] = None, model_config: Optional[Any] = None
) -> None:
    """Install the process-level config used by :func:`get_serving`/:func:`get_model`."""
    global _SERVER_ARGS, _MODEL_CONFIG
    if server_args is not None:
        _SERVER_ARGS = server_args
    if model_config is not None:
        _MODEL_CONFIG = model_config


class _ServingConfig:
    """Read-only view of the serving-related ``ServerArgs`` fields.

    Reads fall through to the installed ``ServerArgs`` so that *any* field the
    retained layer asks for resolves, exactly as upstream's namespace bag does.
    Maintaining a hand-written allowlist here is what previously dropped
    ``hf_chat_template_name`` and silently cost models their HF chat template.
    """

    __slots__ = ("_args",)

    #: Values used only when no ServerArgs has been installed yet.
    _DEFAULTS = {
        "allow_auto_truncate": False,
        "default_chat_template_kwargs": None,
        "enable_cache_report": False,
        "hf_chat_template_name": None,
        "incremental_streaming_output": False,
        "return_input_ids": False,
        "return_output_ids": False,
        "stream_response_default_include_usage": False,
        "trust_request_chat_template": False,
    }

    def __init__(self, args: Optional[Any]) -> None:
        self._args = args

    def __getattr__(self, name: str) -> Any:
        args = object.__getattribute__(self, "_args")
        if args is not None and hasattr(args, name):
            return getattr(args, name)
        try:
            return _ServingConfig._DEFAULTS[name]
        except KeyError:
            raise AttributeError(name) from None


class _ModelConfigView:
    """Read-only view of the model attributes ``serving_chat`` reads."""

    __slots__ = ("_config",)

    def __init__(self, config: Optional[Any]) -> None:
        self._config = config

    def __getattr__(self, name: str) -> Any:
        config = object.__getattribute__(self, "_config")
        if config is None:
            raise AttributeError(
                f"No ModelConfig installed in the runtime context; cannot read {name!r}"
            )
        return getattr(config, name)


def get_serving() -> _ServingConfig:
    return _ServingConfig(_SERVER_ARGS)


def get_model() -> _ModelConfigView:
    return _ModelConfigView(_MODEL_CONFIG)
