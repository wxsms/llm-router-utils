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

Upstream's ``runtime_context`` is a ~2,200-line server-args projection layer
(arg_groups namespaces, role-scoped reads, Dynamo-friendly config bags) used by
the scheduler, the model executor and the HTTP server. None of that machinery
survives this repo's extraction: the retained serving layer only needs
read-only access to a handful of *serving* settings and *model* attributes.

Two accessors are exposed — :func:`get_serving` and :func:`get_model` — backed by
attribute proxies over the ``ServerArgs`` and ``ModelConfig`` this repo already
carries.

Scoping
-------
A router serves several models from one process, and it constructs one
``OpenAIServingChat`` (and one ``TokenizerManager``) per model. A single
process-global slot would make the last-constructed model win, silently serving
model A with model B's settings. So there are two levels:

* :func:`set_runtime_context` registers the **process default** (used for
  import-time reads and single-model services);
* :func:`use_runtime_context` binds a **scope** (a ``ContextVar``) that both
  accessors prefer while active.

A multi-model router must enter :func:`use_runtime_context` around the
*construction* of each model's serving objects. Nothing does this automatically:
``OpenAIServingChat`` is kept byte-identical to upstream and never opens a scope
itself, so without an explicit scope every instance reads whichever manager was
constructed last.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_UNSET = object()

#: Process-level default, registered by TokenizerManager.
_SERVER_ARGS: Optional[Any] = None
_MODEL_CONFIG: Optional[Any] = None

#: Per-scope binding; prefers over the process default while set.
_SCOPED: ContextVar[Optional[tuple]] = ContextVar("sglang_runtime_context", default=None)


def set_runtime_context(
    *,
    server_args: Any = _UNSET,
    model_config: Any = _UNSET,
) -> None:
    """Register the process-level default config.

    Pass ``None`` explicitly to clear a slot. Omitting an argument leaves it
    unchanged — so cleanup code must pass ``None`` (it used to be a no-op).
    """
    global _SERVER_ARGS, _MODEL_CONFIG
    if server_args is not _UNSET:
        _SERVER_ARGS = server_args
    if model_config is not _UNSET:
        _MODEL_CONFIG = model_config


def reset_runtime_context() -> None:
    """Drop the process-level default entirely (test teardown)."""
    global _SERVER_ARGS, _MODEL_CONFIG
    _SERVER_ARGS = None
    _MODEL_CONFIG = None


@contextmanager
def use_runtime_context(
    *, server_args: Any = _UNSET, model_config: Any = _UNSET
) -> Iterator[None]:
    """Bind config for the duration of the block, restoring on exit.

    An omitted argument inherits the process default rather than clearing it,
    matching :func:`set_runtime_context`. Pass ``None`` explicitly to blank a
    slot for the block. (Previously both defaulted to ``None``, so scoping only
    ``server_args`` silently hid ``model_config`` and ``get_model()`` started
    failing closed.)
    """
    current = _SCOPED.get()
    inherited_args = current[0] if current is not None else _SERVER_ARGS
    inherited_model = current[1] if current is not None else _MODEL_CONFIG
    bound_args = inherited_args if server_args is _UNSET else server_args
    bound_model = inherited_model if model_config is _UNSET else model_config
    token = _SCOPED.set((bound_args, bound_model))
    try:
        yield
    finally:
        _SCOPED.reset(token)


def _resolve(name: str) -> Any:
    scoped = _SCOPED.get()
    if scoped is not None:
        return scoped[0] if name == "server_args" else scoped[1]
    return _SERVER_ARGS if name == "server_args" else _MODEL_CONFIG


class _ServingConfig:
    """Read-only view of the serving-related ``ServerArgs`` fields.

    Reads fall through to the bound ``ServerArgs`` so that *any* field the
    retained layer asks for resolves, exactly as upstream's namespace bag does.
    A hand-written allowlist here previously dropped ``hf_chat_template_name``
    and silently cost models their HF chat template.
    """

    __slots__ = ("_args",)

    #: Values used only when no ServerArgs has been bound.
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
    return _ServingConfig(_resolve("server_args"))


def get_model() -> _ModelConfigView:
    return _ModelConfigView(_resolve("model_config"))
