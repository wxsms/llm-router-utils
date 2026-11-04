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
"""Slimmed sglang.srt.arg_groups.overrides.

Upstream records, per server arg, *why* a value ended up as it did (CLI, env
var, model override, detection, default) in a ``_resolved_overrides`` stash on
the ``ServerArgs`` instance, and reads it back through
:func:`resolution_result` so downstream components prefer the authoritative
decision over the raw input.

The retained parser layer only *declares* detections and *reads* results, so the
stash and both accessors are carried here verbatim in behaviour. The
runtime-context publication machinery, ``with_fallback`` field projection and
the plugin/foreign-default recorder are server-launch concerns and are not
carried.

Derivative work: slimmed for llm-router-utils.
"""

from __future__ import annotations

from typing import Any


class ResolvingView:
    """Live view of declarations over raw inputs.

    Mirrors upstream's ``ResolvingConfig``: every read replays the
    ``_resolved_overrides`` stash, so declarations made *after* the view was
    constructed are still observed. A plain passthrough proxy would silently
    hide ``resolve_auto_parsers``' decisions from the callers that consult a
    view — the exact drift this module exists to avoid.
    """

    __slots__ = ("_target",)

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        for _source, declared in reversed(
            getattr(target, "_resolved_overrides", None) or ()
        ):
            if name in declared:
                return declared[name]
        return getattr(target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ResolvingView is read-only; resolution writes through declarations"
        )


def resolving_view(target: Any) -> ResolvingView:
    """A live read view of what resolution has decided so far."""
    return ResolvingView(target)


def declare_resolution(server_args: Any, source: str, **fields: Any) -> None:
    """Append field decisions without changing raw inputs."""
    stash = getattr(server_args, "_resolved_overrides", None)
    if stash is None:
        stash = []
        try:
            server_args._resolved_overrides = stash
        except AttributeError:
            return
    stash.append((source, dict(fields)))


def resolution_result(server_args: Any, field: str, default: Any = None) -> Any:
    """Read the last declaration, then the raw attribute, then ``default``."""
    for _source, declared in reversed(
        getattr(server_args, "_resolved_overrides", None) or ()
    ):
        if field in declared:
            return declared[field]
    return getattr(server_args, field, default)


__all__ = [
    "ResolvingView",
    "declare_resolution",
    "resolution_result",
    "resolving_view",
]
