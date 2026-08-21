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
"""Monkey-patches on transformers internals.

Slimmed for llm-router-utils: only the on-demand helpers
(``normalize_rope_scaling_compat``, ``_ensure_gguf_version``) and the
idempotent ``apply_all`` entry point are retained.  All torch-dependent
and multimodal-specific patches from upstream have been removed because
llm-router-utils is a frontend-only extraction and never imports torch.

Import this module early (before any ``from_pretrained`` call) to activate
all patches.  It is safe to import multiple times -- patches are idempotent.
"""

from llm_router_utils.sglang.srt.utils import logger

_applied = False


# ---------------------------------------------------------------------------
# Public API: apply_all() -- import-time patches (idempotent)
# ---------------------------------------------------------------------------


def apply_all():
    """Apply all transformers compatibility patches (idempotent).

    Call this once at import time.  It is safe to call multiple times.

    No-op when the ``transformers`` package is not installed -- frontend-only
    sglang users should not be forced to install transformers just to import
    the top-level ``sglang`` package.

    Slimmed vs upstream: the torch-dependent / multimodal-specific patches
    have been removed because llm-router-utils never imports torch.  Only the
    idempotent guard and transformers-availability check remain.
    """
    global _applied
    if _applied:
        return
    try:
        import transformers  # noqa: F401
    except ImportError:
        _applied = True
        return
    _applied = True

    logger.debug("transformers compatibility patches applied (slimmed)")


# ---------------------------------------------------------------------------
# Public API: on-demand helpers (called explicitly by other modules)
# ---------------------------------------------------------------------------


def normalize_rope_scaling_compat(config) -> None:
    """Ensure rope_scaling dicts have ``"type"`` alongside ``"rope_type"``.

    Transformers v5 standardises rope_scaling to use ``"rope_type"`` and may
    omit the legacy ``"type"`` key.  Remote-code models (e.g. Kimi-VL) still
    read ``rope_scaling["type"]``, causing a ``KeyError``.  This helper adds
    ``"type"`` from ``"rope_type"`` whenever it is missing, recursively across
    the config and all its sub-configs.
    """

    def _patch(cfg):
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        # Recurse into sub-configs
        for attr in (
            "text_config",
            "llm_config",
            "language_config",
            "vision_config",
            "thinker_config",
        ):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                _patch(sub)

    _patch(config)


def _ensure_gguf_version():
    """Workaround for transformers v5 bug where is_gguf_available() fails
    when the gguf package lacks __version__ and metadata lookup also fails,
    resulting in packaging.version.InvalidVersion: Invalid version: 'N/A'."""
    try:
        import gguf

        if not hasattr(gguf, "__version__"):
            import importlib.metadata

            try:
                gguf.__version__ = importlib.metadata.version("gguf")
            except importlib.metadata.PackageNotFoundError:
                gguf.__version__ = "0.0.0"
            except (ValueError, OSError, TypeError) as e:
                logger.warning(
                    "Failed to determine gguf package version: %s. "
                    "Falling back to '0.0.0'.",
                    e,
                )
                gguf.__version__ = "0.0.0"
    except ImportError:
        pass
