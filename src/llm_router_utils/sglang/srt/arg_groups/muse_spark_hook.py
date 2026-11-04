"""Stub for sglang.srt.arg_groups.muse_spark_hook.

Upstream applies Muse Spark launcher defaults (graph-pool/KV-sizing switches)
and derives ``--chat-template`` from the checkpoint before the server starts.
Neither is reachable from this repo: there is no scheduler, no CUDA graph pool,
and no ``ServerArgs._model_config`` attached by a launch pipeline.

:func:`muse_spark_chat_template_default` therefore reports "nothing to default",
which is exactly what upstream returns when ``--chat-template`` is already set
or when the architecture is not Muse Spark.
"""

from __future__ import annotations

from typing import Any

MUSE_SPARK_ARCHITECTURES = ("MuseSparkForConditionalGeneration",)

MUSE_SPARK_CHAT_TEMPLATE_FILE = "chat_template.jinja"


def muse_spark_chat_template_default(server_args: Any) -> str | None:
    """Return the checkpoint chat template for Muse Spark, else ``None``."""
    return None


def apply_muse_spark_env_defaults(server_args: Any) -> None:
    """Accept the call; environment defaults are a launcher concern."""


__all__ = [
    "MUSE_SPARK_ARCHITECTURES",
    "MUSE_SPARK_CHAT_TEMPLATE_FILE",
    "apply_muse_spark_env_defaults",
    "muse_spark_chat_template_default",
]
