"""Lightweight utils extracted from sglang.srt.utils.common.

Only contains symbols referenced by the _process_messages call chain,
plus helpers required by the hf_transformers tokenizer/config modules
(copied verbatim from upstream sglang/srt/utils/common.py).
"""
from __future__ import annotations

import functools
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import try_to_load_from_cache  # type: ignore
except ImportError:  # pragma: no cover
    try_to_load_from_cache = None  # type: ignore


@dataclass
class ImageData:
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"
    max_dynamic_patch: Optional[int] = None
    preprocess_kwargs: Optional[Dict] = None


@dataclass
class VideoData:
    url: str
    preprocess_kwargs: Optional[Dict] = None


def find_local_repo_dir(repo_id: str, revision: Optional[str] = None) -> Optional[str]:
    """Best-effort lookup of a local HF cache dir for ``repo_id``."""
    if try_to_load_from_cache is None:
        return None
    try:
        path = try_to_load_from_cache(repo_id, "config.json", revision=revision)
        if path is None or not os.path.exists(path):
            return None
        return str(Path(path).parent)
    except Exception:
        return None


def read_system_prompt_from_file(model_name: str) -> str:
    """Read SYSTEM_PROMPT.txt from the HuggingFace cache directory if present."""
    try:
        local_repo_dir = find_local_repo_dir(model_name)
        if local_repo_dir:
            system_prompt_file = os.path.join(local_repo_dir, "SYSTEM_PROMPT.txt")
            if os.path.exists(system_prompt_file):
                with open(system_prompt_file, "r", encoding="utf-8") as f:
                    return f.read()
        return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Helpers copied verbatim from upstream sglang/srt/utils/common.py
# (used by hf_transformers tokenizer/config modules).
# ---------------------------------------------------------------------------


_warned_bool_env_var_keys = set()


def get_bool_env_var(name: str, default: str = "false") -> bool:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    if (value not in truthy_values) and (value not in falsy_values):
        # Warn once per env var key (not per value), otherwise different keys that share the
        # same invalid value may suppress warnings incorrectly.
        if name not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) encountered unrecognized value={value} and will treat as false"
            )
        _warned_bool_env_var_keys.add(name)

    return value in truthy_values


@functools.lru_cache(maxsize=1)
def is_hip() -> bool:
    """ROCm/HIP detection. Torch-free: returns False when torch is absent.

    Upstream uses ``torch.version.hip is not None``; we lazily import torch so
    this module stays import-safe without a torch install.
    """
    try:
        import torch
    except Exception:
        return False
    return torch.version.hip is not None


@functools.lru_cache(maxsize=1)
def is_npu() -> bool:
    """NPU detection. Torch-free: returns False when torch is absent.

    Upstream calls ``torch.npu.is_available()``; we lazily import torch so
    this module stays import-safe without a torch install.
    """
    try:
        import torch
    except Exception:
        return False
    if not hasattr(torch, "npu"):
        return False

    if not torch.npu.is_available():
        raise RuntimeError(
            "torch_npu detected, but NPU device is not available or visible."
        )

    return True


@functools.lru_cache(None)
def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg)


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]


def is_remote_url(url: Union[str, Path]) -> bool:
    """
    Check if the URL is a remote URL of the format:
    <connector_type>://<host>:<port>/<model_name>
    """
    if isinstance(url, Path):
        return False

    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    return m is not None


def lru_cache_frozenset(maxsize=128):
    def _to_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            # Not hashable; convert based on type
            if isinstance(o, (dict)):
                return frozenset(
                    (_to_hashable(k), _to_hashable(v)) for k, v in o.items()
                )
            elif isinstance(o, set):
                return frozenset(_to_hashable(v) for v in o)
            elif isinstance(o, (list, tuple)) or (
                isinstance(o, Sequence) and not isinstance(o, (str, bytes))
            ):
                return tuple(_to_hashable(v) for v in o)
            else:
                raise TypeError(f"Cannot make hashable: {type(o)}")

    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            h_args = tuple(_to_hashable(a) for a in args)
            h_kwargs = frozenset(
                (_to_hashable(k), _to_hashable(v)) for k, v in kwargs.items()
            )
            key = (h_args, h_kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        wrapper.cache_clear = cache.clear  # For manual cache clearing
        return wrapper

    return decorator
