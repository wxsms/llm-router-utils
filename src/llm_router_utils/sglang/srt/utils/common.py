"""Lightweight utils extracted from sglang.srt.utils.common.

Only contains symbols referenced by the _process_messages call chain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

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
