"""Stub for sglang.srt.utils.runai_utils.

RunAI object storage is an inference-engine feature and is not supported
in llm-router-utils.  The stub exists so that modules copied verbatim
from upstream sglang (hf_transformers/common.py, config.py, tokenizer.py)
can import ``is_runai_obj_uri`` / ``ObjectStorageModel`` without
modification; ``is_runai_obj_uri`` always returns False and
``ObjectStorageModel`` raises NotImplementedError on instantiation.
"""
from __future__ import annotations


def is_runai_obj_uri(*args, **kwargs):
    return False


class ObjectStorageModel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RunAI object storage is not supported in llm-router-utils."
        )


__all__ = ["is_runai_obj_uri", "ObjectStorageModel"]
