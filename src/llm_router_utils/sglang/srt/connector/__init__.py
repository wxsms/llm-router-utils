"""Stub for sglang.srt.connector.

Remote storage connectors are an inference-engine feature and are not
supported in llm-router-utils.  The stub exists so that modules copied
verbatim from upstream sglang (hf_transformers/config.py, tokenizer.py)
can import ``create_remote_connector`` without modification; calling it
raises NotImplementedError.
"""
from __future__ import annotations


def create_remote_connector(*args, **kwargs):
    raise NotImplementedError(
        "Remote storage connectors are not supported in llm-router-utils. "
        "This is an inference-engine feature."
    )


__all__ = ["create_remote_connector"]
