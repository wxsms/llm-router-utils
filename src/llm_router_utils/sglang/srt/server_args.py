"""Lightweight ServerArgs/PortArgs extracted from sglang.srt.server_args.

Only fields referenced by the _process_messages call chain are kept.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ServerArgs:
    """Subset of sglang.srt.server_args.ServerArgs."""

    model_path: str
    tokenizer_path: Optional[str] = None
    chat_template: Optional[str] = None
    served_model_name: Optional[str] = None
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    context_length: Optional[int] = None
    default_chat_template_kwargs: Optional[dict] = None
    enable_cache_report: bool = False
    incremental_streaming_output: bool = False
    stream_response_default_include_usage: bool = False
    allow_auto_truncate: bool = False
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    tokenizer_mode: str = "auto"
    revision: Optional[str] = None
    tokenizer_backend: str = "huggingface"

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path


@dataclasses.dataclass
class PortArgs:
    """Minimal PortArgs — fields kept for signature compatibility."""

    tokenizer_ipc_name: str = ""
    scheduler_input_ipc_name: str = ""
    detokenizer_ipc_name: str = ""
    nccl_port: int = 0
    rpc_ipc_name: str = ""
    metrics_ipc_name: str = ""
    tokenizer_worker_ipc_name: Optional[str] = None
    load_collector_ipc_name: str = ""
    instance_id: str = ""
