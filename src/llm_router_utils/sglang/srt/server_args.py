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
# Derivative work: slimmed for llm-router-utils. See DEVELOPMENT.md.
# Original copyright notice retained per Apache 2.0 §4(b)/§4(c).
# ==============================================================================
"""Lightweight ServerArgs/PortArgs extracted from sglang.srt.server_args.

Only fields referenced by the _process_messages call chain are kept,
plus ``device`` and ``PortArgs.init_new`` for API compatibility with
callers that construct a TokenizerManager directly.
"""
from __future__ import annotations

import dataclasses
import socket
import tempfile
import uuid
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
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            # Upstream auto-detects via torch; we default to "cuda" as that's
            # the common case for router services backed by GPU engines.
            # Callers can pass "cpu" explicitly for CPU-only setups.
            self.device = "cuda"
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = self.model_path


def _get_free_port() -> int:
    """Bind to port 0, read the OS-assigned port, close, return it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    finally:
        sock.close()
    return port


@dataclasses.dataclass
class PortArgs:
    """Minimal PortArgs — fields kept for signature compatibility.

    Upstream's ``init_new`` supports DP-attention, multi-node, and decoupled
    speculative decoding. This slimmed version only supports the single-node
    non-DP path (sufficient for router services). Pass ``dp_rank=None``.
    """

    tokenizer_ipc_name: str = ""
    scheduler_input_ipc_name: str = ""
    detokenizer_ipc_name: str = ""
    nccl_port: int = 0
    rpc_ipc_name: str = ""
    metrics_ipc_name: str = ""
    tokenizer_worker_ipc_name: Optional[str] = None
    load_collector_ipc_name: str = ""
    decoupled_spec_ipc_config: Optional[object] = None
    instance_id: str = ""

    @staticmethod
    def init_new(
        server_args: "ServerArgs",
        dp_rank: Optional[int] = None,
        worker_ports: Optional[list] = None,
    ) -> "PortArgs":
        """Build a PortArgs for single-node, non-DP-attention setup.

        Mirrors the corresponding branch of upstream ``PortArgs.init_new``.
        DP-attention / multi-node / decoupled speculative decoding are not
        supported here and will raise if the caller opts into them.
        """
        if getattr(server_args, "enable_dp_attention", False):
            raise NotImplementedError(
                "DP-attention is not supported in llm-router-utils' PortArgs.init_new. "
                "Construct PortArgs() directly for non-default topologies."
            )
        if getattr(server_args, "decoupled_spec_role", "null") != "null":
            raise NotImplementedError(
                "Decoupled speculative decoding is not supported in llm-router-utils."
            )

        nccl_port = _get_free_port()
        tokenizer_worker_num = getattr(server_args, "tokenizer_worker_num", 1)
        if tokenizer_worker_num == 1:
            tokenizer_worker_ipc_name: Optional[str] = None
        else:
            tokenizer_worker_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

        return PortArgs(
            tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            nccl_port=nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            tokenizer_worker_ipc_name=tokenizer_worker_ipc_name,
            decoupled_spec_ipc_config=None,
            instance_id=uuid.uuid4().hex[:12],
        )
