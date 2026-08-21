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
"""Lightweight metrics_collector — data classes for scheduler stats.

Extracted from sglang.srt.observability.metrics_collector (release/v0.5.16).
Only the pure-data symbols are retained:
- ``QueueCount``
- ``SchedulerStats``
- ``compute_routing_key_stats``

Stripped (inference-engine dependencies):
- ``SchedulerMetricsCollector`` / ``TokenizerMetricsCollector`` / ``StorageMetricsCollector``
  / ``ExpertDispatchCollector`` / ``RadixCacheMetricsCollector`` / ``EncoderMetricsCollector``
  (depend on prometheus_client, GaugeHistogram, forward_batch_info, schedule_batch.Req)
- ``DPCooperationInfo`` (depends on ForwardMode)
- ``resolve_collector_class`` (depends on ServerArgs.stat_loggers, which
  llm-router-utils' slimmed ServerArgs does not expose)
- ``ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS`` (only used by removed collectors)

``QueueCount.from_reqs`` is kept with ``Req`` referenced only under
``TYPE_CHECKING`` — at runtime it duck-types ``req.priority``, so callers
can pass any object exposing that attribute (or reuse the function with
real sglang ``Req`` objects if they happen to have them).

Zero torch, zero sglang-internal deps.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    # Type-only import — runtime duck-types `req.priority`.
    from llm_router_utils.sglang.srt.managers.schedule_batch import Req  # noqa: F401


@dataclass
class QueueCount:
    """Holds both the total count and optional per-priority breakdown for a queue."""

    total: int = 0
    by_priority: Optional[Dict[int, int]] = None

    @classmethod
    def from_reqs(cls, reqs: List["Req"], enable_priority_scheduling: bool = False):
        # NOTE: If requests have priority=None (no --default-priority-value set),
        # Counter will produce {None: N}, resulting in priority="None" Prometheus labels.
        # Set --default-priority-value when enabling priority scheduling to avoid this.
        by_priority = (
            dict(Counter(req.priority for req in reqs))
            if enable_priority_scheduling
            else None
        )
        return cls(total=len(reqs), by_priority=by_priority)


@dataclass
class SchedulerStats:
    # Basics
    num_running_reqs: QueueCount = field(default_factory=QueueCount)
    num_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_grammar_queue_reqs: int = 0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    decode_sum_seq_lens: int = 0

    # Memory pool usage ratios (0.0–1.0).
    # Each pool tracks: used = total - available - evictable, usage = used / total.
    #
    # token_usage:      max(full, swa, mamba) — the bottleneck across all pools.
    #                   FIXME: misleadingly named "token_usage"; rename requires API deprecation.
    # full_token_usage: full-attention KV cache pool usage (always active).
    # swa_token_usage:  sliding-window attention KV cache pool usage (hybrid SWA models only, e.g. Gemma2).
    # mamba_usage:      Mamba SSM state pool usage (hybrid SSM models only, e.g. Jamba).
    token_usage: float = 0.0
    full_token_usage: float = 0.0
    swa_token_usage: float = 0.0
    mamba_usage: float = 0.0

    # Absolute token counts for the full-attention KV cache pool.
    # Invariant: kv_available_tokens + kv_evictable_tokens + kv_used_tokens <= max_total_num_tokens
    # (the gap accounts for protected/session-held tokens not exposed here).
    # max_total_num_tokens is emitted once at startup via emit_constants.
    #
    # kv_available_tokens:  free (unallocated) slots in the pool.
    # kv_evictable_tokens:  slots holding radix-cached KV data that can be evicted for new requests.
    # kv_used_tokens:       actively used slots (locked by running requests). Equals full_num_used.
    # num_used_tokens:      max(full_num_used, swa_num_used) for hybrid-SWA models, else full_num_used.
    #                       Does NOT include the mamba pool.
    num_used_tokens: int = 0
    kv_available_tokens: int = 0
    kv_evictable_tokens: int = 0
    kv_used_tokens: int = 0

    swa_available_tokens: int = 0
    swa_evictable_tokens: int = 0
    swa_used_tokens: int = 0
    mamba_available_tokens: int = 0
    mamba_evictable_tokens: int = 0
    mamba_used_tokens: int = 0

    # Speculative decoding
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    spec_cap_length: float = 0.0
    spec_block_accept_length: float = 0.0
    # Adaptive speculative decoding (currently active tier).
    spec_num_steps: int = 0
    spec_num_draft_tokens: int = 0

    # Retract
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    # PD disaggregation
    num_prefill_bootstrap_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_prefill_inflight_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_decode_prealloc_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_decode_transfer_queue_reqs: QueueCount = field(default_factory=QueueCount)
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0
    pending_prealloc_token_usage: float = 0.0

    # Utilization
    utilization: float = 0.0
    fwd_occupancy: float = float("nan")

    # Scheduler policy
    new_token_ratio: float = 0.0

    # CUDA graph
    is_cuda_graph: int = 0

    # LoRA pool metrics
    lora_pool_slots_used: int = 0
    lora_pool_slots_total: int = 0
    lora_pool_utilization: float = 0.0

    # HiCache metrics
    hicache_host_used_tokens: int = 0
    hicache_host_total_tokens: int = 0

    # Streaming session metrics
    num_streaming_sessions: int = 0
    streaming_session_held_tokens: int = 0

    # Routing key metrics
    num_unique_running_routing_keys: int = 0
    routing_key_running_req_counts: List[int] = field(default_factory=list)
    routing_key_all_req_counts: List[int] = field(default_factory=list)


def compute_routing_key_stats(routing_keys: List[Optional[str]]) -> tuple:
    """Returns (num_unique_keys, per_key_counts)."""
    from collections import Counter

    key_counts = Counter(k for k in routing_keys if k is not None)
    return len(key_counts), list(key_counts.values())
