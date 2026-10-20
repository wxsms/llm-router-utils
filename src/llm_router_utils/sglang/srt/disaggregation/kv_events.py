"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Derivative work: slimmed for llm-router-utils. See DEVELOPMENT.md.
# Original copyright notice retained per Apache 2.0 §4(b)/§4(c).

"""
KV cache event structs.

Extracted from sglang.srt.disaggregation.kv_events (structs first taken from
release/v0.5.16; kept in sync through release/v0.5.22). Only the msgspec Struct
definitions + StorageMedium enum are retained — the publisher
machinery (EventPublisher / NullEventPublisher / ZmqEventPublisher /
select_kv_publisher_dp_rank / OffloadedState) is stripped because router services
consume KV events rather than publish them.

Zero torch, zero sglang-internal dependencies.
"""

import enum
import hashlib
from typing import Any, Optional, Union

import msgspec


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: list[Any]
    attn_dp_rank: Optional[int] = None


class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


class StorageMedium(str, enum.Enum):
    """Storage tier for KV cache events."""

    GPU = "GPU"  # L1: device HBM
    CPU = "CPU_PINNED"  # L2: host pinned memory
    DISK = "DISK"  # L3: SSD / NVMe
    EXTERNAL = "EXTERNAL"  # L4: shared / remote pool (e.g. Mooncake)


class BlockStoredMetadata(msgspec.Struct, omit_defaults=True, gc=False):
    """Typed request metadata attached to a stored KV block."""

    cache_salt: str


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None
    # Salt of the request that stored these blocks. Block hashes are already
    # namespaced by it; consumers index the emitted hashes rather than
    # recompute them.
    cache_salt: Optional[str] = None
    # Session that triggered this store. Attribution only: the blocks may be
    # shared with other sessions, and the hash does not depend on it.
    session_id: Optional[str] = None
    # The LoRA adapter name of the request that stored these blocks.
    # Set only in the dynamo format.
    lora_name: Optional[str] = None


class BlockStoredWithMetadata(BlockStored, tag="BlockStored", kw_only=True):
    """BlockStored wire extension used only when typed metadata is present.

    A separate struct keeps unsalted events at their legacy array length; an
    optional field on BlockStored would still serialize a trailing null.
    """

    metadata: BlockStoredMetadata


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]
    medium: Optional[str] = None


class AllBlocksCleared(KVCacheEvent):
    pass


# The dynamo format publishes namespaced hashes as block hashes. If you change
# this tag, all of these hashes change. unified_tree_core.rs has a Rust copy.
_NAMESPACE_SEED_TAG = b"sglang-kv-event-namespace-v1"


def kv_event_namespace_seed(
    *, extra_key: Optional[str], cache_salt: Optional[str]
) -> Optional[bytes]:
    """Return the seed for namespaced_block_hash, or None for no namespace."""
    if extra_key is None and cache_salt is None:
        return None
    digest = hashlib.sha256(_NAMESPACE_SEED_TAG)
    # The presence byte and the length prefix make ("a", "bc") and ("ab", "c") differ.
    for part in (extra_key, cache_salt):
        if part is None:
            digest.update(b"\x00")
            continue
        encoded = part.encode("utf-8")
        digest.update(b"\x01" + len(encoded).to_bytes(8, "little") + encoded)
    return digest.digest()


def namespaced_block_hash(block_hash: int, *, namespace_seed: Optional[bytes]) -> int:
    """Return the namespaced hash of a published block hash.

    If namespace_seed is None, the result is ``block_hash``. If not, the result is
    the first 8 bytes of SHA-256(seed, block_hash as 8 big-endian bytes), as int64.
    """
    if namespace_seed is None:
        return block_hash
    digest = hashlib.sha256(
        namespace_seed + block_hash.to_bytes(8, "big", signed=True)
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


class KVEventBatch(EventBatch):
    # BlockStoredWithMetadata deliberately stays out of this tagged union.
    # Existing typed consumers decode its shared "BlockStored" tag as the base
    # type and ignore the trailing metadata; adding both types would give
    # msgspec duplicate tags and make the union invalid.
    events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]
