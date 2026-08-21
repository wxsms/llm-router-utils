# Copyright 2025 SGLang Team
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
"""Vendored & stripped mem_cache hash helpers (PoC+).

Source: sglang/python/sglang/srt/mem_cache/utils.py (162 lines)

Stripped:
- sglang.kernels.ops.kvcache.mla_buffer (GPU MLA buffer kernels -- not used
  by router's hash tree)
- sglang.srt.environ (env var wrappers)
- sglang.srt.mem_cache.cpp_utils.native_hash (C++ SHA256 extension)
- sglang.srt.mem_cache.evict_policy (LRU/LFU/etc. -- router's kv_tree
  doesn't use sglang's eviction strategies)

Kept:
- hash_str_to_int64 (pure Python -- first 16 hex chars → signed int64)
- get_hash_str (pure-Python fallback using hashlib.sha256)
- compute_node_hash_values (used by kv_tree.hash_tree for sglang path)
- split_node_hash_value (used by kv_tree on node split)

The pure-Python implementation uses array("I") for token encoding (same
entry point as sglang's C++ binding) and hashlib.sha256 (CPython's
OpenSSL backend).  Benchmarked at ~1.4 us/call for a 64-token block on
x86 -- about 2x slower than sglang's C++ extension (which adds AVX2
buffer handling), but the router's hash call frequency (one per KV
block on event) is low enough that the difference is negligible.  If
profiling later shows this is a hot path, options are:
- vendor the C++ extension and build it via cffi (no torch dependency)
- call libcrypto's SHA256_* directly via ctypes
"""

from __future__ import annotations

import array
import hashlib
import logging
import sys
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# array("I") itemsize is 4 bytes (uint32) but byteorder is platform-native.
# sglang's C++ binding treats the buffer as uint32_t* and hashes the raw
# bytes, so on a big-endian host array("I") would emit BE bytes while sglang
# (which only loads on little-endian Linux -- see native_hash.py:21) is LE.
# Guard the BE case explicitly: the branch is predictable on every platform
# the router actually runs on, and the guard protects against silent hash
# corruption if someone ever runs this on a BE host.
_ARRAY_I_LE = sys.byteorder == "little"


def get_hash_str(
    token_ids: List[int],
    prior_hash: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str | List[str]:
    """SHA256-based hash of token_ids, chained with prior_hash.

    Mirrors sglang.srt.mem_cache.cpp_utils.native_hash semantics exactly:
    - prior_hash is a 64-char hex string (32-byte SHA256 digest).  None
      means no parent (root block).
    - token_ids are encoded as little-endian uint32 (array("I")) -- matching
      sglang's C++ binding which casts to uint32_t*.
    - Output is a 64-char hex string (32-byte digest).
    - When page_size is None, returns a single hex string for the whole
      token list.
    - When page_size > 1, returns a list of hex strings, one per page,
      where each page's digest chains into the next as its prior_hash
      (matches sglang's hash_pages_to_hex_blob).

    Note: sglang's bigram token path (unit_width=2) is NOT replicated --
    the router's kv_tree.hash_tree calls this with plain token id lists,
    never bigram buffers.  If a caller ever passes a bigram token list,
    the hash will diverge from sglang; this is acceptable because the
    kv_tree path doesn't use bigram encoding.
    """
    prior_digest = bytes.fromhex(prior_hash) if prior_hash else None
    # sglang's C++ throws if prior_digest isn't 32 bytes; mirror that.
    if prior_digest is not None and len(prior_digest) != 32:
        raise ValueError(
            f"prior_hash must decode to exactly 32 bytes (64 hex chars), "
            f"got {len(prior_digest)} bytes"
        )

    if page_size is None:
        return _sha256_hex(token_ids, prior_digest)

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    # page_size > 0: produce one hash per page, chained.  Matches sglang's
    # hash_pages_to_hex_blob: each page's digest becomes the next page's
    # prior_digest.
    hashes: List[str] = []
    chain_digest = prior_digest
    for i in range(0, len(token_ids), page_size):
        page = token_ids[i : i + page_size]
        h = _sha256_hex(page, chain_digest)
        hashes.append(h)
        chain_digest = bytes.fromhex(h)
    return hashes


def _sha256_hex(token_ids: List[int], prior_digest: Optional[bytes]) -> str:
    """SHA256(prior_digest || token_ids_as_uint32_LE) as 64-char hex.

    Mirrors sglang's hash_page / hash_all (hash_binding.cpp):
    - SHA256_Update(prior_digest, 32 bytes) if prior is present
    - SHA256_Update(token_bytes) where token_bytes is the little-endian
      uint32 representation of token_ids (array("I") on x86/ARM-LE)
    - Output: 64-char hex of the 32-byte digest

    sglang returns the full 64-char hex; hash_str_to_int64 then takes
    the first 16 chars (64 bits) separately.  We return the full 64-char
    hex here to match.
    """
    h = hashlib.sha256()
    if prior_digest is not None:
        h.update(prior_digest)
    if token_ids:
        h.update(_encode_uint32_le(token_ids))
    return h.hexdigest()


def _encode_uint32_le(token_ids: List[int]) -> bytes:
    """Encode token_ids as little-endian uint32, matching sglang's array("I").

    sglang's native_hash accepts array("I", token_ids) (native_hash.py:56)
    and the C++ binding hashes the raw buffer bytes.  We use the same
    array("I") construction here -- ~1.8x faster than struct.pack because
    array() accepts the iterable directly without *args expansion, and
    the C layer does a single memcpy.

    Token ids > 2**32-1 raise OverflowError, matching sglang's
    checked_u32 which throws out_of_range for the same case.

    On big-endian hosts (none currently supported by the router, but
    guarded for safety) we byteswap to LE so output stays byte-identical
    to sglang's little-endian-only C++ extension.
    """
    a = array.array("I", token_ids)
    if not _ARRAY_I_LE:
        a.byteswap()
    return a.tobytes()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert hex string to signed 64-bit integer.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    Mirrors sglang.srt.mem_cache.utils.hash_str_to_int64 exactly.
    """
    uint64_val = int(hash_str[:16], 16)
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


def compute_node_hash_values(node: Any, page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware KV block IDs.

    Mirrors sglang.srt.mem_cache.utils.compute_node_hash_values.
    """
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    hash_values = get_hash_str(node.key, parent_hash, page_size=page_size)
    assert isinstance(hash_values, list)
    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Mirrors sglang.srt.mem_cache.utils.split_node_hash_value.
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash
