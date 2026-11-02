"""CPU metadata references, not a GPU cache or a byte-compatible vLLM clone."""

import hashlib
import json
from collections import deque
from dataclasses import replace

from models import (
    BlockView,
    PoolView,
    PrefixContext,
    require_positive_int,
    require_text,
)


def _digest(value: list) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def block_hashes(
    tokens: list[int] | tuple[int, ...], block_size: int, context: PrefixContext
) -> tuple[str, ...]:
    """Chain only complete blocks; context and block size seed the first link."""
    require_positive_int(block_size, "block_size")
    if not isinstance(context, PrefixContext):
        raise ValueError("context must be PrefixContext")
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError("tokens must be a non-empty list or tuple")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("token IDs must be non-negative integers, not bools")
    parent = _digest(
        ["context-v1", block_size, context.namespace, context.adapter, context.salt]
    )
    keys = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = _digest(["block-v1", parent, list(tokens[start : start + block_size])])
        keys.append(parent)
    return tuple(keys)


def reusable_tokens(
    tokens: list[int] | tuple[int, ...],
    block_size: int,
    context: PrefixContext,
    available: set[str] | frozenset[str],
) -> int:
    """Aligned lookup, reserving at least the last input position for logits."""
    keys = block_hashes(tokens, block_size, context)
    if not isinstance(available, (set, frozenset)) or any(
        not isinstance(key, str) or not key for key in available
    ):
        raise ValueError("available must be a set of non-empty string keys")
    limit = (len(tokens) - 1) // block_size
    hits = 0
    for key in keys[:limit]:
        if key not in available:
            break
        hits += 1
    return hits * block_size


class PrefixPool:
    """Fixed full-block metadata pool with explicit completion and warm eviction.

    Keys are opaque distinct prefix keys in one request, in prefix order.
    Only complete() publishes new blocks. Real engines can register scheduled
    blocks earlier with additional execution dependencies. No KV bytes, partial
    blocks, threads, appends, transfers, or authentication are modeled here.
    """

    def __init__(self, capacity: int) -> None:
        require_positive_int(capacity, "capacity")
        self._blocks = [BlockView(i, None, 0, False) for i in range(capacity)]
        self._free = deque(range(capacity))
        self._cache: dict[str, set[int]] = {}
        self._requests: dict[str, tuple[int, ...]] = {}

    def acquire(self, request_id: str, keys: tuple[str, ...]) -> tuple[int, ...]:
        """Attach ready prefix hits, allocate the tail, or fail without changes."""
        require_text(request_id, "request_id")
        if request_id in self._requests:
            raise ValueError("request already exists")
        if not isinstance(keys, tuple) or not keys:
            raise ValueError("keys must be a non-empty tuple")
        for key in keys:
            require_text(key, "key")
        if len(set(keys)) != len(keys):
            raise ValueError("keys within a request must be distinct")

        hits = []
        for key in keys:
            candidates = self._cache.get(key)
            if not candidates:
                break
            # Sharing an already-pinned copy preserves a warm free copy for
            # allocations; consuming the warm one can falsely exhaust capacity.
            hits.append(
                min(candidates, key=lambda i: (self._blocks[i].ref_count == 0, i))
            )
        warm_hits = sum(self._blocks[i].ref_count == 0 for i in hits)
        if len(keys) - len(hits) > len(self._free) - warm_hits:
            raise RuntimeError("insufficient unpinned blocks")

        # Protect every hit before overwriting any free candidate.
        for block_id in hits:
            block = self._blocks[block_id]
            if block.ref_count == 0:
                self._free.remove(block_id)
            self._blocks[block_id] = replace(block, ref_count=block.ref_count + 1)
        allocated = list(hits)
        for key in keys[len(hits) :]:
            block_id = self._free.popleft()
            old = self._blocks[block_id]
            if old.ready:
                copies = self._cache[old.key]
                copies.remove(block_id)
                if not copies:
                    del self._cache[old.key]
            self._blocks[block_id] = BlockView(block_id, key, 1, False)
            allocated.append(block_id)
        self._requests[request_id] = tuple(allocated)
        return self._requests[request_id]

    def _request_blocks(self, request_id: str) -> tuple[int, ...]:
        require_text(request_id, "request_id")
        if request_id not in self._requests:
            raise ValueError("unknown request")
        return self._requests[request_id]

    def complete(self, request_id: str) -> None:
        """Declare this request's full blocks computed; repeated calls are safe."""
        for block_id in self._request_blocks(request_id):
            block = self._blocks[block_id]
            self._blocks[block_id] = replace(block, ready=True)
            self._cache.setdefault(block.key, set()).add(block_id)

    def release(self, request_id: str) -> None:
        """Release references, preserving ready keys until actual eviction."""
        block_ids = self._request_blocks(request_id)
        uncached, warm = [], []
        for block_id in reversed(block_ids):
            block = self._blocks[block_id]
            block = replace(block, ref_count=block.ref_count - 1)
            self._blocks[block_id] = block
            if block.ref_count == 0:
                (warm if block.ready else uncached).append(block_id)
        self._free.extendleft(reversed(uncached))
        self._free.extend(warm)
        del self._requests[request_id]

    def snapshot(self) -> PoolView:
        """Return immutable, deterministic metadata for inspection."""
        return PoolView(
            tuple(self._blocks),
            tuple(self._free),
            tuple(
                (key, tuple(sorted(ids))) for key, ids in sorted(self._cache.items())
            ),
            tuple(sorted(self._requests.items())),
        )
