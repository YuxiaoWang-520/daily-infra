"""Single-layer, single-head CPU paging model; not a vLLM allocator or kernel."""

from collections import deque
from collections.abc import Sequence
from itertools import islice

import torch


def _integer(value: int, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}")


def slot_for_token(
    block_table: Sequence[int],
    position: int,
    *,
    block_size: int,
    length: int,
    num_blocks: int,
) -> int:
    """Map an existing token to a flat token slot, not a byte address.

    Only the selected table entry is checked. Position must be inside the
    valid history, even if a later position fits in an allocated tail page.
    IDs start at zero in this toy pool; negative IDs are rejected, not indexed.
    """
    _integer(block_size, "block_size", minimum=1)
    _integer(num_blocks, "num_blocks", minimum=1)
    _integer(length, "length", minimum=0)
    _integer(position, "position", minimum=0)
    if position >= length:
        raise IndexError("position is outside the valid cached prefix")
    logical_block, offset = divmod(position, block_size)
    if logical_block >= len(block_table):
        raise ValueError("block table does not cover this position")
    physical_block = block_table[logical_block]
    _integer(physical_block, "physical_block", minimum=0)
    if physical_block >= num_blocks:
        raise ValueError("physical block is outside the pool")
    return physical_block * block_size + offset


def page_usage(
    lengths: Sequence[int], *, block_size: int, bytes_per_token: int
) -> dict[str, int]:
    """Count no-sharing, on-demand pages; exclude metadata and pool reservation.

    bytes_per_token already includes both K and V and all layers/heads counted
    by the caller. There is no speculative lookahead or heterogeneous layout.
    """
    _integer(block_size, "block_size", minimum=1)
    _integer(bytes_per_token, "bytes_per_token", minimum=1)
    tokens = 0
    blocks = 0
    for length in lengths:
        _integer(length, "length", minimum=0)
        tokens += length
        blocks += (length + block_size - 1) // block_size
    return {
        "tokens": tokens,
        "blocks": blocks,
        "unused_slots": blocks * block_size - tokens,
        "content_bytes": tokens * bytes_per_token,
        "allocated_bytes": blocks * block_size * bytes_per_token,
    }


def _request_id(value: str) -> None:
    if type(value) is not str or not value:
        raise ValueError("request_id must be a nonempty string")


class PagedKVCache:
    """Fixed CPU float64 K/V pool with exclusive pages and FIFO allocation.

    Inputs are [tokens, head_dim]. Append copies data without autograd history;
    read returns independent copies in logical token order. A request starts
    with its first nonempty append. Invalid inputs and insufficient capacity do
    not change its valid prefix or ownership. This is single-threaded and has
    no sharing, prefix cache, eviction, GPU synchronization, or security boundary.
    """

    def __init__(self, *, num_blocks: int, block_size: int, head_dim: int) -> None:
        for name, value in (
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("head_dim", head_dim),
        ):
            _integer(value, name, minimum=1)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self._key = torch.full(
            (num_blocks * block_size, head_dim),
            torch.nan,
            dtype=torch.float64,
            device="cpu",
        )
        self._value = torch.full_like(self._key, torch.nan)
        self._free: deque[int] = deque(range(num_blocks))
        self._tables: dict[str, list[int]] = {}
        self._lengths: dict[str, int] = {}

    @property
    def free_block_count(self) -> int:
        return len(self._free)

    def block_table(self, request_id: str) -> tuple[int, ...]:
        _request_id(request_id)
        return tuple(self._tables[request_id])

    def length(self, request_id: str) -> int:
        _request_id(request_id)
        return self._lengths[request_id]

    def append(
        self, request_id: str, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[int, ...]:
        """Append one or more already-computed K/V rows; return their slots."""
        _request_id(request_id)
        for tensor in (key, value):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float64
            ):
                raise ValueError("K/V must be CPU float64 tensors")
            if (
                tensor.ndim != 2
                or tensor.shape[0] == 0
                or tensor.shape[1] != self.head_dim
            ):
                raise ValueError("K/V must have nonempty shape [tokens, head_dim]")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("K/V must be finite")
        if key.shape != value.shape:
            raise ValueError("K and V must have equal shapes")

        old_length = self._lengths.get(request_id, 0)
        new_length = old_length + key.shape[0]
        old_table = self._tables.get(request_id, [])
        required = (new_length + self.block_size - 1) // self.block_size
        needed = required - len(old_table)
        if needed > len(self._free):
            raise MemoryError("insufficient free KV blocks")
        new_blocks = list(islice(self._free, needed))
        table = old_table + new_blocks
        slots = tuple(
            slot_for_token(
                table,
                position,
                block_size=self.block_size,
                length=new_length,
                num_blocks=self.num_blocks,
            )
            for position in range(old_length, new_length)
        )
        indices = torch.tensor(slots, dtype=torch.int64, device="cpu")
        with torch.no_grad():
            self._key.index_copy_(0, indices, key)
            self._value.index_copy_(0, indices, value)
        for _ in new_blocks:
            self._free.popleft()
        self._tables[request_id] = table
        self._lengths[request_id] = new_length
        return slots

    def read(self, request_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather valid rows only. Real paged kernels need not gather first."""
        length = self.length(request_id)
        table = self.block_table(request_id)
        slots = [
            slot_for_token(
                table,
                position,
                block_size=self.block_size,
                length=length,
                num_blocks=self.num_blocks,
            )
            for position in range(length)
        ]
        indices = torch.tensor(slots, dtype=torch.int64, device="cpu")
        return self._key.index_select(0, indices), self._value.index_select(0, indices)

    def release(self, request_id: str) -> None:
        """Return exclusively owned blocks; unknown or repeated release fails.

        Storage is not erased. Subsequent reads still use the new owner's valid
        length; this example does not implement multi-tenant data sanitization.
        """
        _request_id(request_id)
        blocks = self._tables.pop(request_id)
        del self._lengths[request_id]
        self._free.extend(blocks)
