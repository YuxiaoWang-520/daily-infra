"""CPU teaching models, not fused attention kernels or distributed ring backward."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class KVBlock:
    """K/V rows and their global sequence positions, in matching row order."""

    k: Tensor
    v: Tensor
    positions: Tensor


def _matrix(tensor: Tensor, name: str, *, empty_rows: bool = False) -> None:
    if (
        tensor.ndim != 2
        or tensor.shape[1] < 1
        or (not empty_rows and tensor.shape[0] < 1)
    ):
        raise ValueError(f"{name} must have shape [tokens, positive_features]")
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64 in this teaching model")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _positions(positions: Tensor, rows: int, device: torch.device) -> None:
    if positions.ndim != 1 or positions.numel() != rows:
        raise ValueError("positions must have one element per token row")
    if positions.dtype not in (torch.int32, torch.int64) or positions.device != device:
        raise TypeError("positions must be integer tensors on the same device")


def streaming_attention(
    q: Tensor,
    blocks: Iterable[KVBlock],
    query_positions: Tensor,
    *,
    causal: bool = True,
) -> tuple[Tensor, Tensor]:
    """Merge block scores using one global row maximum, denominator and numerator.

    Return (output [Nq, Dv], logsumexp [Nq]). A fully masked query returns zero
    output and -inf logsumexp. Blocks must together cover the intended keys once;
    the caller owns this partition invariant. Autograd retains intermediates,
    so this implementation makes no training-memory-efficiency claim.
    """
    _matrix(q, "q")
    _positions(query_positions, q.shape[0], q.device)
    maximum = q.new_full((q.shape[0],), -math.inf)
    denominator = q.new_zeros(q.shape[0])
    numerator = None
    for block in blocks:
        k, v = block.k, block.v
        _matrix(k, "k", empty_rows=True)
        _matrix(v, "v", empty_rows=True)
        if k.shape[1] != q.shape[1] or k.shape[0] != v.shape[0]:
            raise ValueError("K features must match Q, and K/V rows must match")
        if any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
            raise TypeError("Q/K/V must share dtype and device")
        _positions(block.positions, k.shape[0], q.device)
        if numerator is None:
            numerator = q.new_zeros((q.shape[0], v.shape[1]))
        elif numerator.shape[1] != v.shape[1]:
            raise ValueError("all V blocks must have the same feature width")
        if k.shape[0] == 0:
            numerator = numerator + (q * 0).sum() + k.sum() + v.sum()
            continue
        scores = q @ k.T / math.sqrt(q.shape[1])
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(
                "score computation overflowed; use smaller values or a wider dtype"
            )
        if causal:
            allowed = query_positions[:, None] >= block.positions[None, :]
            scores = scores.masked_fill(~allowed, -math.inf)
        new_maximum = torch.maximum(maximum, scores.amax(dim=1))
        # A completely masked prefix has m=-inf. Subtract zero in that case:
        # exp(-inf - 0)=0 avoids an undefined -inf - (-inf) in both passes.
        reference = torch.where(torch.isfinite(new_maximum), new_maximum, 0.0)
        old_scale = torch.exp(maximum - reference)
        weights = torch.exp(scores - reference[:, None])
        denominator = old_scale * denominator + weights.sum(dim=1)
        numerator = old_scale[:, None] * numerator + weights @ v
        maximum = new_maximum
    if numerator is None:
        raise ValueError(
            "provide at least one K/V block, even if its row count is zero"
        )
    safe_denominator = torch.where(denominator > 0, denominator, 1.0)
    output = numerator / safe_denominator[:, None]
    lse = torch.where(denominator > 0, maximum + safe_denominator.log(), -math.inf)
    return output, lse


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def sequence_shards(
    seq_len: int, cp_size: int, *, layout: str = "contiguous"
) -> tuple[tuple[int, ...], ...]:
    """Equal shards; zigzag pairs chunk r with chunk (2p-1-r)."""
    _positive_integer(seq_len, "seq_len")
    _positive_integer(cp_size, "cp_size")
    if layout not in ("contiguous", "zigzag"):
        raise ValueError("layout must be contiguous or zigzag")
    if cp_size == 1:
        return (tuple(range(seq_len)),)
    divisor = cp_size if layout == "contiguous" else 2 * cp_size
    if seq_len % divisor:
        raise ValueError(
            "sequence length must be divisible by the layout's chunk count"
        )
    width = seq_len // divisor
    if layout == "contiguous":
        return tuple(tuple(range(r * width, (r + 1) * width)) for r in range(cp_size))
    return tuple(
        tuple(range(r * width, (r + 1) * width))
        + tuple(range((2 * cp_size - 1 - r) * width, (2 * cp_size - r) * width))
        for r in range(cp_size)
    )


def ring_order(rank: int, cp_size: int) -> tuple[int, ...]:
    """Send to rank+1, receive from rank-1, with the local owner visited first."""
    _positive_integer(cp_size, "cp_size")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < cp_size:
        raise ValueError("rank must be an integer in [0, cp_size)")
    return tuple((rank - step) % cp_size for step in range(cp_size))


def causal_pair_counts(shards: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Count allowed (query,key) pairs for a full self-attention sequence."""
    flattened = [position for shard in shards for position in shard]
    if not flattened or sorted(flattened) != list(range(len(flattened))):
        raise ValueError("shards must partition global positions 0..S-1 once")
    return tuple(sum(position + 1 for position in shard) for shard in shards)


def cp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cp_size: int,
    *,
    layout: str = "contiguous",
    causal: bool = True,
) -> Tensor:
    """One-process CP simulator; shared autograd sums all owners' K/V gradients."""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _matrix(tensor, name)
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            "this self-attention model requires equal Q/K/V sequence lengths"
        )
    shards = sequence_shards(q.shape[0], cp_size, layout=layout)
    indices = [
        torch.tensor(shard, device=q.device, dtype=torch.int64) for shard in shards
    ]
    outputs = []
    for rank, query_indices in enumerate(indices):
        kv_blocks = (
            KVBlock(
                k.index_select(0, indices[owner]),
                v.index_select(0, indices[owner]),
                indices[owner],
            )
            for owner in ring_order(rank, cp_size)
        )
        local_output, _ = streaming_attention(
            q.index_select(0, query_indices), kv_blocks, query_indices, causal=causal
        )
        outputs.append(local_output)
    inverse_order = torch.cat(indices).argsort()
    return torch.cat(outputs).index_select(0, inverse_order)
