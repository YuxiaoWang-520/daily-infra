"""Teaching costs and CPU reference semantics; no optimized varlen kernel here."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be int, not bool")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _lengths(values: Sequence[int]) -> list[int]:
    result = list(values)
    for value in result:
        _integer("sequence length", value, 1)
    return result


def _order(values: Sequence[int], size: int) -> list[int]:
    result = list(values)
    for value in result:
        _integer("sequence index", value)
    if sorted(result) != list(range(size)):
        raise ValueError("indices must contain every sequence exactly once")
    return result


@dataclass(frozen=True)
class CostEstimate:
    useful_tokens: int
    token_slots: int
    dense_square_area: int
    varlen_square_area: int
    valid_causal_pairs: int

    @property
    def utilization(self) -> float | None:
        return self.useful_tokens / self.token_slots if self.token_slots else None


def _cost(lengths: list[int], slots: int, area: int) -> CostEstimate:
    return CostEstimate(
        sum(lengths),
        slots,
        area,
        sum(n * n for n in lengths),
        sum(n * (n + 1) // 2 for n in lengths),
    )


def first_fit_decreasing(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """Stable descending-length first fit; a heuristic, not an optimal bin solver."""
    _integer("capacity", capacity, 1)
    values = _lengths(lengths)
    if any(n > capacity for n in values):
        raise ValueError(
            "oversize sequence: choose a separate truncation/splitting policy"
        )
    groups: list[list[int]] = []
    remaining: list[int] = []
    for index in sorted(range(len(values)), key=lambda i: (-values[i], i)):
        for slot, free in enumerate(remaining):
            if free >= values[index]:
                groups[slot].append(index)
                remaining[slot] -= values[index]
                break
        else:
            groups.append([index])
            remaining.append(capacity - values[index])
    return groups


def padded_cost(
    lengths: Sequence[int],
    batch_size: int,
    *,
    order: Sequence[int] | None = None,
) -> CostEstimate:
    """Batch-local maximum length, including the final incomplete microbatch."""
    values = _lengths(lengths)
    _integer("batch_size", batch_size, 1)
    indices = list(range(len(values))) if order is None else _order(order, len(values))
    slots = area = 0
    for offset in range(0, len(indices), batch_size):
        group = indices[offset : offset + batch_size]
        maximum = max(values[i] for i in group)
        slots += len(group) * maximum
        area += len(group) * maximum * maximum
    return _cost(values, slots, area)


def packed_cost(
    lengths: Sequence[int],
    capacity: int,
    groups: Sequence[Sequence[int]],
) -> CostEstimate:
    """Fixed-capacity dense packs; a block mask alone does not remove dense area."""
    values = _lengths(lengths)
    _integer("capacity", capacity, 1)
    copies = [list(g) for g in groups]
    if any(not g for g in copies):
        raise ValueError("empty pack is not allowed")
    _order([i for g in copies for i in g], len(values))
    if any(sum(values[i] for i in g) > capacity for g in copies):
        raise ValueError("pack exceeds capacity")
    return _cost(values, len(copies) * capacity, len(copies) * capacity * capacity)


@dataclass(frozen=True)
class PackedBatch:
    tokens: torch.Tensor
    positions: torch.Tensor
    segments: torch.Tensor
    labels: torch.Tensor
    cu_seqlens: torch.Tensor
    allow: torch.Tensor


def pack_batch(sequences: Sequence[Sequence[int]]) -> PackedBatch:
    """Build unpadded metadata; labels are ALREADY shifted at predictor positions."""
    if not sequences:
        raise ValueError("at least one sequence is required")
    tokens: list[int] = []
    positions: list[int] = []
    segments: list[int] = []
    labels: list[int] = []
    cumulative = [0]
    for segment, sequence in enumerate(sequences):
        values = list(sequence)
        if not values:
            raise ValueError("empty sequence is not supported")
        for value in values:
            _integer("token ID", value)
        tokens.extend(values)
        positions.extend(range(len(values)))
        segments.extend([segment] * len(values))
        labels.extend([*values[1:], -100])
        cumulative.append(len(tokens))
    if len(tokens) >= 2**31:
        raise ValueError("int32 cumulative lengths would overflow")
    segment_tensor = torch.tensor(segments, dtype=torch.long, device="cpu")
    indices = torch.arange(len(tokens), device="cpu")
    same_sequence = segment_tensor[:, None] == segment_tensor[None, :]
    causal = indices[:, None] >= indices[None, :]
    return PackedBatch(
        torch.tensor(tokens, dtype=torch.long, device="cpu"),
        torch.tensor(positions, dtype=torch.long, device="cpu"),
        segment_tensor,
        torch.tensor(labels, dtype=torch.long, device="cpu"),
        torch.tensor(cumulative, dtype=torch.int32, device="cpu"),
        same_sequence & causal,
    )


def loss_sum_and_count(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Unweighted CPU class-index CE sum; leave normalization to the update scope."""
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be tensors")
    if not logits.is_floating_point() or labels.dtype != torch.long:
        raise TypeError("floating logits and int64 labels are required")
    if logits.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("this checked reference accepts CPU tensors only")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("expected logits [T,V] and labels [T]")
    if logits.shape[0] == 0 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError(
            "nonempty finite logits with at least two classes are required"
        )
    valid = labels != -100
    if torch.any((labels[valid] < 0) | (labels[valid] >= logits.shape[1])):
        raise ValueError("target class outside the vocabulary")
    count = int(valid.sum().item())
    total = F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
    return total, count


def ddp_sum_scale(valid_counts: Sequence[int]) -> float:
    """Scale each local SUM before standard R-rank gradient averaging: R / global N."""
    counts = list(valid_counts)
    if not counts:
        raise ValueError("at least one data-parallel rank is required")
    for count in counts:
        _integer("valid target count", count)
    total = sum(counts)
    if total == 0:
        raise ValueError(
            "no supervised targets; coordinate the zero-count update policy"
        )
    return len(counts) / total
