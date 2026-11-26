"""Learner API skeleton; dataclasses are supplied so the focus stays on mechanisms."""

from collections.abc import Sequence

import torch
from answers import CostEstimate, PackedBatch


def first_fit_decreasing(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """Exercise 1: stable length-descending first-fit assignment by original ID."""
    raise NotImplementedError


def padded_cost(
    lengths: Sequence[int], batch_size: int, *, order: Sequence[int] | None = None
) -> CostEstimate:
    """Exercise 1: count token slots, dense square area and useful causal pairs."""
    raise NotImplementedError


def packed_cost(
    lengths: Sequence[int], capacity: int, groups: Sequence[Sequence[int]]
) -> CostEstimate:
    """Exercise 1: validate whole-sequence coverage and fixed pack capacity."""
    raise NotImplementedError


def pack_batch(sequences: Sequence[Sequence[int]]) -> PackedBatch:
    """Exercise 2: reset absolute positions, isolate causal blocks, shift labels locally."""
    raise NotImplementedError


def loss_sum_and_count(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Exercise 2: sum CE at valid predictor positions, without a premature mean."""
    raise NotImplementedError


def ddp_sum_scale(valid_counts: Sequence[int]) -> float:
    """Exercise 2: compensate standard rank averaging using global valid target count."""
    raise NotImplementedError
