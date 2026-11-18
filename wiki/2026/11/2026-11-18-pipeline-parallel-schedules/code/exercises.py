"""Starter interfaces; the full solutions are in answers.py."""

from collections.abc import Sequence

import torch
from torch import nn


def simulate_pipeline(
    stages: int,
    microbatches: int,
    schedule: str,
    *,
    forward_ticks: Sequence[int] | None = None,
    backward_ticks: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return events, makespan, compute utilization and live-state peaks."""
    raise NotImplementedError


def microbatch_backward(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_sizes: Sequence[int],
) -> dict[str, object]:
    """Accumulate full-batch MSE gradients without an optimizer update."""
    raise NotImplementedError
