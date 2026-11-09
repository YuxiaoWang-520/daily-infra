"""Integer planning results for a uniformly padded flat-vector teaching model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StateBudget:
    shard_elements: int
    padded_elements: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class RingBudget:
    all_gather_calls: int
    all_gather_sent_bytes: int
    reduce_scatter_sent_bytes: int
    total_sent_bytes: int
    ddp_sent_bytes: int
