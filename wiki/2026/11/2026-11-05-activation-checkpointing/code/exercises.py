"""Exercise 1: implement the explicitly simplified uniform-chain model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentPlan:
    segment_size: int
    segments: int
    activation_slots: int
    activation_bytes: int
    baseline_us: float
    checkpoint_us: float

    @property
    def slowdown(self) -> float:
        return self.checkpoint_us / self.baseline_us


def estimate_uniform(
    layers: int,
    segment_size: int,
    activation_bytes: int,
    forward_us: float,
    backward_us: float,
    overhead_us: float = 0.0,
) -> SegmentPlan:
    """Estimate ceil(n/s)+s slots and 2F+B+ceil(n/s)*overhead time."""
    raise NotImplementedError("Exercise 1: derive the model before implementing it")


def choose_plan(
    layers: int,
    activation_bytes: int,
    memory_budget_bytes: int,
    forward_us: float,
    backward_us: float,
    overhead_us: float = 0.0,
) -> SegmentPlan | None:
    """Minimize estimated time, then bytes, then segment size; None if infeasible."""
    raise NotImplementedError("Exercise 1: enumerate the feasible segment sizes")
