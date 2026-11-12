"""Single-host, single-device-queue teaching model, measured in microseconds."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayEstimate:
    eager_us: float
    graph_us: float
    device_us: float
    eager_steady_us: float
    graph_steady_us: float
    first_faster_replays: int | None


def estimate_replays(
    kernels_us: Sequence[float],
    launch_us: float,
    graph_launch_us: float,
    setup_us: float,
    replays: int,
) -> ReplayEstimate:
    """Estimate completion from time zero, including graph setup."""
    raise NotImplementedError("Implement the queue recurrence and break-even rule")
