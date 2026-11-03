"""Fill in these interfaces, then set EXERCISE_IMPL=exercises for the tests."""

from collections.abc import Sequence

from models import Schedule


def global_mean_from_local_means(
    means: Sequence[Sequence[float]], counts: Sequence[int]
) -> tuple[float, ...]:
    """Weight each rank's mean gradient by its positive effective count."""
    raise NotImplementedError


def ddp_loss_scales(counts: Sequence[int]) -> tuple[float, ...]:
    """Scale local mean losses before an ordinary rank-averaging DDP backward."""
    raise NotImplementedError


def schedule_buckets(
    ready_ms: Sequence[Sequence[float]],
    sizes_bytes: Sequence[int],
    buckets: Sequence[Sequence[int]],
    startup_ms: float,
    bytes_per_ms: float,
) -> Schedule:
    """Evaluate fixed-order, serial communication with fixed gradient readiness."""
    raise NotImplementedError
