"""Reference arithmetic and a dependency model, not a distributed runtime."""

import math
from collections.abc import Sequence

from models import BucketEvent, Schedule


def _sequence(value: object, name: str) -> Sequence:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty list or tuple")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is too large") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite range")
    return result


def _counts(counts: Sequence[int]) -> tuple[int, ...]:
    return tuple(_positive_integer(c, "count") for c in _sequence(counts, "counts"))


def global_mean_from_local_means(
    means: Sequence[Sequence[float]], counts: Sequence[int]
) -> tuple[float, ...]:
    """Return the effective-count weighted mean of rank-local mean gradients."""
    weights = _counts(counts)
    rows = tuple(
        tuple(_finite(x, "gradient") for x in _sequence(row, "gradient row"))
        for row in _sequence(means, "means")
    )
    if len(rows) != len(weights) or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("means must be rectangular with one row per count")
    total = sum(weights)
    return tuple(
        math.fsum(row[j] * (count / total) for row, count in zip(rows, weights))
        for j in range(len(rows[0]))
    )


def ddp_loss_scales(counts: Sequence[int]) -> tuple[float, ...]:
    """Multiply each local mean loss by R*n_r/N before default DDP averaging."""
    weights = _counts(counts)
    total = sum(weights)
    return tuple(len(weights) * (count / total) for count in weights)


def schedule_buckets(
    ready_ms: Sequence[Sequence[float]],
    sizes_bytes: Sequence[int],
    buckets: Sequence[Sequence[int]],
    startup_ms: float,
    bytes_per_ms: float,
) -> Schedule:
    """Evaluate fixed bucket order on a single serial communication lane.

    Readiness is fixed independently of communication. An operation's effective
    start waits for all its gradients on all ranks and the preceding operation.
    This is not a trace of per-rank enqueue, network progress or NCCL kernels.
    """
    sizes = tuple(
        _positive_integer(size, "size")
        for size in _sequence(sizes_bytes, "sizes_bytes")
    )
    ready = tuple(
        tuple(_finite(t, "ready time", 0) for t in _sequence(row, "ready row"))
        for row in _sequence(ready_ms, "ready_ms")
    )
    if any(len(row) != len(sizes) for row in ready):
        raise ValueError("every rank needs one ready time per gradient")
    alpha = _finite(startup_ms, "startup_ms", 0)
    rate = _finite(bytes_per_ms, "bytes_per_ms", 0)
    if rate == 0:
        raise ValueError("bytes_per_ms must be positive")
    groups = tuple(
        tuple(_sequence(group, "bucket")) for group in _sequence(buckets, "buckets")
    )
    indices = [i for group in groups for i in group]
    if any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
        raise ValueError("gradient indices must be integers")
    if sorted(indices) != list(range(len(sizes))):
        raise ValueError("buckets must partition all gradients exactly once")

    events = []
    previous_end = 0.0
    durations = []
    for index, group in enumerate(groups):
        size = sum(sizes[i] for i in group)
        ready_at = max(row[i] for row in ready for i in group)
        try:
            raw_duration = alpha + size / rate
        except OverflowError as exc:
            raise ValueError("communication duration is too large") from exc
        duration = _finite(raw_duration, "communication duration", 0)
        start = max(ready_at, previous_end)
        finish = _finite(start + duration, "finish time", 0)
        events.append(BucketEvent(index, group, size, ready_at, start, finish))
        durations.append(duration)
        previous_end = finish
    backward = max(map(max, ready))
    finish = max(backward, previous_end)
    try:
        raw_serial = backward + math.fsum(durations)
    except OverflowError as exc:
        raise ValueError("serial finish is too large") from exc
    serial = _finite(raw_serial, "serial finish", 0)
    return Schedule(tuple(events), backward, finish, finish - backward, serial)
