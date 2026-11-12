"""Single-host, single-device-queue teaching model, measured in microseconds."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


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
    if isinstance(replays, bool) or not isinstance(replays, int) or replays < 1:
        raise ValueError("replays must be a positive integer")
    try:
        durations = tuple(_cost(value) for value in kernels_us)
    except TypeError as error:
        raise ValueError("kernels_us must be a sequence of costs") from error
    if not durations:
        raise ValueError("at least one kernel is required")
    host = _cost(launch_us)
    graph_host = _cost(graph_launch_us)
    setup = _cost(setup_us)

    first_eager = Fraction(0)
    for index, duration in enumerate(durations, start=1):
        first_eager = max(first_eager, index * host) + duration
    device = sum(durations, Fraction(0))
    eager_step = max(len(durations) * host, device)
    graph_step = max(graph_host, device)
    first_graph = setup + graph_host + device
    gain = eager_step - graph_step
    if first_graph < first_eager:
        first_faster = 1
    elif gain <= 0:
        first_faster = None
    else:
        first_faster = (first_graph - first_eager) // gain + 2

    return ReplayEstimate(
        eager_us=_finite_float(first_eager + (replays - 1) * eager_step),
        graph_us=_finite_float(first_graph + (replays - 1) * graph_step),
        device_us=_finite_float(device),
        eager_steady_us=_finite_float(eager_step),
        graph_steady_us=_finite_float(graph_step),
        first_faster_replays=first_faster,
    )


def _cost(value: float) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("costs must be finite, nonnegative numbers")  # noqa: TRY004 - exercise contract
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError("cost is outside the finite float range") from error
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError("costs must be finite, nonnegative numbers")
    return Fraction(normalized)


def _finite_float(value: Fraction) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(
            "estimated duration is outside the finite float range"
        ) from error
    if not math.isfinite(result):
        raise ValueError("estimated duration is outside the finite float range")
    return result
