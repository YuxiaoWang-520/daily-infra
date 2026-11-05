"""Uniform-chain planning proxy; bytes and microseconds are model inputs."""

import math

from exercises import SegmentPlan


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _duration(value: float, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite duration")  # noqa: TRY004 -- exercise API
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError(
            f"{name} is outside the finite floating-point range"
        ) from error
    if not finite or value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


def estimate_uniform(
    layers: int,
    segment_size: int,
    activation_bytes: int,
    forward_us: float,
    backward_us: float,
    overhead_us: float = 0.0,
) -> SegmentPlan:
    """Use ceil(n/s)+s slots, and assume every segment replays in full once.

    F and B describe the whole network, overhead_us one segment. Endpoints,
    gradients, parameters, workspaces, RNG state and allocator costs are omitted.
    This is a conservative slot proxy within that abstraction, not a live-memory
    simulator or an upper bound on a real model's total device memory.
    """
    for name, value in (
        ("layers", layers),
        ("segment_size", segment_size),
        ("activation_bytes", activation_bytes),
    ):
        _integer(value, name)
    if segment_size > layers:
        raise ValueError("segment_size must not exceed layers")
    _duration(forward_us, "forward_us", positive=True)
    _duration(backward_us, "backward_us")
    _duration(overhead_us, "overhead_us")
    forward_us, backward_us, overhead_us = (
        float(forward_us),
        float(backward_us),
        float(overhead_us),
    )
    segments = (layers + segment_size - 1) // segment_size
    slots = segments + segment_size
    baseline = forward_us + backward_us
    # Scale as an exact ratio before converting, so huge integer counts do not
    # overflow before multiplication by a tiny (or zero) duration can reduce it.
    numerator, denominator = overhead_us.as_integer_ratio()
    try:
        overhead_time = (segments * numerator) / denominator
    except OverflowError as error:
        raise ValueError(
            "segment overhead exceeds the finite floating-point range"
        ) from error
    checkpoint_time = 2 * forward_us + backward_us + overhead_time
    if not math.isfinite(baseline) or not math.isfinite(checkpoint_time):
        raise ValueError("combined durations exceed the finite floating-point range")
    return SegmentPlan(
        segment_size,
        segments,
        slots,
        slots * activation_bytes,
        float(baseline),
        float(checkpoint_time),
    )


def choose_plan(
    layers: int,
    activation_bytes: int,
    memory_budget_bytes: int,
    forward_us: float,
    backward_us: float,
    overhead_us: float = 0.0,
) -> SegmentPlan | None:
    """Enumerate O(n) choices using O(1) auxiliary space for small teaching n.

    The budget covers only the modeled activation slots. First minimize time;
    ties prefer fewer bytes, then a smaller segment size. No fit returns None.
    """
    _integer(memory_budget_bytes, "memory_budget_bytes", minimum=0)
    estimate_uniform(layers, 1, activation_bytes, forward_us, backward_us, overhead_us)
    best = None
    for size in range(1, layers + 1):
        plan = estimate_uniform(
            layers, size, activation_bytes, forward_us, backward_us, overhead_us
        )
        if plan.activation_bytes > memory_budget_bytes:
            continue
        key = (plan.checkpoint_us, plan.activation_bytes, plan.segment_size)
        if best is None or key < (
            best.checkpoint_us,
            best.activation_bytes,
            best.segment_size,
        ):
            best = plan
    return best
