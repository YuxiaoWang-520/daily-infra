"""Idealized synchronous-round communication capacity models.

Routes map logical directed ring edges to physical shared resources. Per-round
loads sum on each resource; the capacity term is max(load / capacity).
This fluid, fully pipelined per-round abstraction ignores reduction compute,
per-hop latency, scheduling, packet headers and finite pipeline effects.
It estimates this model only, not NCCL algorithm choices or measured latency.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Cost:
    rounds: int
    per_round_load_bytes: dict[str, int]
    bottleneck_seconds_per_round: float
    estimated_seconds: float


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _finite_number(value: float, name: str, *, allow_zero: bool = False) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f"{name} must be a finite float-representable number"
        ) from None
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return number


def _finite_time(value: Fraction) -> float:
    """Convert only after division, and reject an unrepresentable final time."""
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("modeled time must have a finite float result") from None
    if not math.isfinite(result):
        raise ValueError("modeled time must have a finite float result")
    return result


def ring_cost(
    payload_bytes: int,
    order: Sequence[int],
    routes: Mapping[tuple[int, int], tuple[str, ...]],
    bandwidth_bytes_per_second: Mapping[str, float],
    alpha_seconds: float,
) -> Cost:
    """Estimate two ring phases, with all resources directional and shared."""
    _positive_integer(payload_bytes, "payload_bytes")
    alpha = _finite_number(alpha_seconds, "alpha_seconds", allow_zero=True)
    ranks = len(order)
    if not ranks or any(type(rank) is not int for rank in order):
        raise ValueError("order must be a nonempty integer rank permutation")
    if set(order) != set(range(ranks)):
        raise ValueError("order must be a rank permutation of [0, P)")
    if payload_bytes % ranks:
        raise ValueError("payload_bytes must be divisible by rank count")
    if ranks == 1:
        return Cost(0, {}, 0.0, 0.0)
    loads: dict[str, int] = {}
    capacities: dict[str, float] = {}
    chunk_bytes = payload_bytes // ranks
    for index, source in enumerate(order):
        destination = order[(index + 1) % ranks]
        route = routes.get((source, destination))
        if not isinstance(route, tuple) or not route:
            raise ValueError(f"missing or empty route for {source}->{destination}")
        if any(not isinstance(resource, str) or not resource for resource in route):
            raise ValueError("resource names must be nonempty strings")
        if len(set(route)) != len(route):
            raise ValueError("a route must not repeat a physical resource")
        for resource in route:
            if resource not in bandwidth_bytes_per_second:
                raise ValueError(f"unknown resource: {resource}")
            capacities[resource] = _finite_number(
                bandwidth_bytes_per_second[resource], "resource capacity"
            )
            loads[resource] = loads.get(resource, 0) + chunk_bytes
    capacity_fraction = max(
        Fraction(load) / Fraction(capacities[resource])
        for resource, load in loads.items()
    )
    rounds = 2 * (ranks - 1)
    total = _finite_time(rounds * (Fraction(alpha) + capacity_fraction))
    return Cost(rounds, loads, _finite_time(capacity_fraction), total)


def binomial_tree_cost(
    payload_bytes: int,
    ranks: int,
    bandwidth_bytes_per_second: float,
    alpha_seconds: float,
) -> float:
    """Full-vector binomial reduce then broadcast, one peer per level.

    Power-of-two ranks, independent links, no contention and no pipelining.
    This baseline is deliberately simpler than NCCL's tree implementations.
    """
    _positive_integer(payload_bytes, "payload_bytes")
    _positive_integer(ranks, "ranks")
    if ranks & (ranks - 1):
        raise ValueError("ranks must be a power of two")
    bandwidth = _finite_number(bandwidth_bytes_per_second, "bandwidth")
    alpha = _finite_number(alpha_seconds, "alpha_seconds", allow_zero=True)
    if ranks == 1:
        return 0.0
    levels = ranks.bit_length() - 1
    return _finite_time(
        2 * levels * (Fraction(alpha) + Fraction(payload_bytes) / Fraction(bandwidth))
    )
