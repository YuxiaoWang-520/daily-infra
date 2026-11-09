"""Exact integer estimates, with explicit padding and ring-volume assumptions."""

from models import RingBudget, StateBudget


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def state_budget(
    parameters: int,
    world_size: int,
    stage: int,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_bytes: int = 12,
) -> StateBudget:
    """Return per-rank vector storage, excluding temporary buffers and activations."""
    for name, value in (
        ("parameters", parameters),
        ("world_size", world_size),
        ("parameter_bytes", parameter_bytes),
        ("gradient_bytes", gradient_bytes),
        ("optimizer_bytes", optimizer_bytes),
    ):
        _positive_integer(value, name)
    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in range(4):
        raise ValueError("stage must be an integer in [0, 3]")
    shard = (parameters + world_size - 1) // world_size
    parameter_count = shard if stage >= 3 else parameters
    gradient_count = shard if stage >= 2 else parameters
    optimizer_count = shard if stage >= 1 else parameters
    parameter_storage = parameter_count * parameter_bytes
    gradient_storage = gradient_count * gradient_bytes
    optimizer_storage = optimizer_count * optimizer_bytes
    return StateBudget(
        shard,
        shard * world_size,
        parameter_storage,
        gradient_storage,
        optimizer_storage,
        parameter_storage + gradient_storage + optimizer_storage,
    )


def ring_budget(
    parameters: int,
    world_size: int,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    all_gather_calls: int = 2,
) -> RingBudget:
    """Count ideal ring sends per rank using the uniformly padded length."""
    for name, value in (
        ("parameters", parameters),
        ("world_size", world_size),
        ("parameter_bytes", parameter_bytes),
        ("gradient_bytes", gradient_bytes),
    ):
        _positive_integer(value, name)
    if (
        isinstance(all_gather_calls, bool)
        or not isinstance(all_gather_calls, int)
        or all_gather_calls not in (1, 2)
    ):
        raise ValueError("all_gather_calls must be 1 or 2")
    shard = (parameters + world_size - 1) // world_size
    sent_elements = shard * (world_size - 1)
    gather = all_gather_calls * sent_elements * parameter_bytes
    scatter = sent_elements * gradient_bytes
    return RingBudget(all_gather_calls, gather, scatter, gather + scatter, 2 * scatter)
