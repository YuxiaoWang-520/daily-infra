"""Exercise 1. Implement exact integer storage and ideal ring send volumes."""

from models import RingBudget, StateBudget


def state_budget(
    parameters: int,
    world_size: int,
    stage: int,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_bytes: int = 12,
) -> StateBudget:
    """Use uniform ceil-padding only for states that are sharded."""
    raise NotImplementedError("Implement the four model-state budgets")


def ring_budget(
    parameters: int,
    world_size: int,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    all_gather_calls: int = 2,
) -> RingBudget:
    """Count ideal per-rank sent bytes using the same padded length."""
    raise NotImplementedError("Derive one or two gathers plus reduce-scatter")
