"""Learner starting points; contracts and reference code are in README.md."""

from typing import Any


def simulate_checkpoints(
    num_steps: int, compute_ms: int, interval_steps: int, stage_ms: int, write_ms: int
) -> dict[str, Any]:
    """Exercise 1: one pending save, fixed staging/write durations, final drain."""
    raise NotImplementedError(
        "Implement the event schedule and backpressure accounting"
    )


def run_checkpoint_case(
    *, async_staging: bool = True, fail_write: bool = False
) -> dict[str, Any]:
    """Exercise 2: save a stable CPU snapshot, observe futures, reload and compare."""
    raise NotImplementedError(
        "Use CPU DCP and explicit event gates, with no sleep assumptions"
    )
