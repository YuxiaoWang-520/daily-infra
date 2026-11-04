"""Complete the interfaces before running EXERCISE_IMPL=exercises tests."""

from collections.abc import Sequence

from models import Event, PeakReport, StateBytes


def estimate_state_bytes(
    numel: int,
    parameter_bytes: int,
    gradient_bytes: int,
    moment_bytes: int,
    master_bytes: int = 0,
    *,
    gradients_live: bool = True,
    state_initialized: bool = True,
    amsgrad: bool = False,
) -> StateBytes:
    """Count a homogeneous, nonaliased Adam parameter group's payload."""
    raise NotImplementedError


def replay_lifetimes(events: Sequence[Event]) -> PeakReport:
    """Replay storage allocations, alias references and last-reference drops."""
    raise NotImplementedError
