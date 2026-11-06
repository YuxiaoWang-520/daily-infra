"""Exercise 1: distinguish stage capacity, task slots and completed batches."""

from models import PipelineBudget


def pipeline_budget(
    storage_ms: float,
    cpu_ms: float,
    workers: int,
    transfer_ms: float,
    compute_ms: float,
    prefetch_factor: int,
    batch_bytes: int,
    allow_device_overlap: bool = False,
) -> PipelineBudget:
    """Use the ideal stage-capacity model defined in code/README.md."""
    raise NotImplementedError("Exercise 1: derive the capacity and payload model")


def exposed_pause(
    pause_ms: float, ready_batches: int, consumption_interval_ms: float
) -> float:
    """Return the part of a supply pause not covered by already-ready batches."""
    raise NotImplementedError("Exercise 1: distinguish ready batches from task slots")
