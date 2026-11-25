"""Starter contracts; see README.md for acceptance criteria and answers.py for solutions."""

from dataclasses import dataclass


def rank_indices(
    sample_count: int,
    world_size: int,
    rank: int,
    *,
    seed: int = 0,
    epoch: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[int]:
    """Exercise 1: permute globally, pad/truncate, then stride by rank."""
    raise NotImplementedError


@dataclass(frozen=True)
class ResumeToken:
    plan_digest: str
    committed_batches: int

    def __post_init__(self) -> None:
        """Validate a SHA-256 hex digest and a nonnegative integer cursor."""
        raise NotImplementedError


@dataclass(frozen=True)
class DataPlan:
    sample_count: int
    world_size: int
    rank: int
    batch_size: int
    seed: int = 0
    epoch: int = 0
    shuffle: bool = True
    sampler_drop_last: bool = False
    loader_drop_last: bool = False
    dataset_revision: str = "synthetic-manifest-v1"

    def __post_init__(self) -> None:
        """Validate the plan's configuration."""
        raise NotImplementedError

    @property
    def batches(self) -> list[list[int]]:
        """Reuse exercise one's rank order, then batch within this rank."""
        raise NotImplementedError

    @property
    def digest(self) -> str:
        """Hash canonical JSON containing schema, configuration and concrete batches."""
        raise NotImplementedError

    def checkpoint(self, committed_batches: int) -> ResumeToken:
        """Reject an out-of-range cursor; record progress after completed work."""
        raise NotImplementedError

    def resume(self, token: ResumeToken) -> list[list[int]]:
        """Validate identity and cursor bounds before returning the remaining batches."""
        raise NotImplementedError
