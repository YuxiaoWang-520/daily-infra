"""Immutable payload accounting, separate from any device allocator."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StateBytes:
    parameters: int
    gradients: int
    first_moment: int
    second_moment: int
    master: int
    amsgrad_max: int

    @property
    def total(self) -> int:
        return sum(
            (
                self.parameters,
                self.gradients,
                self.first_moment,
                self.second_moment,
                self.master,
                self.amsgrad_max,
            )
        )


@dataclass(frozen=True)
class Event:
    op: Literal["alloc", "alias", "drop"]
    name: str
    nbytes: int = 0
    source: str | None = None


@dataclass(frozen=True)
class MemoryPoint:
    event_index: int
    live_bytes: int
    live_storage_count: int
    live_names: tuple[str, ...]


@dataclass(frozen=True)
class PeakReport:
    peak_bytes: int
    peak_event_index: int | None
    final_bytes: int
    points: tuple[MemoryPoint, ...]
