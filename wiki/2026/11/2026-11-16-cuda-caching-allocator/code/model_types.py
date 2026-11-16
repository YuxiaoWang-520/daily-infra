"""Immutable byte-based values for the single-pool allocator exercises."""

from dataclasses import dataclass
from typing import Literal

State = Literal["active_allocated", "active_awaiting_free", "inactive"]


def positive_bytes(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("Byte counts must be positive integers")


@dataclass(frozen=True)
class Block:
    size: int
    state: State
    handle: int | None = None

    def __post_init__(self) -> None:
        positive_bytes(self.size)
        if self.state not in ("active_allocated", "active_awaiting_free", "inactive"):
            raise ValueError("Unknown block state")


@dataclass(frozen=True)
class Segment:
    identifier: int
    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("A segment must contain at least one block")


@dataclass(frozen=True)
class Totals:
    reserved_bytes: int
    allocated_bytes: int
    awaiting_bytes: int
    inactive_bytes: int
    releasable_bytes: int
    trapped_bytes: int
    largest_free_bytes: int
    segment_count: int
