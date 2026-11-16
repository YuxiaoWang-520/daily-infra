"""Exercise two, fill in this API using byte counts and immutable snapshots."""

from model_types import Segment


class Allocator:
    def __init__(self, capacity: int, segment_size: int) -> None:
        raise NotImplementedError

    def malloc(self, size: int) -> int:
        raise NotImplementedError

    def free(self, handle: int, *, deferred: bool = False) -> None:
        raise NotImplementedError

    def complete_deferred(self) -> int:
        raise NotImplementedError

    def release_free_segments(self) -> int:
        raise NotImplementedError

    def snapshot(self) -> tuple[Segment, ...]:
        raise NotImplementedError
