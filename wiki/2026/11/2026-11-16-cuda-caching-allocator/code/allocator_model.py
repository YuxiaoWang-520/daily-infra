"""Answer for exercise two, a CPU teaching model of one ordinary memory pool."""

from model_types import Block, Segment, positive_bytes


def _coalesce(blocks: list[Block]) -> list[Block]:
    merged: list[Block] = []
    for block in blocks:
        if merged and merged[-1].state == block.state == "inactive":
            merged[-1] = Block(merged[-1].size + block.size, "inactive")
        else:
            merged.append(block)
    return merged


class Allocator:
    """Best-fit reuse with splitting, coalescing and a finite byte capacity."""

    def __init__(self, capacity: int, segment_size: int) -> None:
        positive_bytes(capacity)
        positive_bytes(segment_size)
        self.capacity = capacity
        self.segment_size = segment_size
        self.segment_allocations = 0
        self.allocation_retries = 0
        self.ooms = 0
        self._segments: dict[int, list[Block]] = {}
        self._next_segment = 0
        self._next_handle = 1

    def malloc(self, size: int) -> int:
        """Return a unique handle, reclaiming whole free segments on pressure."""
        positive_bytes(size)
        candidates = [
            (block.size, identifier, index)
            for identifier, blocks in self._segments.items()
            for index, block in enumerate(blocks)
            if block.state == "inactive" and block.size >= size
        ]
        candidate = min(candidates, default=None)
        if candidate is None:
            segment_bytes = max(self.segment_size, size)
            reserved = sum(b.size for blocks in self._segments.values() for b in blocks)
            if reserved + segment_bytes > self.capacity:
                self.allocation_retries += 1
                reserved -= self.release_free_segments()
            if reserved + segment_bytes > self.capacity:
                self.ooms += 1
                raise MemoryError("No sufficient block or capacity for a new segment")
            identifier = self._next_segment
            self._next_segment += 1
            self._segments[identifier] = [Block(segment_bytes, "inactive")]
            self.segment_allocations += 1
            candidate = (segment_bytes, identifier, 0)

        block_size, identifier, index = candidate
        handle = self._next_handle
        self._next_handle += 1
        replacements = [Block(size, "active_allocated", handle)]
        if block_size > size:
            replacements.append(Block(block_size - size, "inactive"))
        self._segments[identifier][index : index + 1] = replacements
        return handle

    def free(self, handle: int, *, deferred: bool = False) -> None:
        """Release ownership, optionally leaving bytes pending explicit completion."""
        if type(handle) is not int:
            raise KeyError(handle)
        for identifier, blocks in self._segments.items():
            for index, block in enumerate(blocks):
                if block.handle == handle and block.state == "active_allocated":
                    if deferred:
                        blocks[index] = Block(
                            block.size, "active_awaiting_free", handle
                        )
                    else:
                        blocks[index] = Block(block.size, "inactive")
                    self._segments[identifier] = _coalesce(blocks)
                    return
        raise KeyError(handle)

    def complete_deferred(self) -> int:
        """Mark every pending use complete and return the affected byte count."""
        completed = 0
        for identifier, blocks in self._segments.items():
            ready = []
            for block in blocks:
                if block.state == "active_awaiting_free":
                    completed += block.size
                    ready.append(Block(block.size, "inactive"))
                else:
                    ready.append(block)
            self._segments[identifier] = _coalesce(ready)
        return completed

    def release_free_segments(self) -> int:
        """Return wholly inactive segments without completing pending operations."""
        identifiers = [
            identifier
            for identifier, blocks in self._segments.items()
            if all(block.state == "inactive" for block in blocks)
        ]
        released = 0
        for identifier in identifiers:
            released += sum(block.size for block in self._segments.pop(identifier))
        return released

    def snapshot(self) -> tuple[Segment, ...]:
        """Return immutable values detached from future allocator updates."""
        return tuple(
            Segment(identifier, tuple(blocks))
            for identifier, blocks in self._segments.items()
        )
