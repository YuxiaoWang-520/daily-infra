"""Answer for exercise one, using the documented teaching snapshot types."""

from collections.abc import Sequence

from model_types import Segment, Totals


def summarize(segments: Sequence[Segment]) -> Totals:
    """Account for allocated, awaiting and inactive bytes in ordinary segments."""
    reserved = allocated = awaiting = inactive = releasable = trapped = largest = 0
    for segment in segments:
        free_sizes = [b.size for b in segment.blocks if b.state == "inactive"]
        segment_free = sum(free_sizes)
        reserved += sum(b.size for b in segment.blocks)
        allocated += sum(
            b.size for b in segment.blocks if b.state == "active_allocated"
        )
        awaiting += sum(
            b.size for b in segment.blocks if b.state == "active_awaiting_free"
        )
        inactive += segment_free
        largest = max(largest, max(free_sizes, default=0))
        if all(b.state == "inactive" for b in segment.blocks):
            releasable += segment_free
        else:
            trapped += segment_free
    return Totals(
        reserved_bytes=reserved,
        allocated_bytes=allocated,
        awaiting_bytes=awaiting,
        inactive_bytes=inactive,
        releasable_bytes=releasable,
        trapped_bytes=trapped,
        largest_free_bytes=largest,
        segment_count=len(segments),
    )
