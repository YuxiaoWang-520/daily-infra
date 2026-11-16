"""Behavioral acceptance tests, with byte counts reduced for easy inspection."""

import dataclasses
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

from allocator_model import Allocator
from model_types import Block, Segment, Totals
from snapshot_stats import summarize


class LabTests(unittest.TestCase):
    def test_optimized_execution_performs_state_transitions(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", str(Path(__file__).with_name("run_lab.py"))],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        lab = json.loads(result.stdout)
        self.assertEqual(lab["cache_reuse"]["final_reserved_bytes"], 0)
        self.assertEqual(lab["cache_reuse"]["released_bytes"], 8 * 1024 * 1024)
        self.assertEqual(lab["cache_reuse"]["model_final_segment_releases"], 1)
        self.assertEqual(
            lab["deferred"]["after"]["totals"]["allocated_bytes"], 8 * 1024 * 1024
        )
        self.assertEqual(lab["fragmentation"]["expected_rejection"], "MemoryError")


class SnapshotTests(unittest.TestCase):
    def test_empty_snapshot(self) -> None:
        self.assertEqual(summarize(()), Totals(0, 0, 0, 0, 0, 0, 0, 0))

    def test_three_states_and_whole_segment_release(self) -> None:
        segments = (
            Segment(
                0,
                (
                    Block(4, "active_allocated"),
                    Block(2, "active_awaiting_free"),
                    Block(2, "inactive"),
                ),
            ),
            Segment(1, (Block(8, "inactive"),)),
        )
        self.assertEqual(summarize(segments), Totals(16, 4, 2, 10, 8, 2, 8, 2))

    def test_reserved_gap_can_be_all_awaiting(self) -> None:
        totals = summarize((Segment(0, (Block(8, "active_awaiting_free"),)),))
        self.assertEqual(totals.reserved_bytes - totals.allocated_bytes, 8)
        self.assertEqual(totals.inactive_bytes, 0)
        self.assertEqual(totals.trapped_bytes, 0)

    def test_same_gap_can_be_all_releasable(self) -> None:
        totals = summarize((Segment(0, (Block(8, "inactive"),)),))
        self.assertEqual(totals.reserved_bytes - totals.allocated_bytes, 8)
        self.assertEqual(totals.releasable_bytes, 8)
        self.assertEqual(totals.trapped_bytes, 0)

    def test_snapshot_types_are_immutable(self) -> None:
        block = Block(1, "inactive")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            block.size = 2

    def test_invalid_block_sizes(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Block(value, "inactive")

    def test_invalid_state(self) -> None:
        with self.assertRaises(ValueError):
            Block(1, "unknown")

    def test_empty_segment_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            Segment(0, ())


class AllocatorTests(unittest.TestCase):
    def test_invalid_capacity_and_segment_size(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Allocator(value, 8)
            with self.subTest(value=value), self.assertRaises(ValueError):
                Allocator(16, value)

    def test_invalid_requests_preserve_state(self) -> None:
        allocator = Allocator(16, 8)
        allocator.malloc(2)
        before = allocator.snapshot()
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                allocator.malloc(value)
            self.assertEqual(allocator.snapshot(), before)

    def test_split_and_reuse(self) -> None:
        allocator = Allocator(16, 8)
        first, second = allocator.malloc(6), allocator.malloc(2)
        self.assertNotEqual(first, second)
        self.assertEqual(allocator.segment_allocations, 1)
        self.assertEqual([b.size for b in allocator.snapshot()[0].blocks], [6, 2])
        allocator.free(first)
        allocator.malloc(5)
        self.assertEqual(allocator.segment_allocations, 1)
        self.assertEqual(summarize(allocator.snapshot()).inactive_bytes, 1)

    def test_exact_fit_has_no_empty_remainder(self) -> None:
        allocator = Allocator(8, 8)
        allocator.malloc(8)
        self.assertEqual(len(allocator.snapshot()[0].blocks), 1)

    def test_best_fit_chooses_smallest_sufficient_block(self) -> None:
        allocator = Allocator(24, 8)
        handles = [allocator.malloc(size) for size in (6, 2, 4, 4)]
        allocator.free(handles[0])
        allocator.free(handles[2])
        chosen = allocator.malloc(3)
        self.assertEqual(allocator.snapshot()[1].blocks[0].handle, chosen)
        self.assertEqual(allocator.snapshot()[0].blocks[0].size, 6)

    def test_free_coalesces_both_neighbors(self) -> None:
        allocator = Allocator(8, 8)
        left, middle, right = [allocator.malloc(size) for size in (2, 3, 3)]
        allocator.free(left)
        allocator.free(right)
        allocator.free(middle)
        self.assertEqual(allocator.snapshot()[0].blocks, (Block(8, "inactive"),))

    def test_different_segments_do_not_merge(self) -> None:
        allocator = Allocator(16, 8)
        first, second = allocator.malloc(8), allocator.malloc(8)
        allocator.free(first)
        allocator.free(second)
        totals = summarize(allocator.snapshot())
        self.assertEqual((totals.inactive_bytes, totals.largest_free_bytes), (16, 8))
        self.assertEqual(totals.segment_count, 2)

    def test_fragmented_oom_then_neighbor_release(self) -> None:
        allocator = Allocator(16, 8)
        handles = [allocator.malloc(size) for size in (6, 2, 6, 2)]
        allocator.free(handles[0])
        allocator.free(handles[2])
        before = allocator.snapshot()
        with self.assertRaises(MemoryError):
            allocator.malloc(7)
        self.assertEqual(allocator.snapshot(), before)
        self.assertEqual(allocator.release_free_segments(), 0)
        allocator.free(handles[1])
        allocator.malloc(7)
        self.assertEqual(allocator.segment_allocations, 2)
        self.assertEqual(summarize(allocator.snapshot()).allocated_bytes, 9)

    def test_oom_retry_returns_whole_segments_before_growing(self) -> None:
        allocator = Allocator(16, 8)
        first, second = allocator.malloc(8), allocator.malloc(8)
        allocator.free(first)
        allocator.free(second)
        allocator.malloc(12)
        self.assertEqual(allocator.allocation_retries, 1)
        self.assertEqual(allocator.segment_allocations, 3)
        self.assertEqual(summarize(allocator.snapshot()).reserved_bytes, 12)

    def test_recovery_releases_cache_even_if_final_request_fails(self) -> None:
        allocator = Allocator(8, 8)
        allocator.free(allocator.malloc(8))
        with self.assertRaises(MemoryError):
            allocator.malloc(9)
        self.assertEqual(allocator.snapshot(), ())
        self.assertEqual(allocator.ooms, 1)
        allocator.malloc(8)

    def test_deferred_free_stays_active_until_completion(self) -> None:
        allocator = Allocator(8, 8)
        handle = allocator.malloc(8)
        allocator.free(handle, deferred=True)
        totals = summarize(allocator.snapshot())
        self.assertEqual((totals.allocated_bytes, totals.awaiting_bytes), (0, 8))
        self.assertEqual(allocator.release_free_segments(), 0)
        with self.assertRaises(MemoryError):
            allocator.malloc(1)
        self.assertEqual(allocator.complete_deferred(), 8)
        self.assertEqual(allocator.complete_deferred(), 0)
        allocator.malloc(8)
        self.assertEqual(allocator.segment_allocations, 1)

    def test_release_does_not_wait_for_deferred_blocks(self) -> None:
        allocator = Allocator(16, 8)
        pending, cached = allocator.malloc(8), allocator.malloc(8)
        allocator.free(pending, deferred=True)
        allocator.free(cached)
        self.assertEqual(allocator.release_free_segments(), 8)
        self.assertEqual(summarize(allocator.snapshot()).awaiting_bytes, 8)

    def test_unknown_double_and_pending_free_are_rejected(self) -> None:
        allocator = Allocator(8, 8)
        with self.assertRaises(KeyError):
            allocator.free(999)
        handle = allocator.malloc(2)
        allocator.free(handle, deferred=True)
        with self.assertRaises(KeyError):
            allocator.free(handle)
        allocator.complete_deferred()
        with self.assertRaises(KeyError):
            allocator.free(handle)

    def test_snapshot_is_detached_from_later_mutation(self) -> None:
        allocator = Allocator(8, 8)
        handle = allocator.malloc(4)
        before = allocator.snapshot()
        allocator.free(handle)
        self.assertEqual(before[0].blocks[0].state, "active_allocated")
        self.assertEqual(allocator.snapshot()[0].blocks[0].state, "inactive")

    def test_randomized_traces_preserve_byte_accounting(self) -> None:
        generator = random.Random(163)
        allocator = Allocator(64, 8)
        live = {}
        for _ in range(500):
            if live and generator.random() < 0.45:
                handle = generator.choice(list(live))
                allocator.free(handle, deferred=generator.random() < 0.25)
                del live[handle]
            else:
                size = generator.randint(1, 12)
                try:
                    handle = allocator.malloc(size)
                except MemoryError:
                    pass
                else:
                    live[handle] = size
            if generator.random() < 0.1:
                allocator.complete_deferred()
            totals = summarize(allocator.snapshot())
            self.assertEqual(totals.allocated_bytes, sum(live.values()))
            self.assertEqual(
                totals.reserved_bytes,
                totals.allocated_bytes + totals.awaiting_bytes + totals.inactive_bytes,
            )
            self.assertEqual(
                totals.inactive_bytes, totals.releasable_bytes + totals.trapped_bytes
            )
            self.assertLessEqual(totals.reserved_bytes, 64)
            for segment in allocator.snapshot():
                for left, right in zip(segment.blocks, segment.blocks[1:]):
                    self.assertFalse(left.state == right.state == "inactive")


if __name__ == "__main__":
    unittest.main()
