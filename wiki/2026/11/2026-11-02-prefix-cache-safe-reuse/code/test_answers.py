"""Prefix identity counterexamples and independently counted pool invariants."""

import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from importlib import import_module
from pathlib import Path

from models import PrefixContext

implementation = import_module(os.environ.get("EXERCISE_IMPL", "answers"))
block_hashes = implementation.block_hashes
reusable_tokens = implementation.reusable_tokens
PrefixPool = implementation.PrefixPool
CONTEXT = PrefixContext("toy-model@revision-1/position-v1/fp16", "base", "group-a")


class HashTests(unittest.TestCase):
    def test_common_eight_positions_reuse_two_blocks(self) -> None:
        cached = set(block_hashes(list(range(1, 11)), 4, CONTEXT))
        request = list(range(1, 9)) + [90, 91]
        self.assertEqual(reusable_tokens(request, 4, CONTEXT, cached), 8)

    def test_seven_common_positions_reuse_only_four(self) -> None:
        cached = set(block_hashes(list(range(1, 11)), 4, CONTEXT))
        request = list(range(1, 8)) + [90, 91, 92]
        self.assertEqual(reusable_tokens(request, 4, CONTEXT, cached), 4)

    def test_equal_local_block_with_different_parent_does_not_match(self) -> None:
        original = block_hashes(list(range(1, 13)), 4, CONTEXT)
        changed = block_hashes([90, 2, 3, 4] + list(range(5, 13)), 4, CONTEXT)
        self.assertTrue(all(a != b for a, b in zip(original, changed)))
        self.assertEqual(
            reusable_tokens(
                [90, 2, 3, 4] + list(range(5, 13)), 4, CONTEXT, set(original)
            ),
            0,
        )

    def test_changing_later_tokens_preserves_earlier_hashes(self) -> None:
        original = block_hashes(list(range(1, 13)), 4, CONTEXT)
        changed = block_hashes(list(range(1, 9)) + [90, 91, 92, 93], 4, CONTEXT)
        self.assertEqual(original[:2], changed[:2])
        self.assertNotEqual(original[2], changed[2])

    def test_context_changes_propagate_to_every_block(self) -> None:
        original = block_hashes(list(range(1, 13)), 4, CONTEXT)
        alternatives = (
            PrefixContext("toy-model@revision-2/position-v1/fp16", "base", "group-a"),
            PrefixContext(CONTEXT.namespace, "adapter-b", "group-a"),
            PrefixContext(CONTEXT.namespace, "base", "group-b"),
        )
        for context in alternatives:
            with self.subTest(context=context):
                changed = block_hashes(list(range(1, 13)), 4, context)
                self.assertTrue(all(a != b for a, b in zip(original, changed)))
                self.assertEqual(
                    reusable_tokens(list(range(1, 13)), 4, context, set(original)), 0
                )

    def test_block_size_is_part_of_teaching_namespace(self) -> None:
        keys = block_hashes(list(range(1, 9)), 4, CONTEXT)
        self.assertNotIn(block_hashes(list(range(1, 9)), 8, CONTEXT)[0], keys)

    def test_full_hit_leaves_a_position_for_logits_and_rounds_down(self) -> None:
        for length, expected in ((1, 0), (4, 0), (8, 4), (9, 8), (10, 8)):
            tokens = list(range(length))
            cached = set(block_hashes(tokens, 4, CONTEXT))
            with self.subTest(length=length):
                self.assertEqual(reusable_tokens(tokens, 4, CONTEXT, cached), expected)

    def test_lookup_stops_at_first_missing_block(self) -> None:
        tokens = list(range(13))
        a, b, c = block_hashes(tokens, 4, CONTEXT)
        self.assertEqual(reusable_tokens(tokens, 4, CONTEXT, {b, c}), 0)
        self.assertEqual(reusable_tokens(tokens, 4, CONTEXT, {a, c}), 4)

    def test_incomplete_tail_has_no_independent_hash(self) -> None:
        self.assertEqual(len(block_hashes(list(range(10)), 4, CONTEXT)), 2)
        self.assertEqual(block_hashes([1, 2, 3], 4, CONTEXT), ())

    def test_serialization_is_unambiguous(self) -> None:
        self.assertNotEqual(
            block_hashes([1, 23], 2, CONTEXT), block_hashes([12, 3], 2, CONTEXT)
        )

    def test_deterministic_across_python_hash_seeds(self) -> None:
        module = implementation.__name__
        script = (
            "from importlib import import_module; from models import PrefixContext; "
            "import sys; m=import_module(sys.argv[1]); "
            "print(m.block_hashes([1,2,3,4,5],2,PrefixContext('m','base','a')))"
        )
        results = []
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            results.append(
                subprocess.check_output(
                    [sys.executable, "-c", script, module],
                    cwd=Path(__file__).parent,
                    env=env,
                    text=True,
                )
            )
        self.assertEqual(*results)

    def test_invalid_inputs_are_rejected(self) -> None:
        for tokens in ([], (), [True], [-1], [1.5], "123", None):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                block_hashes(tokens, 4, CONTEXT)
        for size in (0, -1, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                block_hashes([1, 2], size, CONTEXT)
        with self.assertRaises(ValueError):
            block_hashes([1, 2], 2, "not-context")
        for cache in (["a"], {1}, {""}):
            with self.subTest(cache=cache), self.assertRaises(ValueError):
                reusable_tokens([1, 2], 2, CONTEXT, cache)

    def test_context_is_validated_and_immutable(self) -> None:
        for values in (("", "a", "b"), ("m", None, "b"), ("m", "a", "")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PrefixContext(*values)
        with self.assertRaises(FrozenInstanceError):
            CONTEXT.salt = "changed"

    def test_inputs_are_not_modified(self) -> None:
        tokens = list(range(10))
        cache = set(block_hashes(tokens, 4, CONTEXT))
        saved = (tokens.copy(), cache.copy())
        reusable_tokens(tokens, 4, CONTEXT, cache)
        self.assertEqual((tokens, cache), saved)


class PoolTests(unittest.TestCase):
    def assert_invariants(self, pool) -> None:
        view = pool.snapshot()
        self.assertEqual(len(view.free_order), len(set(view.free_order)))
        counts = [0] * len(view.blocks)
        for _, ids in view.requests:
            for block_id in ids:
                counts[block_id] += 1
        for block in view.blocks:
            self.assertEqual(block.ref_count, counts[block.block_id])
            self.assertEqual(block.block_id in view.free_order, block.ref_count == 0)
        entries = dict(view.cache_entries)
        indexed = set()
        for key, ids in entries.items():
            for block_id in ids:
                self.assertTrue(view.blocks[block_id].ready)
                self.assertEqual(view.blocks[block_id].key, key)
                indexed.add(block_id)
        self.assertEqual(indexed, {b.block_id for b in view.blocks if b.ready})

    def test_pending_blocks_cannot_be_reused(self) -> None:
        pool = PrefixPool(2)
        self.assertEqual(pool.acquire("r1", ("a",)), (0,))
        self.assertEqual(pool.snapshot().cache_entries, ())
        self.assertEqual(pool.acquire("r2", ("a",)), (1,))
        self.assert_invariants(pool)

    def test_live_ready_blocks_are_shared_with_reference_counts(self) -> None:
        pool = PrefixPool(2)
        pool.acquire("r1", ("a",))
        pool.complete("r1")
        self.assertEqual(pool.acquire("r2", ("a",)), (0,))
        self.assertEqual(pool.snapshot().blocks[0].ref_count, 2)
        pool.release("r1")
        self.assertEqual(pool.snapshot().blocks[0].ref_count, 1)
        self.assertNotIn(0, pool.snapshot().free_order)
        self.assert_invariants(pool)

    def test_release_keeps_ready_blocks_warm(self) -> None:
        pool = PrefixPool(2)
        ids = pool.acquire("r1", ("a",))
        pool.complete("r1")
        pool.release("r1")
        self.assertEqual(dict(pool.snapshot().cache_entries), {"a": (0,)})
        self.assertEqual(pool.snapshot().blocks[0].ref_count, 0)
        self.assertEqual(pool.acquire("r2", ("a",)), ids)
        self.assert_invariants(pool)

    def test_hits_are_touched_before_allocating_misses(self) -> None:
        pool = PrefixPool(3)
        self.assertEqual(pool.acquire("r1", ("a", "b")), (0, 1))
        pool.complete("r1")
        pool.release("r1")
        self.assertEqual(pool.snapshot().free_order, (2, 1, 0))
        self.assertEqual(pool.acquire("r2", ("a", "b", "c")), (0, 1, 2))
        self.assertEqual([b.ref_count for b in pool.snapshot().blocks], [1, 1, 1])
        self.assertEqual(pool.snapshot().free_order, ())
        self.assert_invariants(pool)

    def test_warm_eviction_removes_stale_index(self) -> None:
        pool = PrefixPool(1)
        pool.acquire("r1", ("a",))
        pool.complete("r1")
        pool.release("r1")
        pool.acquire("r2", ("b",))
        self.assertNotIn("a", dict(pool.snapshot().cache_entries))
        self.assertFalse(pool.snapshot().blocks[0].ready)
        pool.complete("r2")
        self.assertEqual(dict(pool.snapshot().cache_entries), {"b": (0,)})
        self.assert_invariants(pool)

    def test_capacity_failure_is_atomic_including_hit_touches(self) -> None:
        pool = PrefixPool(2)
        pool.acquire("r1", ("a", "b"))
        pool.complete("r1")
        pool.release("r1")
        before = pool.snapshot()
        with self.assertRaises(RuntimeError):
            pool.acquire("too-large", ("a", "b", "c"))
        self.assertEqual(pool.snapshot(), before)
        self.assert_invariants(pool)

    def test_pinned_capacity_cannot_be_evicted(self) -> None:
        pool = PrefixPool(1)
        pool.acquire("r1", ("a",))
        pool.complete("r1")
        before = pool.snapshot()
        with self.assertRaises(RuntimeError):
            pool.acquire("r2", ("b",))
        self.assertEqual(pool.snapshot(), before)

    def test_probe_stops_at_missing_prefix_even_if_suffix_is_warm(self) -> None:
        pool = PrefixPool(3)
        pool.acquire("r1", ("b",))
        pool.complete("r1")
        pool.release("r1")
        ids = pool.acquire("r2", ("a", "b"))
        self.assertNotEqual(ids[1], 0)
        self.assert_invariants(pool)

    def test_completed_duplicates_survive_single_copy_eviction(self) -> None:
        pool = PrefixPool(2)
        pool.acquire("r1", ("a",))
        pool.acquire("r2", ("a",))
        pool.complete("r1")
        pool.complete("r2")
        pool.release("r1")
        pool.release("r2")
        pool.acquire("r3", ("b",))
        self.assertEqual(dict(pool.snapshot().cache_entries), {"a": (1,)})
        self.assertEqual(pool.acquire("r4", ("a",)), (1,))
        self.assert_invariants(pool)

    def test_active_duplicate_is_preferred_to_preserve_free_capacity(self) -> None:
        pool = PrefixPool(2)
        pool.acquire("r1", ("a",))
        pool.acquire("r2", ("a",))
        pool.complete("r1")
        pool.complete("r2")
        pool.release("r1")
        self.assertEqual(pool.snapshot().free_order, (0,))
        self.assertEqual(pool.acquire("r3", ("a", "b")), (1, 0))
        self.assertEqual(pool.snapshot().blocks[1].ref_count, 2)
        self.assertEqual(dict(pool.snapshot().cache_entries), {"a": (1,)})
        self.assert_invariants(pool)

    def test_uncached_released_blocks_are_reused_before_warm_blocks(self) -> None:
        pool = PrefixPool(2)
        pool.acquire("warm", ("a",))
        pool.complete("warm")
        pool.acquire("pending", ("b",))
        pool.release("warm")
        pool.release("pending")
        self.assertEqual(pool.snapshot().free_order, (1, 0))
        self.assertEqual(pool.acquire("next", ("c",)), (1,))
        self.assertIn("a", dict(pool.snapshot().cache_entries))
        self.assert_invariants(pool)

    def test_completion_is_idempotent(self) -> None:
        pool = PrefixPool(1)
        pool.acquire("r1", ("a",))
        pool.complete("r1")
        before = pool.snapshot()
        pool.complete("r1")
        self.assertEqual(pool.snapshot(), before)

    def test_invalid_operations_leave_state_unchanged(self) -> None:
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                PrefixPool(capacity)
        pool = PrefixPool(2)
        pool.acquire("r1", ("a",))
        before = pool.snapshot()
        for request_id, keys in (
            ("", ("a",)),
            ("r1", ("b",)),
            ("r2", ()),
            ("r2", ("a", "a")),
            ("r2", ["b"]),
            ("r2", (1,)),
        ):
            with (
                self.subTest(request_id=request_id, keys=keys),
                self.assertRaises(ValueError),
            ):
                pool.acquire(request_id, keys)
            self.assertEqual(pool.snapshot(), before)
        for method in (pool.release, pool.complete):
            with self.assertRaises(ValueError):
                method("unknown")
            self.assertEqual(pool.snapshot(), before)

    def test_snapshots_are_immutable(self) -> None:
        pool = PrefixPool(1)
        view = pool.snapshot()
        with self.assertRaises(FrozenInstanceError):
            view.blocks[0].ref_count = 7
        self.assertEqual(pool.snapshot(), view)

    def test_shared_lifecycle_preserves_all_invariants(self) -> None:
        pool = PrefixPool(4)
        operations = (
            (pool.acquire, ("a", ("p", "q"))),
            (pool.complete, ("a",)),
            (pool.acquire, ("b", ("p", "q", "r"))),
            (pool.complete, ("b",)),
            (pool.release, ("a",)),
            (pool.release, ("b",)),
            (pool.acquire, ("c", ("s", "t"))),
            (pool.complete, ("c",)),
            (pool.release, ("c",)),
        )
        for method, args in operations:
            method(*args)
            self.assert_invariants(pool)


if __name__ == "__main__":
    unittest.main()
