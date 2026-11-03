"""Independent arithmetic cases and dependency checks, without GPUs."""

import importlib
import math
import os
import random
import unittest
from dataclasses import FrozenInstanceError

impl = importlib.import_module(os.environ.get("EXERCISE_IMPL", "answers"))
MIB = 1024 * 1024


class MeanTests(unittest.TestCase):
    def test_equal_counts(self):
        self.assertEqual(impl.global_mean_from_local_means([[1], [9]], [1, 1]), (5,))

    def test_unequal_counts(self):
        result = impl.global_mean_from_local_means([[1], [29 / 3]], [1, 3])
        self.assertAlmostEqual(result[0], 7.5)
        self.assertNotAlmostEqual(result[0], (1 + 29 / 3) / 2)

    def test_vector_gradient(self):
        self.assertEqual(
            impl.global_mean_from_local_means([[1, 2], [9, -2]], [1, 3]), (7, -1)
        )

    def test_loss_scales(self):
        self.assertEqual(impl.ddp_loss_scales([1, 3]), (0.5, 1.5))
        self.assertEqual(impl.ddp_loss_scales([2, 2, 2]), (1, 1, 1))

    def test_scaled_rank_average_matches_global_objective(self):
        means = (1, 29 / 3)
        scales = impl.ddp_loss_scales([1, 3])
        actual = sum(s * g for s, g in zip(scales, means)) / 2
        self.assertAlmostEqual(actual, 7.5)

    def test_partitioned_samples(self):
        rng = random.Random(17)
        groups = [[rng.uniform(-4, 4) for _ in range(n)] for n in (2, 5, 3)]
        local = [[math.fsum(group) / len(group)] for group in groups]
        result = impl.global_mean_from_local_means(local, [2, 5, 3])
        expected = math.fsum(x for group in groups for x in group) / 10
        self.assertAlmostEqual(result[0], expected)

    def test_single_rank(self):
        self.assertEqual(impl.global_mean_from_local_means([[2, -3]], [8]), (2, -3))
        self.assertEqual(impl.ddp_loss_scales([8]), (1,))

    def test_inputs_unchanged(self):
        means, counts = [[1, 2], [3, 4]], [2, 3]
        impl.global_mean_from_local_means(means, counts)
        impl.ddp_loss_scales(counts)
        self.assertEqual((means, counts), ([[1, 2], [3, 4]], [2, 3]))

    def test_invalid_counts(self):
        for counts in ([], [0], [-1], [True], [1.5], "1"):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                impl.ddp_loss_scales(counts)

    def test_invalid_gradient_shapes(self):
        for means, counts in (
            ([], []),
            ([[]], [1]),
            ([[1], [2, 3]], [1, 1]),
            ([[1]], [1, 2]),
        ):
            with self.subTest(means=means), self.assertRaises(ValueError):
                impl.global_mean_from_local_means(means, counts)

    def test_invalid_gradient_values(self):
        for value in (True, "1", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                impl.global_mean_from_local_means([[value]], [1])


class TimelineTests(unittest.TestCase):
    def plan(self, buckets, ready=None, alpha=2):
        return impl.schedule_buckets(
            ready or [[2, 4, 6, 8], [2, 4, 6, 8]],
            [4 * MIB] * 4,
            buckets,
            alpha,
            4 * MIB,
        )

    def test_two_buckets_hand_calculation(self):
        result = self.plan([[0, 1], [2, 3]])
        self.assertEqual(
            [(e.ready_ms, e.start_ms, e.finish_ms) for e in result.events],
            [(4, 4, 8), (8, 8, 12)],
        )
        self.assertEqual(
            (result.backward_ms, result.finish_ms, result.tail_ms, result.serial_ms),
            (8, 12, 4, 16),
        )

    def test_smaller_is_not_always_faster(self):
        plans = [
            self.plan(b)
            for b in ([[0], [1], [2], [3]], [[0, 1], [2, 3]], [[0, 1, 2, 3]])
        ]
        self.assertEqual([p.finish_ms for p in plans], [14, 12, 14])
        self.assertEqual([p.serial_ms for p in plans], [20, 16, 14])

    def test_straggler_limits_progress(self):
        result = self.plan([[0, 1], [2, 3]], [[2, 4, 6, 8], [2, 4, 6, 12]])
        self.assertEqual(
            (result.backward_ms, result.finish_ms, result.tail_ms), (12, 16, 4)
        )

    def test_rank_readiness_is_maximum(self):
        result = self.plan([[0], [1], [2], [3]], [[1, 1, 1, 1], [7, 7, 7, 7]])
        self.assertEqual(result.events[0].start_ms, 7)

    def test_order_blocks_early_ready_bucket(self):
        result = self.plan([[3], [0], [1], [2]])
        self.assertEqual(result.events[1].ready_ms, 2)
        self.assertEqual(result.events[1].start_ms, 11)
        self.assertEqual(result.finish_ms, 20)

    def test_one_rank_one_gradient(self):
        result = impl.schedule_buckets([[3]], [8], [[0]], 1, 4)
        self.assertEqual((result.finish_ms, result.serial_ms), (6, 6))

    def test_zero_startup_allowed(self):
        result = self.plan([[0, 1], [2, 3]], alpha=0)
        self.assertEqual(result.finish_ms, 10)

    def test_dependency_invariants(self):
        ready = [[2, 6, 4, 7], [3, 4, 5, 8]]
        result = self.plan([[2, 0], [3, 1]], ready)
        previous_end = 0
        for event in result.events:
            expected_ready = max(
                row[i] for row in ready for i in event.gradient_indices
            )
            self.assertEqual(event.ready_ms, expected_ready)
            self.assertGreaterEqual(event.start_ms, expected_ready)
            self.assertGreaterEqual(event.start_ms, previous_end)
            self.assertEqual(
                event.finish_ms - event.start_ms, 2 + event.size_bytes / (4 * MIB)
            )
            previous_end = event.finish_ms
        self.assertGreaterEqual(result.finish_ms, max(map(max, ready)))
        self.assertEqual(result.finish_ms, previous_end)

    def test_no_input_mutation_and_immutable_result(self):
        ready, sizes, buckets = [[1, 2]], [4, 8], [[0], [1]]
        result = impl.schedule_buckets(ready, sizes, buckets, 1, 4)
        self.assertEqual((ready, sizes, buckets), ([[1, 2]], [4, 8], [[0], [1]]))
        with self.assertRaises(FrozenInstanceError):
            result.finish_ms = 1
        with self.assertRaises(FrozenInstanceError):
            result.events[0].start_ms = 1

    def test_partition_must_cover_each_gradient_once(self):
        for buckets in (
            [[0], [1]],
            [[0, 1], [1, 2, 3]],
            [[], [0, 1, 2, 3]],
            [[0, 1, 2, 4]],
            [[True, 1, 2, 3]],
        ):
            with self.subTest(buckets=buckets), self.assertRaises(ValueError):
                self.plan(buckets)

    def test_invalid_ready_times(self):
        for ready in (
            [],
            [[1]],
            [[1, 2, 3, -1]],
            [[1, 2, 3, float("nan")]],
            [[1, 2, 3, True]],
        ):
            with self.subTest(ready=ready), self.assertRaises(ValueError):
                impl.schedule_buckets(ready, [4 * MIB] * 4, [[0, 1, 2, 3]], 2, 4 * MIB)

    def test_invalid_sizes_and_communication_parameters(self):
        for sizes, alpha, rate in (
            ([0], 1, 4),
            ([True], 1, 4),
            ([8], -1, 4),
            ([8], 1, 0),
            ([8], float("inf"), 4),
            ([8], 1, True),
        ):
            with (
                self.subTest(sizes=sizes, alpha=alpha, rate=rate),
                self.assertRaises(ValueError),
            ):
                impl.schedule_buckets([[1]], sizes, [[0]], alpha, rate)

    def test_derived_time_overflow_uses_value_error(self):
        with self.assertRaises(ValueError):
            impl.schedule_buckets([[0]], [10**400], [[0]], 0, 1)
        with self.assertRaises(ValueError):
            impl.schedule_buckets([[0]], [4], [[0]], 0, 1e-308)


if __name__ == "__main__":
    unittest.main()
