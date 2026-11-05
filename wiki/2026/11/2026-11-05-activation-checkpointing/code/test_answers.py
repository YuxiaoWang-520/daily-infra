"""Independent hand counts for a teaching proxy, not a CUDA memory estimator."""

import importlib
import math
import os
import unittest
from dataclasses import FrozenInstanceError

impl = importlib.import_module(os.environ.get("EXERCISE_IMPL", "answers"))


class UniformPlanTests(unittest.TestCase):
    def plan(self, **overrides):
        arguments = {
            "layers": 16,
            "segment_size": 4,
            "activation_bytes": 1024**2,
            "forward_us": 16.0,
            "backward_us": 32.0,
            "overhead_us": 0.2,
        }
        arguments.update(overrides)
        return impl.estimate_uniform(**arguments)

    def test_hand_counted_four_segments(self):
        plan = self.plan()
        self.assertEqual((plan.segment_size, plan.segments), (4, 4))
        self.assertEqual(
            (plan.activation_slots, plan.activation_bytes), (8, 8 * 1024**2)
        )
        self.assertEqual(plan.baseline_us, 48.0)
        self.assertAlmostEqual(plan.checkpoint_us, 64.8)
        self.assertAlmostEqual(plan.slowdown, 1.35)

    def test_tail_segment_uses_ceiling(self):
        plan = self.plan(layers=17)
        self.assertEqual((plan.segments, plan.activation_slots), (5, 9))

    def test_fine_and_coarse_extremes_do_not_guarantee_savings(self):
        for size in (1, 16):
            with self.subTest(size=size):
                self.assertEqual(self.plan(segment_size=size).activation_slots, 17)

    def test_one_layer_proxy_includes_two_slots(self):
        plan = self.plan(layers=1, segment_size=1)
        self.assertEqual((plan.segments, plan.activation_slots), (1, 2))

    def test_large_integer_bytes_do_not_round(self):
        plan = self.plan(layers=10**12, segment_size=10**6, activation_bytes=10**12)
        self.assertEqual(plan.activation_bytes, 2 * 10**18)

    def test_arbitrary_integer_layers_with_zero_overhead(self):
        plan = self.plan(
            layers=10**309,
            segment_size=1,
            activation_bytes=1,
            forward_us=1,
            backward_us=1,
            overhead_us=0,
        )
        self.assertEqual(plan.activation_bytes, 10**309 + 1)
        self.assertEqual(plan.checkpoint_us, 3)

    def test_huge_count_times_small_overhead_can_remain_finite(self):
        plan = self.plan(
            layers=10**309,
            segment_size=1,
            forward_us=1,
            backward_us=1,
            overhead_us=1e-309,
        )
        self.assertAlmostEqual(plan.checkpoint_us, 4)

    def test_segment_overhead_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(layers=10**309, segment_size=1, overhead_us=1)

    def test_no_overhead_gives_four_thirds_for_b_equals_two_f(self):
        self.assertAlmostEqual(self.plan(overhead_us=0).slowdown, 4 / 3)

    def test_non_two_to_one_backward_changes_the_ratio(self):
        self.assertAlmostEqual(self.plan(backward_us=16, overhead_us=0).slowdown, 1.5)

    def test_result_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.plan().segments = 0

    def test_invalid_sizes(self):
        for overrides in (
            {"layers": 0},
            {"layers": True},
            {"layers": 2.5},
            {"segment_size": 0},
            {"segment_size": 17},
            {"segment_size": False},
            {"activation_bytes": -1},
            {"activation_bytes": 1.5},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.plan(**overrides)

    def test_invalid_times(self):
        for name in ("forward_us", "backward_us", "overhead_us"):
            for value in (-1, True, math.inf, math.nan, "slow"):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    self.plan(**{name: value})
        with self.assertRaises(ValueError):
            self.plan(forward_us=0)

    def test_combined_duration_overflow_is_rejected(self):
        for duration in (1e308, 10**308):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.plan(forward_us=duration, backward_us=duration)


class SelectionTests(unittest.TestCase):
    def choose(self, budget, overhead=0.2):
        return impl.choose_plan(16, 1024**2, budget * 1024**2, 16, 32, overhead)

    def test_exact_budget(self):
        self.assertEqual(self.choose(8).segment_size, 4)

    def test_larger_budget_reduces_per_segment_overhead(self):
        plan = self.choose(10)
        self.assertEqual(
            (plan.segment_size, plan.segments, plan.activation_slots), (8, 2, 10)
        )
        self.assertAlmostEqual(plan.checkpoint_us, 64.4)

    def test_no_feasible_plan(self):
        self.assertIsNone(self.choose(7))
        self.assertIsNone(self.choose(0))

    def test_no_overhead_tie_prefers_less_memory(self):
        self.assertEqual(self.choose(100, overhead=0).segment_size, 4)

    def test_invalid_budget(self):
        for budget in (-1, True, 0.5, "8"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                impl.choose_plan(16, 1024, budget, 16, 32)


if __name__ == "__main__":
    unittest.main()
