"""Hand-computed storage and communication cases, without device measurements."""

import importlib
import os
import unittest
from dataclasses import FrozenInstanceError

impl = importlib.import_module(os.environ.get("EXERCISE_IMPL", "answers"))


class StateTests(unittest.TestCase):
    def test_billion_parameters_eight_ranks(self):
        for stage, expected in enumerate(
            (16_000_000_000, 5_500_000_000, 3_750_000_000, 2_000_000_000)
        ):
            with self.subTest(stage=stage):
                self.assertEqual(
                    impl.state_budget(10**9, 8, stage).total_bytes, expected
                )

    def test_individual_state_ownership(self):
        expected = ((16, 16, 96), (16, 16, 24), (16, 4, 24), (4, 4, 24))
        for stage, sizes in enumerate(expected):
            result = impl.state_budget(8, 4, stage)
            self.assertEqual(
                (result.parameter_bytes, result.gradient_bytes, result.optimizer_bytes),
                sizes,
            )

    def test_uniform_padding_for_uneven_partition(self):
        result = impl.state_budget(5, 2, 3)
        self.assertEqual(
            (result.shard_elements, result.padded_elements, result.total_bytes),
            (3, 6, 48),
        )

    def test_replicated_states_use_real_count(self):
        result = impl.state_budget(5, 2, 1)
        self.assertEqual(
            (result.parameter_bytes, result.gradient_bytes, result.optimizer_bytes),
            (10, 10, 36),
        )

    def test_empty_logical_shards_still_have_padding_slots(self):
        result = impl.state_budget(1, 4, 3)
        self.assertEqual(
            (result.shard_elements, result.padded_elements, result.total_bytes),
            (1, 4, 16),
        )

    def test_one_rank_has_identical_budgets(self):
        self.assertEqual(
            {impl.state_budget(7, 1, stage).total_bytes for stage in range(4)}, {112}
        )

    def test_large_integer_arithmetic_is_exact(self):
        result = impl.state_budget(10**100 + 1, 3, 3)
        self.assertEqual(result.total_bytes, ((10**100 + 3) // 3) * 16)

    def test_result_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            impl.state_budget(5, 2, 3).total_bytes = 0

    def test_invalid_counts_and_stage(self):
        for values in (
            (0, 2, 3),
            (5, 0, 3),
            (True, 2, 3),
            (5, 2.0, 3),
            (5, 2, True),
            (5, 2, -1),
            (5, 2, 4),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                impl.state_budget(*values)

    def test_invalid_state_widths(self):
        for key in ("parameter_bytes", "gradient_bytes", "optimizer_bytes"):
            for value in (0, -1, True, 2.0):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    impl.state_budget(5, 2, 3, **{key: value})


class RingTests(unittest.TestCase):
    def test_two_gathers_equal_widths(self):
        result = impl.ring_budget(8, 4)
        self.assertEqual(
            (
                result.all_gather_sent_bytes,
                result.reduce_scatter_sent_bytes,
                result.total_sent_bytes,
                result.ddp_sent_bytes,
            ),
            (24, 12, 36, 24),
        )

    def test_one_gather_retains_parameters(self):
        result = impl.ring_budget(8, 4, all_gather_calls=1)
        self.assertEqual((result.all_gather_calls, result.total_sent_bytes), (1, 24))

    def test_mixed_reduction_width_changes_ratio(self):
        result = impl.ring_budget(8, 4, gradient_bytes=4)
        self.assertEqual((result.total_sent_bytes, result.ddp_sent_bytes), (48, 48))

    def test_padding_is_included_in_ring_volume(self):
        result = impl.ring_budget(5, 2)
        self.assertEqual((result.total_sent_bytes, result.ddp_sent_bytes), (18, 12))

    def test_single_rank_has_no_network_send(self):
        self.assertEqual(impl.ring_budget(5, 1).total_sent_bytes, 0)

    def test_invalid_gather_count_and_widths(self):
        for options in (
            {"all_gather_calls": 0},
            {"all_gather_calls": 3},
            {"all_gather_calls": True},
            {"parameter_bytes": 0},
            {"gradient_bytes": False},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                impl.ring_budget(5, 2, **options)


if __name__ == "__main__":
    unittest.main()
