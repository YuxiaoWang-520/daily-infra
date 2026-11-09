"""Verify shard ownership and Adam equivalence, without timing assertions."""

import importlib
import os
import unittest

impl = importlib.import_module(os.environ.get("PROBE_IMPL", "sharded_probe"))


class DistributedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = impl.run_probe()

    def cases(self, name):
        return [
            case
            for rank in self.report["ranks"]
            for case in rank["cases"]
            if case["normalization"] == name
        ]

    def test_two_real_ranks_and_gloo(self):
        self.assertEqual(self.report["backend"], "gloo")
        self.assertEqual([rank["rank"] for rank in self.report["ranks"]], [0, 1])

    def test_sample_weighted_adam_matches_reference(self):
        for case in self.cases("sample_weighted"):
            for key in (
                "parameter_error",
                "first_moment_error",
                "second_moment_error",
                "gradient_error",
            ):
                with self.subTest(rank=case["rank"], key=key):
                    self.assertLessEqual(case[key], 1e-12)

    def test_unequal_counts_use_global_denominator(self):
        cases = self.cases("sample_weighted")
        self.assertEqual([case["local_samples"] for case in cases], [1, 3])
        self.assertEqual([case["global_samples"] for case in cases], [4, 4])

    def test_first_gradient_matches_hand_calculation(self):
        expected = (0.175, 0.4125, 0.525, 0.7625, 0.9375)
        for case in self.cases("sample_weighted"):
            for actual, value in zip(
                case["first_global_gradient"], expected, strict=True
            ):
                self.assertAlmostEqual(actual, value, places=12)

    def test_rank_mean_negative_control_is_detected(self):
        for case in self.cases("rank_mean"):
            self.assertGreater(case["gradient_error"], 1e-4)
            self.assertGreater(case["first_moment_error"], 1e-4)

    def test_only_local_adam_vector_states_persist(self):
        for case in self.cases("sample_weighted"):
            self.assertEqual(case["shard_elements"], 3)
            self.assertEqual(case["local_parameter_and_moments_elements"], 9)
            self.assertEqual(case["local_gradient_elements"], 3)
            self.assertTrue(case["padding_zero"])

    def test_training_collectives_have_consistent_order(self):
        expected = ["all_gather_parameters", "reduce_scatter_gradients"] * 3
        for rank in self.report["ranks"]:
            for case in rank["cases"]:
                self.assertEqual(case["training_collectives"], expected)


class InputTests(unittest.TestCase):
    def test_invalid_step_count_fails_before_spawn(self):
        for steps in (0, -1, True, 1.5, 11):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                impl.run_probe(steps)


if __name__ == "__main__":
    unittest.main()
