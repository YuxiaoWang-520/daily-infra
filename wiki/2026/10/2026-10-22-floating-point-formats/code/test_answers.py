"""Acceptance cases for dtype conversion, accumulation and finite comparison."""

import math
import sys
import unittest
from decimal import localcontext

import torch

from answers import compare_finite, roundtrip, sequential_sum


class FormatTests(unittest.TestCase):
    def test_known_roundtrip_values(self):
        self.assertEqual(roundtrip([1.001], torch.float16).item(), 1.0009765625)
        self.assertEqual(roundtrip([1.001], torch.bfloat16).item(), 1.0)
        self.assertTrue(math.isinf(roundtrip([70000.0], torch.float16).item()))
        self.assertTrue(math.isfinite(roundtrip([70000.0], torch.bfloat16).item()))
        self.assertEqual(roundtrip([1e-8], torch.float16).item(), 0.0)

    def test_promoting_after_quantization_cannot_restore_input(self):
        value = roundtrip([1.001], torch.bfloat16)
        self.assertEqual(value.dtype, torch.float64)
        self.assertNotEqual(value.item(), 1.001)

    def test_dtype_info_and_one_spacing(self):
        for dtype, eps in [
            (torch.float32, 2.0**-23),
            (torch.float16, 2.0**-10),
            (torch.bfloat16, 2.0**-7),
        ]:
            self.assertEqual(torch.finfo(dtype).eps, eps)
            self.assertEqual(roundtrip([1.0 + eps], dtype).item(), 1.0 + eps)

    def test_addition_order(self):
        big = torch.tensor(1e8, dtype=torch.float32)
        small = torch.tensor(1.0, dtype=torch.float32)
        self.assertEqual(((big + small) - big).item(), 0.0)
        self.assertEqual(((big - big) + small).item(), 1.0)

    def test_larger_accumulator_reduces_this_accumulation_error(self):
        values = torch.full((1000,), 0.1, dtype=torch.float16)
        reference = values.to(torch.float64).sum().item()
        low = sequential_sum(values, torch.float16).item()
        high = sequential_sum(values, torch.float32).item()
        self.assertEqual(low, 105.1875)
        self.assertEqual(reference, 99.9755859375)
        self.assertLess(abs(high - reference), abs(low - reference))

    def test_empty_and_invalid_sum(self):
        self.assertEqual(sequential_sum(torch.empty(0), torch.float32).item(), 0.0)
        with self.assertRaises(ValueError):
            sequential_sum(torch.ones(2, 3), torch.float32)
        with self.assertRaises(ValueError):
            sequential_sum(torch.ones(2), torch.int64)
        with self.assertRaises(ValueError):
            roundtrip([1.0], torch.int64)


class ComparisonTests(unittest.TestCase):
    def test_near_zero_needs_absolute_tolerance(self):
        actual = torch.tensor([0.001], dtype=torch.float64)
        ref = torch.zeros(1, dtype=torch.float64)
        self.assertFalse(compare_finite(actual, ref, atol=0.0, rtol=1e-3).passed)
        result = compare_finite(actual, ref, atol=0.002, rtol=0.0)
        self.assertTrue(result.passed)
        self.assertTrue(math.isinf(result.relative_l2))

    def test_large_values_relative_tolerance(self):
        result = compare_finite(
            torch.tensor([1001.0]), torch.tensor([1000.0]), atol=0.0, rtol=0.002
        )
        self.assertEqual(result.max_absolute, 1.0)
        self.assertEqual(result.failed, 0)
        self.assertAlmostEqual(result.relative_l2, 0.001)

    def test_one_failure_not_hidden_by_aggregate(self):
        result = compare_finite(
            torch.tensor([1.0, 3.0]), torch.tensor([1.0, 2.0]), atol=0.0, rtol=0.0
        )
        self.assertEqual(result.total, 2)
        self.assertEqual(result.failed, 1)
        self.assertFalse(result.passed)

    def test_absolute_tolerance_boundary(self):
        report = compare_finite(
            torch.tensor([3.0]), torch.tensor([2.0]), atol=1.0, rtol=0.0
        )
        self.assertTrue(report.passed)

    def test_extreme_finite_values_do_not_compare_inf_to_inf(self):
        actual = torch.tensor([1e308], dtype=torch.float64)
        reference = -actual
        self.assertFalse(compare_finite(actual, reference, atol=0.0, rtol=1.9).passed)
        self.assertTrue(compare_finite(actual, reference, atol=0.0, rtol=2.1).passed)

    def test_overflow_does_not_hide_tolerance_margin(self):
        actual = torch.tensor([sys.float_info.max], dtype=torch.float64)
        reference = torch.tensor([-(3.0 * 2.0**969)], dtype=torch.float64)
        self.assertFalse(
            compare_finite(actual, reference, atol=sys.float_info.max, rtol=0.75).passed
        )

    def test_relative_norm_underflow_regressions(self):
        cases = [([1e200], [1.0], 1e200), ([1e200, 1e20], [1e200, 0.0], 1e-180)]
        for actual, reference, expected in cases:
            with self.subTest(expected=expected):
                report = compare_finite(
                    torch.tensor(actual, dtype=torch.float64),
                    torch.tensor(reference, dtype=torch.float64),
                    atol=0.0,
                    rtol=0.0,
                )
                self.assertTrue(
                    math.isclose(report.relative_l2, expected, rel_tol=1e-12)
                )

    def test_relative_norm_preserves_small_difference(self):
        actual = torch.tensor([3.0, 1.0000000000000002], dtype=torch.float64)
        reference = torch.tensor([3.0, 1.0], dtype=torch.float64)
        report = compare_finite(actual, reference, atol=0.0, rtol=0.0)
        self.assertTrue(
            math.isclose(report.relative_l2, 2.0**-52 / math.sqrt(10), rel_tol=1e-12)
        )

    def test_tolerance_is_exact_for_the_stored_binary_values(self):
        actual = torch.tensor([2.0**53], dtype=torch.float64)
        reference = torch.tensor([-1.0], dtype=torch.float64)
        self.assertFalse(
            compare_finite(actual, reference, atol=2.0**53, rtol=0.0).passed
        )

    def test_decimal_caller_context_is_preserved(self):
        with localcontext() as context:
            context.prec = 6
            context.Emax = 10
            result = compare_finite(
                torch.tensor([1e200], dtype=torch.float64),
                torch.tensor([1.0], dtype=torch.float64),
                atol=0.0,
                rtol=0.0,
            )
            self.assertTrue(math.isclose(result.relative_l2, 1e200, rel_tol=1e-12))
            self.assertEqual(context.prec, 6)
            self.assertEqual(context.Emax, 10)

    def test_reject_nonfinite_even_when_positions_match(self):
        for value in [float("inf"), float("-inf"), float("nan")]:
            result = compare_finite(
                torch.tensor([value]), torch.tensor([value]), atol=0.0, rtol=0.0
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.nonfinite, 1)
            self.assertEqual(result.failed, 1)
            self.assertTrue(math.isinf(result.max_absolute))

    def test_empty_and_all_zero(self):
        for values in [torch.empty(0), torch.zeros(3)]:
            result = compare_finite(values, values, atol=0.0, rtol=0.0)
            self.assertTrue(result.passed)
            self.assertEqual(result.max_absolute, 0.0)
            self.assertEqual(result.relative_l2, 0.0)

    def test_invalid_shape_dtype_and_tolerance(self):
        with self.assertRaises(ValueError):
            compare_finite(torch.ones(2), torch.ones(3), atol=0.0, rtol=0.0)
        with self.assertRaises(ValueError):
            compare_finite(
                torch.ones(2, dtype=torch.int64), torch.ones(2), atol=0.0, rtol=0.0
            )
        for bad in [-1.0, float("nan"), float("inf")]:
            with self.subTest(tolerance=bad), self.assertRaises(ValueError):
                compare_finite(torch.ones(2), torch.ones(2), atol=bad, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
