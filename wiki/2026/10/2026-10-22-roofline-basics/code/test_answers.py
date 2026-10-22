"""Small exact workloads and unit checks for an illustrative Roofline model."""

import math
import unittest

from answers import Work, effective_rates, estimate, gemm_work, vector_add_work


class WorkTests(unittest.TestCase):
    def test_vector_add_counts_and_empty(self):
        self.assertEqual(vector_add_work(4, 4), Work(4, 48))
        self.assertEqual(vector_add_work(0, 2), Work(0, 0))

    def test_gemm_non_square_counts(self):
        self.assertEqual(gemm_work(2, 3, 4, 4), Work(48, 104))

    def test_gemm_square_intensity(self):
        work = gemm_work(1024, 1024, 1024, 4)
        self.assertEqual(work.flops, 2147483648)
        self.assertEqual(work.bytes_moved, 12582912)
        self.assertAlmostEqual(work.flops / work.bytes_moved, 1024 / 6)

    def test_reject_invalid_counts(self):
        for n in [-1, True, 2.5]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                vector_add_work(n, 4)
        for size in [0, -1, True]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                vector_add_work(4, size)
        with self.assertRaises(ValueError):
            gemm_work(0, 2, 3, 4)
        with self.assertRaises(ValueError):
            Work(-1, 1)
        with self.assertRaises(ValueError):
            Work(1, 0)


class BoundTests(unittest.TestCase):
    def test_vector_add_memory_bound(self):
        result = estimate(
            vector_add_work(2**20, 4), peak_tflops=100.0, bandwidth_gbs=1000.0
        )
        self.assertEqual(result.limiter, "memory")
        self.assertAlmostEqual(result.intensity, 1 / 12)
        self.assertEqual(result.ridge_intensity, 100.0)
        self.assertAlmostEqual(result.lower_seconds, 12582912 / 1e12)
        self.assertAlmostEqual(result.attainable_tflops, 1 / 12)

    def test_large_gemm_compute_bound(self):
        result = estimate(gemm_work(1024, 1024, 1024, 4), 100.0, 1000.0)
        self.assertEqual(result.limiter, "compute")
        self.assertAlmostEqual(result.lower_seconds, 2147483648 / 1e14)
        self.assertEqual(result.attainable_tflops, 100.0)

    def test_small_gemm_can_be_memory_bound_in_model(self):
        result = estimate(gemm_work(128, 128, 128, 4), 100.0, 1000.0)
        self.assertEqual(result.limiter, "memory")
        self.assertAlmostEqual(result.intensity, 128 / 6)

    def test_balanced_and_empty(self):
        result = estimate(Work(100, 1), 100.0, 1000.0)
        self.assertEqual(result.limiter, "balanced")
        empty = estimate(Work(0, 0), 100.0, 1000.0)
        self.assertEqual(empty.limiter, "empty")
        self.assertEqual(empty.lower_seconds, 0.0)
        self.assertEqual(empty.attainable_tflops, 0.0)

    def test_copy_only_work(self):
        result = estimate(Work(0, 1024), 100.0, 1000.0)
        self.assertEqual(result.limiter, "memory")
        self.assertEqual(result.attainable_tflops, 0.0)

    def test_units_of_effective_rates(self):
        rates = effective_rates(Work(10**12, 10**9), 0.5)
        self.assertEqual(rates["tflops"], 2.0)
        self.assertEqual(rates["effective_gbs"], 2.0)

    def test_synthetic_time_twice_bound(self):
        work = gemm_work(1024, 1024, 1024, 4)
        bound = estimate(work, 100.0, 1000.0)
        rates = effective_rates(work, 2 * bound.lower_seconds)
        self.assertEqual(rates["tflops"], 50.0)

    def test_invalid_hardware_and_elapsed(self):
        for bad in [0.0, -1.0, math.inf, math.nan, True]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                estimate(Work(1, 12), bad, 1000.0)
            with self.subTest(value=bad), self.assertRaises(ValueError):
                estimate(Work(1, 12), 100.0, bad)
            with self.subTest(value=bad), self.assertRaises(ValueError):
                effective_rates(Work(1, 12), bad)


if __name__ == "__main__":
    unittest.main()
