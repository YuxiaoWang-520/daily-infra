"""CPU acceptance tests for both exercises and the CUDA availability probe."""

import math
import unittest

import torch
from cuda_demo import check_output, run_demo
from launch_model import estimate_replays
from static_buffers import StaticAffine


class LaunchModelTests(unittest.TestCase):
    def test_launch_bound_example(self):
        result = estimate_replays([1] * 10, 3, 4, 10000, 1000)
        self.assertEqual(result.eager_us, 30001)
        self.assertEqual(result.graph_us, 20004)
        self.assertEqual(result.device_us, 10)
        self.assertEqual(result.eager_steady_us, 30)
        self.assertEqual(result.graph_steady_us, 10)
        self.assertEqual(result.first_faster_replays, 501)

    def test_strict_break_even_boundary(self):
        before = estimate_replays([1] * 10, 3, 4, 10000, 500)
        after = estimate_replays([1] * 10, 3, 4, 10000, 501)
        self.assertGreaterEqual(before.graph_us, before.eager_us)
        self.assertLess(after.graph_us, after.eager_us)

    def test_equal_time_is_not_faster(self):
        result = estimate_replays([1], 3, 1, 6, 3)
        self.assertEqual(result.graph_us, result.eager_us)
        self.assertEqual(result.first_faster_replays, 4)

    def test_compute_bound(self):
        result = estimate_replays([100] * 10, 3, 4, 10000, 1000)
        self.assertEqual(result.eager_us, 1000003)
        self.assertEqual(result.graph_us, 1010004)
        self.assertIsNone(result.first_faster_replays)

    def test_first_replay_already_faster(self):
        self.assertEqual(estimate_replays([1, 1], 3, 1, 0, 1).first_faster_replays, 1)

    def test_all_zero_costs(self):
        result = estimate_replays([0, 0], 0, 0, 0, 100)
        self.assertEqual(result.eager_us, 0)
        self.assertEqual(result.graph_us, 0)
        self.assertIsNone(result.first_faster_replays)

    def test_worse_graph_submission(self):
        result = estimate_replays([1], 1, 3, 0, 10)
        self.assertGreater(result.graph_us, result.eager_us)
        self.assertIsNone(result.first_faster_replays)

    def test_closed_form_matches_queue_simulation(self):
        for kernels in ([1, 8, 2], [9, 0, 1], [0, 4], [2.5, 0.25]):
            for host in (0, 1, 3, 8):
                for repeats in (1, 2, 7):
                    with self.subTest(kernels=kernels, host=host, repeats=repeats):
                        host_clock = 0.0
                        device_clock = 0.0
                        for _ in range(repeats):
                            for duration in kernels:
                                host_clock += host
                                device_clock = max(device_clock, host_clock) + duration
                        result = estimate_replays(kernels, host, 2, 3, repeats)
                        self.assertEqual(result.eager_us, device_clock)

    def test_graph_queue_formula(self):
        result = estimate_replays([1, 2], 2, 5, 7, 4)
        self.assertEqual(result.graph_us, 30)

    def test_empty_kernels(self):
        with self.assertRaises(ValueError):
            estimate_replays([], 1, 1, 1, 1)

    def test_invalid_replays(self):
        for value in (0, -1, True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimate_replays([1], 1, 1, 1, value)

    def test_invalid_costs(self):
        for value in (-1, math.nan, math.inf, True, "3", 10**400):
            for position in range(4):
                args = [[1], 1, 1, 1, 2]
                args[position] = [value] if position == 0 else value
                with (
                    self.subTest(value=str(value), position=position),
                    self.assertRaises(ValueError),
                ):
                    estimate_replays(*args)

    def test_output_overflow(self):
        with self.assertRaises(ValueError):
            estimate_replays([1e308, 1e308], 1, 1, 1, 2)

    def test_large_finite_costs(self):
        result = estimate_replays([1e307], 1e307, 1e307, 0, 2)
        self.assertTrue(math.isfinite(result.eager_us))
        self.assertEqual(result.eager_us, 3e307)

    def test_huge_replay_count_with_zero_cost(self):
        self.assertEqual(estimate_replays([0], 0, 0, 0, 10**400).eager_us, 0)

    def test_result_is_immutable(self):
        from dataclasses import FrozenInstanceError

        with self.assertRaises(FrozenInstanceError):
            estimate_replays([1], 1, 1, 1, 1).eager_us = 10


class StaticBufferTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def test_values(self):
        buffers = StaticAffine(self.source, 2, 1)
        torch.testing.assert_close(buffers.replay(), self.source * 2 + 1)

    def test_independent_input_storage(self):
        buffers = StaticAffine(self.source, 2, 1)
        self.assertNotEqual(buffers.input_buffer.data_ptr(), self.source.data_ptr())
        self.source.fill_(99)
        torch.testing.assert_close(
            buffers.replay(), torch.tensor([[1.0, 3.0, 5.0], [7.0, 9.0, 11.0]])
        )

    def test_pointer_stability(self):
        buffers = StaticAffine(self.source, 2, 1)
        pointers = (buffers.input_buffer.data_ptr(), buffers.output_buffer.data_ptr())
        for value in (3, 4, 5):
            buffers.copy_input(torch.full_like(self.source, value))
            buffers.replay()
            self.assertEqual(
                pointers,
                (buffers.input_buffer.data_ptr(), buffers.output_buffer.data_ptr()),
            )

    def test_rebinding_does_not_change_retained_input(self):
        source = self.source
        buffers = StaticAffine(source, 2, 1)
        source = torch.full_like(source, 7)
        torch.testing.assert_close(buffers.replay(), self.source * 2 + 1)
        buffers.copy_input(source)
        torch.testing.assert_close(buffers.replay(), source * 2 + 1)

    def test_borrowed_output_and_clone(self):
        buffers = StaticAffine(self.source, 2, 1)
        borrowed = buffers.replay()
        saved = borrowed.clone()
        buffers.copy_input(torch.full_like(self.source, 4))
        second = buffers.replay()
        self.assertIs(borrowed, second)
        self.assertTrue(torch.all(borrowed == 9).item())
        torch.testing.assert_close(saved, self.source * 2 + 1)

    def test_noncontiguous_source(self):
        source = torch.arange(9, dtype=torch.float32).reshape(3, 3).t()
        buffers = StaticAffine(torch.zeros_like(source), 2, 1)
        buffers.copy_input(source)
        self.assertTrue(buffers.input_buffer.is_contiguous())
        torch.testing.assert_close(buffers.replay(), source * 2 + 1)

    def test_overlapping_transposed_input(self):
        original = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        buffers = StaticAffine(original, 2, 1)
        pointer = buffers.input_buffer.data_ptr()
        source = buffers.input_buffer.t()
        expected = source.clone()
        buffers.copy_input(source)
        torch.testing.assert_close(buffers.input_buffer, expected)
        self.assertEqual(buffers.input_buffer.data_ptr(), pointer)

    def test_invalid_shape_preserves_input(self):
        buffers = StaticAffine(self.source, 2, 1)
        with self.assertRaises(ValueError):
            buffers.copy_input(torch.ones(3, 2))
        torch.testing.assert_close(buffers.input_buffer, self.source)

    def test_invalid_dtype_preserves_input(self):
        buffers = StaticAffine(self.source, 2, 1)
        with self.assertRaises(ValueError):
            buffers.copy_input(self.source.double())
        torch.testing.assert_close(buffers.input_buffer, self.source)

    def test_meta_device_rejected_before_copy(self):
        buffers = StaticAffine(self.source, 2, 1)
        with self.assertRaises(ValueError):
            buffers.copy_input(torch.empty((2, 3), device="meta"))
        torch.testing.assert_close(buffers.input_buffer, self.source)

    def test_invalid_examples(self):
        for example in (
            None,
            torch.ones(3),
            torch.empty(0, 3),
            torch.ones(2, 3, dtype=torch.int64),
        ):
            with self.subTest(example=str(example)), self.assertRaises(ValueError):
                StaticAffine(example, 2, 1)

    def test_sparse_example_rejected(self):
        with self.assertRaises(ValueError):
            StaticAffine(self.source.to_sparse(), 2, 1)

    def test_non_tensor_copy_rejected(self):
        with self.assertRaises(ValueError):
            StaticAffine(self.source, 2, 1).copy_input([[1, 2, 3], [4, 5, 6]])

    def test_invalid_scalars(self):
        for value in (True, math.inf, math.nan, "2", 10**400):
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                StaticAffine(self.source, value, 1)

    def test_float64_and_negative_scale(self):
        buffers = StaticAffine(self.source.double(), -0.5, 0)
        torch.testing.assert_close(buffers.replay(), self.source.double() * -0.5)

    def test_forward_only(self):
        buffers = StaticAffine(self.source.requires_grad_(), 2, 1)
        self.assertFalse(buffers.input_buffer.requires_grad)
        self.assertFalse(buffers.replay().requires_grad)

    def test_python_scalar_rebinding(self):
        scale = 2
        buffers = StaticAffine(self.source, scale, 1)
        scale = 7
        torch.testing.assert_close(buffers.replay(), self.source * 2 + 1)


class CUDADemoTests(unittest.TestCase):
    def test_expected_dtype_independent_of_global_default(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            check_output(torch.full((2, 3), 7, dtype=torch.float32, device="cpu"), 3)
        finally:
            torch.set_default_dtype(previous)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(AssertionError):
            check_output(torch.full((1, 6), 7, dtype=torch.float32, device="cpu"), 3)

    def test_expected_device_independent_of_default_context(self):
        actual = torch.full((2, 3), 7, dtype=torch.float32, device="cpu")
        with torch.device("meta"):
            check_output(actual, 3)

    def test_unavailable_reports_pending(self):
        from unittest.mock import patch

        with patch("torch.cuda.is_available", return_value=False):
            result = run_demo()
        self.assertFalse(result["cuda_executed"])
        self.assertEqual(result["status"], "cuda_unavailable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
