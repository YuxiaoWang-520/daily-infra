"""Tiled CPU correctness and exact logical-load accounting."""

import unittest

import torch

from answers import blocked_matmul, tile_traffic


class MatmulTests(unittest.TestCase):
    def test_small_hand_computed(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=torch.float64)
        expected = torch.tensor([[58.0, 64.0], [139.0, 154.0]], dtype=torch.float64)
        torch.testing.assert_close(
            blocked_matmul(a, b, 1, 2, 2), expected, rtol=0, atol=0
        )

    def test_odd_edges_and_tile_choices(self):
        torch.manual_seed(7)
        a = torch.randn(5, 7, dtype=torch.float64)
        b = torch.randn(7, 3, dtype=torch.float64)
        for tile in [(1, 1, 1), (2, 2, 3), (8, 8, 16)]:
            torch.testing.assert_close(
                blocked_matmul(a, b, *tile), a @ b, rtol=1e-12, atol=1e-12
            )

    def test_noncontiguous_inputs_and_no_mutation(self):
        torch.manual_seed(8)
        a = torch.randn(7, 5, dtype=torch.float32).t()
        b = torch.randn(3, 7, dtype=torch.float32).t()
        before_a, before_b = a.clone(), b.clone()
        self.assertFalse(a.is_contiguous())
        torch.testing.assert_close(
            blocked_matmul(a, b, 2, 2, 3), a @ b, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(a, before_a, rtol=0, atol=0)
        torch.testing.assert_close(b, before_b, rtol=0, atol=0)

    def test_empty_dimensions(self):
        for m, n, k in [(0, 3, 2), (2, 0, 3), (2, 3, 0)]:
            a = torch.zeros(m, k)
            b = torch.zeros(k, n)
            torch.testing.assert_close(
                blocked_matmul(a, b, 2, 2, 2), a @ b, rtol=0, atol=0
            )

    def test_inference_only(self):
        a = torch.ones(2, 2, requires_grad=True)
        self.assertFalse(blocked_matmul(a, a, 2, 2, 2).requires_grad)

    def test_invalid_contracts(self):
        a = torch.ones(2, 3)
        for b in [
            torch.ones(2, 2),
            torch.ones(3),
            torch.ones(3, 2, dtype=torch.float64),
        ]:
            with self.assertRaises(ValueError):
                blocked_matmul(a, b, 2, 2, 2)
        with self.assertRaises(ValueError):
            blocked_matmul(a.long(), torch.ones(3, 2).long(), 2, 2, 2)
        with self.assertRaises(ValueError):
            blocked_matmul(a, torch.ones(3, 2), 2, 0, 2)


class TrafficTests(unittest.TestCase):
    def test_odd_shapes_account_for_actual_valid_elements(self):
        report = tile_traffic(5, 7, 3, 2, 4, 2, 4)
        self.assertEqual(report["read_elements"], 93)
        self.assertEqual(report["write_elements"], 35)
        self.assertEqual(report["bytes"], 512)
        self.assertEqual(report["program_count"], 6)
        self.assertEqual(report["peak_valid_input_elements"], 12)
        self.assertEqual(report["accumulator_elements"], 8)

    def test_larger_output_tile_reduces_model_reads(self):
        small = tile_traffic(64, 64, 64, 1, 1, 8, 4)
        large = tile_traffic(64, 64, 64, 8, 8, 8, 4)
        self.assertEqual(small["read_elements"], 2 * 64**3)
        self.assertEqual(large["read_elements"], small["read_elements"] // 8)
        self.assertEqual(small["write_elements"], large["write_elements"])

    def test_k_tile_changes_footprint_not_valid_total_reads(self):
        small = tile_traffic(64, 64, 64, 8, 8, 4, 4)
        large = tile_traffic(64, 64, 64, 8, 8, 16, 4)
        self.assertEqual(small["read_elements"], large["read_elements"])
        self.assertLess(
            small["peak_valid_input_elements"], large["peak_valid_input_elements"]
        )

    def test_empty_and_zero_k(self):
        self.assertEqual(tile_traffic(0, 3, 4, 2, 2, 2, 4)["bytes"], 0)
        self.assertEqual(tile_traffic(0, 3, 4, 2, 2, 2, 4)["k_steps"], 0)
        self.assertEqual(
            tile_traffic(0, 3, 4, 2, 2, 2, 4)["peak_valid_input_elements"], 0
        )
        report = tile_traffic(2, 3, 0, 2, 2, 2, 4)
        self.assertEqual(report["bytes"], 24)
        self.assertEqual(report["read_elements"], 0)

    def test_invalid_counts(self):
        for args in [(-1, 2, 2, 2, 2, 2, 4), (2, 2, 2, True, 2, 2, 4)]:
            with self.assertRaises(ValueError):
                tile_traffic(*args)


if __name__ == "__main__":
    unittest.main()
