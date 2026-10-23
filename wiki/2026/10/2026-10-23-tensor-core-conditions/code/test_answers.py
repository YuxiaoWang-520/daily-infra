"""Padding costs and storage quantization, not Tensor Core execution tests."""

import unittest

import torch

from answers import padding_plan, quantized_padded_matmul


class PaddingTests(unittest.TestCase):
    def test_specific_chosen_instruction_tile(self):
        plan = padding_plan(17, 9, 17, (16, 8, 16))
        self.assertEqual(plan["padded_shape"], (32, 16, 32))
        self.assertEqual(plan["original_flops"], 5202)
        self.assertEqual(plan["padded_flops"], 32768)
        self.assertAlmostEqual(plan["work_ratio"], 32768 / 5202)

    def test_already_aligned_and_empty(self):
        self.assertEqual(padding_plan(32, 16, 32, (16, 8, 16))["work_ratio"], 1.0)
        self.assertIsNone(padding_plan(0, 9, 17, (16, 8, 16))["work_ratio"])

    def test_invalid_dims_and_tile(self):
        for args in [
            (-1, 2, 3, (16, 8, 16)),
            (1, 2, 3, (16, 0, 16)),
            (1, 2, 3, (16, 8)),
            (True, 2, 3, (16, 8, 16)),
        ]:
            with self.assertRaises(ValueError):
                padding_plan(*args)


class QuantizationTests(unittest.TestCase):
    def test_input_rounding_is_not_recovered_by_float32_compute(self):
        a = torch.tensor([[1.001]], dtype=torch.float64)
        b = torch.ones(1, 1, dtype=torch.float64)
        half = quantized_padded_matmul(a, b, torch.float16)
        bf16 = quantized_padded_matmul(a, b, torch.bfloat16)
        self.assertEqual(half.item(), 1.0009765625)
        self.assertEqual(bf16.item(), 1.0)
        self.assertEqual(half.dtype, torch.float32)

    def test_odd_rectangular_inputs_match_quantized_reference(self):
        torch.manual_seed(8)
        a = torch.randn(3, 5, dtype=torch.float64)
        b = torch.randn(5, 7, dtype=torch.float64)
        for dtype in (torch.float16, torch.bfloat16):
            expected = a.to(dtype).float() @ b.to(dtype).float()
            actual = quantized_padded_matmul(a, b, dtype)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            self.assertEqual(actual.shape, (3, 7))
            self.assertTrue(actual.is_contiguous())

    def test_noncontiguous_input_and_no_mutation(self):
        a = torch.arange(15, dtype=torch.float32).reshape(5, 3).t()
        b = torch.ones(5, 2)
        before = a.clone()
        out = quantized_padded_matmul(a, b, torch.float16)
        torch.testing.assert_close(out, a @ b, rtol=0, atol=0)
        torch.testing.assert_close(a, before, rtol=0, atol=0)

    def test_empty_dimensions(self):
        for m, n, k in [(0, 2, 3), (2, 0, 3), (2, 3, 0)]:
            a, b = torch.ones(m, k), torch.ones(k, n)
            torch.testing.assert_close(
                quantized_padded_matmul(a, b, torch.float16), a @ b
            )

    def test_reject_conversion_overflow_and_nonfinite(self):
        for value in [70000.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                quantized_padded_matmul(
                    torch.tensor([[value]]), torch.ones(1, 1), torch.float16
                )

    def test_invalid_contract_and_inference(self):
        with self.assertRaises(ValueError):
            quantized_padded_matmul(torch.ones(2, 3), torch.ones(2, 3), torch.float16)
        with self.assertRaises(ValueError):
            quantized_padded_matmul(torch.ones(2, 2), torch.ones(2, 2), torch.float32)
        a = torch.ones(2, 2, requires_grad=True)
        self.assertFalse(quantized_padded_matmul(a, a, torch.float16).requires_grad)


if __name__ == "__main__":
    unittest.main()
