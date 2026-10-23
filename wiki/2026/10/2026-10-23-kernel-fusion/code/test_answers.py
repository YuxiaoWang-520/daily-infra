"""Pipeline semantics and hypothetical traffic/launch costs, without GPU fusion."""

import math
import unittest

import torch

from answers import chunked_affine_relu, pipeline_model


class PipelineTests(unittest.TestCase):
    def test_hand_example(self):
        x = torch.tensor([[-2.0, 0.0, 2.0], [1.0, -1.0, 3.0]])
        bias = torch.tensor([1.0, 2.0, -1.0])
        expected = torch.tensor([[0.0, 4.0, 2.0], [4.0, 2.0, 4.0]])
        torch.testing.assert_close(
            chunked_affine_relu(x, bias, 2.0, 1), expected, rtol=0, atol=0
        )

    def test_noncontiguous_and_chunk_boundaries(self):
        torch.manual_seed(9)
        x = torch.randn(6, 5, dtype=torch.float64).t()
        bias = torch.randn(12, dtype=torch.float64)[::2]
        before_x, before_b = x.clone(), bias.clone()
        expected = torch.relu((x + bias) * -1.25)
        for rows in [1, 2, 9]:
            torch.testing.assert_close(
                chunked_affine_relu(x, bias, -1.25, rows), expected, rtol=0, atol=0
            )
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(bias, before_b, rtol=0, atol=0)

    def test_empty_and_zero_scale(self):
        for m, n in [(0, 3), (2, 0), (0, 0)]:
            x, bias = torch.ones(m, n), torch.ones(n)
            torch.testing.assert_close(
                chunked_affine_relu(x, bias, 1.0, 2), torch.relu(x + bias)
            )
        self.assertEqual(
            chunked_affine_relu(
                torch.ones(2, 3), torch.ones(3), 0.0, 2
            ).count_nonzero(),
            0,
        )

    def test_inference_and_no_output_alias(self):
        x = torch.ones(2, 3, requires_grad=True)
        out = chunked_affine_relu(x, torch.ones(3), 1.0, 2)
        self.assertFalse(out.requires_grad)
        self.assertNotEqual(out.data_ptr(), x.data_ptr())

    def test_invalid_contracts(self):
        x, bias = torch.ones(2, 3), torch.ones(3)
        for bad_b in [torch.ones(2), torch.ones(1, 3), bias.double()]:
            with self.assertRaises(ValueError):
                chunked_affine_relu(x, bad_b, 1.0, 2)
        for scale, rows in [(math.inf, 2), (True, 2), (1.0, 0), (1.0, True)]:
            with self.assertRaises(ValueError):
                chunked_affine_relu(x, bias, scale, rows)


class ModelTests(unittest.TestCase):
    def test_small_byte_accounting(self):
        report = pipeline_model(2, 3, 4)
        self.assertEqual(report["staged_bytes"], 156)
        self.assertEqual(report["fused_bytes"], 60)
        self.assertEqual(report["saved_bytes"], 96)
        self.assertEqual((report["staged_launches"], report["fused_launches"]), (3, 1))

    def test_slowdown_can_reverse_choice(self):
        good = pipeline_model(1024, 1024, 4)
        bad = pipeline_model(1024, 1024, 4, fused_slowdown=5.0)
        self.assertLess(good["fused_seconds"], good["staged_seconds"])
        self.assertGreater(bad["fused_seconds"], bad["staged_seconds"])
        self.assertEqual(good["fused_bytes"], bad["fused_bytes"])

    def test_empty_and_zero_launch_cost(self):
        empty = pipeline_model(0, 3, 4)
        self.assertEqual(empty["fused_bytes"], 0)
        self.assertEqual(empty["staged_launches"], 0)
        self.assertEqual(empty["fused_seconds"], 0.0)
        report = pipeline_model(2, 3, 4, bandwidth_gbs=1.0, launch_us=0.0)
        self.assertAlmostEqual(report["fused_seconds"], 60 / 1e9)

    def test_invalid_model_parameters(self):
        for options in [
            {"bandwidth_gbs": 0},
            {"launch_us": -1},
            {"fused_slowdown": 0},
            {"launch_us": math.nan},
            {"bandwidth_gbs": True},
        ]:
            with self.assertRaises(ValueError):
                pipeline_model(2, 3, 4, **options)
        with self.assertRaises(ValueError):
            pipeline_model(-1, 3, 4)


if __name__ == "__main__":
    unittest.main()
