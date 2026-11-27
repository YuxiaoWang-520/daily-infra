"""Analytic examples plus real CPU GradScaler and autocast checks."""

import math
import unittest

import torch

from answers import quantized_gradient, unscale_accumulated, unscale_and_clip
from runtime_demo import (
    accumulation_probe,
    autocast_probe,
    clipping_probe,
    protocol_probes,
    training_trace,
)


class ScalingMathTests(unittest.TestCase):
    def test_fp16_underflow_and_early_scale(self):
        r = quantized_gradient(1e-8, 1024)
        self.assertEqual(r.direct_storage, 0)
        self.assertGreater(r.recovered, 0)
        self.assertLess(abs(r.recovered - 1e-8), 2e-11)

    def test_large_scale_can_overflow(self):
        r = quantized_gradient(64, 1024)
        self.assertEqual(r.direct_storage, 64)
        self.assertTrue(math.isinf(r.scaled_storage))

    def test_bfloat16_retains_this_small_value(self):
        r = quantized_gradient(1e-8, 1, dtype=torch.bfloat16)
        self.assertGreater(r.direct_storage, 0)

    def test_invalid_quantization_inputs(self):
        for grad, scale in ((math.nan, 8), (1, math.inf), (1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                quantized_gradient(grad, scale)
        for grad, scale in ((True, 8), (1, False)):
            with self.assertRaises(TypeError):
                quantized_gradient(grad, scale)
        with self.assertRaises(ValueError):
            quantized_gradient(1, 8, dtype=torch.int32)

    def test_unscale_before_clipping(self):
        result = unscale_and_clip([24, 32], 8, 1)
        for got, expected in zip(result, [0.6, 0.8], strict=True):
            self.assertAlmostEqual(got, expected)

    def test_wrong_clip_order_shrinks_by_scale(self):
        clipped_scaled = unscale_and_clip([24, 32], 1, 1)
        wrong = [g / 8 for g in clipped_scaled]
        for got, expected in zip(wrong, [0.075, 0.1], strict=True):
            self.assertAlmostEqual(got, expected)
        self.assertEqual(unscale_and_clip([0, 0], 8, 1), [0, 0])

    def test_clipping_validation(self):
        for values, scale, limit in (([], 8, 1), ([math.inf], 8, 1), ([1], 8, 0)):
            with self.assertRaises(ValueError):
                unscale_and_clip(values, scale, limit)
        with self.assertRaises(TypeError):
            unscale_and_clip([True], 8, 1)

    def test_accumulation_scale_must_stay_constant(self):
        self.assertEqual(unscale_accumulated([1, 1], [8, 8], 8), 2)
        self.assertEqual(unscale_accumulated([1, 1], [8, 16], 16), 1.5)
        with self.assertRaises(ValueError):
            unscale_accumulated([1, 1], [8], 8)
        with self.assertRaises(ValueError):
            unscale_accumulated([], [], 8)

    def test_numeric_overflow_uses_value_error(self):
        cases = (
            lambda: unscale_accumulated([1e308, 1e308], [1, 1], 1),
            lambda: quantized_gradient(10**1000, 8),
        )
        for index, case in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                case()


class RuntimeTests(unittest.TestCase):
    def test_real_trace_scale_and_training_clocks(self):
        result = training_trace([False, False, True, False, False])
        records = result["records"]
        self.assertTrue(result["scaler_enabled"])
        self.assertEqual([r["scale_after"] for r in records], [8, 16, 8, 8, 16])
        self.assertEqual([r["applied_updates"] for r in records], [1, 2, 2, 3, 4])
        self.assertEqual([r["growth_tracker"] for r in records], [1, 0, 0, 1, 0])
        self.assertEqual(result["data_batches"], 5)
        self.assertEqual(result["scheduler_steps"], 4)
        self.assertAlmostEqual(records[-1]["parameter_after"], 0.2268, places=6)

    def test_nonfinite_skip_preserves_parameter_and_momentum(self):
        r = training_trace([False, False, True])["records"]
        self.assertFalse(r[2]["optimizer_step_called"])
        self.assertEqual(r[2]["parameter_after"], r[1]["parameter_after"])
        self.assertEqual(r[2]["momentum_after"], r[1]["momentum_after"])

    def test_step_none_return_cannot_detect_skip(self):
        r = training_trace([False, True])["records"]
        self.assertTrue(all(x["step_return_is_none"] for x in r))
        self.assertEqual([x["optimizer_step_called"] for x in r], [True, False])

    def test_public_state_dict_roundtrip_at_update_boundaries(self):
        faults = [False, False, True, False, False]
        baseline = training_trace(faults)
        for cursor in (0, 1, 2, 3, 5):
            restored = training_trace(faults, restore_after=cursor)
            self.assertEqual(restored["records"], baseline["records"])
            self.assertEqual(restored["final_scaler"], baseline["final_scaler"])

    def test_real_clipping_order(self):
        r = clipping_probe()
        torch.testing.assert_close(
            torch.tensor(r["correct"]), torch.tensor([0.6, 0.8]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            torch.tensor(r["wrong"]), torch.tensor([0.075, 0.1]), atol=1e-6, rtol=0
        )

    def test_real_accumulation_order(self):
        r = accumulation_probe()
        self.assertEqual(r["fixed_scale_gradient"], 2)
        self.assertEqual(r["changed_scale_gradient"], 1.5)

    def test_optimizer_protocol_errors_are_real(self):
        r = protocol_probes()
        self.assertEqual(r["double_unscale_error"], "RuntimeError")
        self.assertEqual(r["double_step_error"], "RuntimeError")
        self.assertEqual(r["half_parameter_unscale_error"], "ValueError")

    def test_multi_optimizer_decisions_are_independent(self):
        r = protocol_probes()["multiple_optimizers"]
        self.assertEqual(r["step_calls"], [1, 0])
        self.assertEqual(r["scale_after"], 4)
        self.assertAlmostEqual(r["parameters_after"][0], 0.8, places=6)
        self.assertEqual(r["parameters_after"][1], 1)

    def test_post_unscale_nonfinite_is_not_rechecked(self):
        r = protocol_probes()["mutation_after_unscale"]
        self.assertEqual(r["step_calls"], 1)
        self.assertTrue(math.isinf(r["parameter_after"]))
        self.assertLess(r["parameter_after"], 0)

    def test_autocast_has_per_operation_dtypes(self):
        r = autocast_probe()
        self.assertEqual(r["output_dtype"], "torch.bfloat16")
        self.assertEqual(r["loss_dtype"], "torch.float32")
        self.assertEqual(r["parameter_dtype"], "torch.float32")
        self.assertEqual(r["gradient_dtype"], "torch.float32")

    def test_cpu_factories_under_meta_default(self):
        with torch.device("meta"):
            self.assertGreater(quantized_gradient(1e-8, 1024).recovered, 0)
            self.assertEqual(training_trace([False, True])["applied_updates"], 1)
            self.assertEqual(autocast_probe()["parameter_device"], "cpu")

    def test_runtime_helpers_do_not_consume_default_rng(self):
        before = torch.get_rng_state().clone()
        training_trace([False, True])
        autocast_probe()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_trace_input_validation(self):
        for faults in ([1], [None]):
            with self.assertRaises(TypeError):
                training_trace(faults)
        for cursor in (-1, 2):
            with self.assertRaises(ValueError):
                training_trace([False], restore_after=cursor)
        with self.assertRaises(TypeError):
            training_trace([False], restore_after=True)
        self.assertEqual(training_trace([])["applied_updates"], 0)

    def test_scale_can_fall_below_one(self):
        r = training_trace([True], init_scale=0.5)
        self.assertEqual(r["final_scaler"]["scale"], 0.25)
        self.assertEqual(r["applied_updates"], 0)


if __name__ == "__main__":
    unittest.main()
