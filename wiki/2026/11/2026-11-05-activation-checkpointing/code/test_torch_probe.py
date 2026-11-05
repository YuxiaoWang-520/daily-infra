"""CPU eager equivalence, saved-storage accounting and a deliberate RNG failure."""

import importlib
import os
import unittest

import torch

impl = importlib.import_module(os.environ.get("CHECKPOINT_IMPL", "torch_probe"))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = impl.run_trial(False)
        cls.preserved = impl.run_trial(True)
        cls.unpreserved = impl.run_trial(True, preserve_rng_state=False)

    def test_same_original_forward_even_when_rng_is_not_preserved(self):
        for result in (self.preserved, self.unpreserved):
            torch.testing.assert_close(result.output, self.plain.output, rtol=0, atol=0)
            self.assertEqual(result.loss, self.plain.loss)

    def test_preserved_rng_matches_input_and_every_parameter_gradient(self):
        torch.testing.assert_close(
            self.preserved.input_gradient,
            self.plain.input_gradient,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(len(self.preserved.parameter_gradients), 8)
        for actual, expected in zip(
            self.preserved.parameter_gradients,
            self.plain.parameter_gradients,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_preserved_rng_matches_every_sgd_updated_parameter(self):
        for actual, expected in zip(
            self.preserved.updated_parameters,
            self.plain.updated_parameters,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_preserved_rng_keeps_the_next_cpu_random_sample(self):
        self.assertTrue(torch.equal(self.preserved.next_random, self.plain.next_random))

    def test_unpreserved_rng_exposes_gradient_mismatch(self):
        self.assertTrue(
            any(
                not torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)
                for actual, expected in zip(
                    self.unpreserved.parameter_gradients,
                    self.plain.parameter_gradients,
                    strict=True,
                )
            )
        )
        self.assertFalse(
            torch.equal(self.unpreserved.next_random, self.plain.next_random)
        )

    def test_block_entries_show_replay_in_reverse_block_order(self):
        self.assertEqual(self.plain.forward_entry_calls, (0, 1, 2, 3))
        self.assertEqual(self.preserved.forward_entry_calls, (0, 1, 2, 3, 3, 2, 1, 0))

    def test_checkpoint_saves_fewer_outer_forward_activation_bytes(self):
        self.assertGreater(self.plain.saved_activation_bytes, 0)
        self.assertGreater(self.preserved.saved_activation_bytes, 0)
        self.assertLess(
            self.preserved.saved_activation_bytes, self.plain.saved_activation_bytes
        )

    def test_all_parameter_gradients_are_finite_and_nonzero(self):
        for gradient in self.preserved.parameter_gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)


class FixtureBoundaryTests(unittest.TestCase):
    def test_no_dropout_is_equivalent_without_rng_preservation(self):
        plain = impl.run_trial(False, dropout_p=0)
        replayed = impl.run_trial(True, preserve_rng_state=False, dropout_p=0)
        for actual, expected in zip(
            replayed.parameter_gradients, plain.parameter_gradients, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_trial_restores_callers_cpu_rng(self):
        before = torch.get_rng_state().clone()
        impl.run_trial(True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_single_block_has_no_extra_saved_boundary_activation(self):
        plain = impl.run_trial(False, blocks=1)
        replayed = impl.run_trial(True, blocks=1)
        self.assertGreater(plain.saved_activation_bytes, 0)
        self.assertEqual(replayed.saved_activation_bytes, 0)

    def test_invalid_fixture_configuration(self):
        for kwargs in (
            {"blocks": 0},
            {"width": True},
            {"batch_size": -1},
            {"dropout_p": 1},
            {"dropout_p": float("nan")},
            {"dropout_p": True},
            {"checkpointed": 1},
            {"preserve_rng_state": "yes"},
            {"seed": -1},
        ):
            options = {"checkpointed": True, **kwargs}
            with self.subTest(options=options), self.assertRaises(ValueError):
                impl.run_trial(**options)


if __name__ == "__main__":
    unittest.main()
