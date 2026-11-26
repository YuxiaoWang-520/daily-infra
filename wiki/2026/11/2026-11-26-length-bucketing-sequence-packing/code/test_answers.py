"""Independent small examples and real CPU attention/gradient checks."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from answers import (
    ddp_sum_scale,
    first_fit_decreasing,
    loss_sum_and_count,
    pack_batch,
    packed_cost,
    padded_cost,
)
from toy_model import compare_models


class CostTests(unittest.TestCase):
    def test_padding_hand_example(self):
        cost = padded_cost([8, 2, 7, 3], 2)
        self.assertEqual(
            (cost.useful_tokens, cost.token_slots, cost.dense_square_area),
            (20, 30, 226),
        )
        self.assertEqual((cost.varlen_square_area, cost.valid_causal_pairs), (126, 73))
        self.assertAlmostEqual(cost.utilization, 2 / 3)

    def test_bucket_order_example(self):
        cost = padded_cost([8, 2, 7, 3], 2, order=[0, 2, 3, 1])
        self.assertEqual((cost.token_slots, cost.dense_square_area), (22, 146))

    def test_packing_example(self):
        groups = first_fit_decreasing([8, 2, 7, 3], 10)
        self.assertEqual(groups, [[0, 1], [2, 3]])
        cost = packed_cost([8, 2, 7, 3], 10, groups)
        self.assertEqual(
            (cost.useful_tokens, cost.token_slots, cost.dense_square_area),
            (20, 20, 200),
        )
        self.assertEqual(cost.utilization, 1)

    def test_heuristic_not_an_optimality_claim(self):
        lengths = [6, 5, 3, 2, 2, 2]
        groups = first_fit_decreasing(lengths, 10)
        self.assertEqual(len(groups), 3)
        optimal_witness = [[0, 3, 4], [1, 2, 5]]
        self.assertEqual(packed_cost(lengths, 10, optimal_witness).token_slots, 20)

    def test_coverage_capacity_and_stable_ties(self):
        cases = [[], [1], [5, 5, 5], [2, 2, 2, 2], [9, 1, 8, 2, 7, 3], [10, 10]]
        for lengths in cases:
            groups = first_fit_decreasing(lengths, 10)
            flat = [i for group in groups for i in group]
            self.assertEqual(sorted(flat), list(range(len(lengths))))
            self.assertTrue(all(sum(lengths[i] for i in g) <= 10 for g in groups))
        self.assertEqual(first_fit_decreasing([5, 5, 5], 10), [[0, 1], [2]])

    def test_empty_costs(self):
        for cost in (padded_cost([], 2), packed_cost([], 10, [])):
            self.assertEqual(cost.token_slots, 0)
            self.assertEqual(cost.useful_tokens, 0)
            self.assertIsNone(cost.utilization)

    def test_invalid_lengths_and_scalars(self):
        for lengths, cap in (([0], 10), ([-1], 10), ([11], 10), ([1], 0)):
            with self.assertRaises(ValueError):
                first_fit_decreasing(lengths, cap)
        for lengths, cap in (([True], 10), ([1.5], 10), ([1], True)):
            with self.assertRaises(TypeError):
                first_fit_decreasing(lengths, cap)
        with self.assertRaises(ValueError):
            padded_cost([1], 0)

    def test_invalid_group_and_order_contracts(self):
        for groups in ([[0]], [[0, 0], [1]], [[], [0, 1]], [[0, 2]], [[0, 1]]):
            with self.assertRaises(ValueError):
                packed_cost([6, 5], 10, groups)
        with self.assertRaises(ValueError):
            padded_cost([1, 2], 1, order=[0, 0])
        with self.assertRaises(TypeError):
            padded_cost([1, 2], 1, order=[False, 1])

    def test_inputs_not_mutated(self):
        lengths = [8, 2, 7, 3]
        groups = first_fit_decreasing(lengths, 10)
        packed_cost(lengths, 10, groups)
        self.assertEqual(lengths, [8, 2, 7, 3])
        self.assertEqual(groups, [[0, 1], [2, 3]])


class SemanticTests(unittest.TestCase):
    def test_metadata_example(self):
        p = pack_batch([[1, 4, 6], [2, 5]])
        self.assertEqual(p.tokens.tolist(), [1, 4, 6, 2, 5])
        self.assertEqual(p.positions.tolist(), [0, 1, 2, 0, 1])
        self.assertEqual(p.segments.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(p.labels.tolist(), [4, 6, -100, 5, -100])
        self.assertEqual(p.cu_seqlens.tolist(), [0, 3, 5])
        self.assertEqual(p.cu_seqlens.dtype, torch.int32)

    def test_mask_matches_pairwise_oracle(self):
        for lengths in ([1], [3, 2], [2, 1, 4]):
            sequences = [list(range(n)) for n in lengths]
            p = pack_batch(sequences)
            segment = [s for s, n in enumerate(lengths) for _ in range(n)]
            expected = [
                [segment[i] == segment[j] and j <= i for j in range(sum(lengths))]
                for i in range(sum(lengths))
            ]
            self.assertEqual(p.allow.tolist(), expected)
            self.assertTrue(p.allow.diagonal().all().item())

    def test_singletons_all_ignored_have_sum_zero(self):
        p = pack_batch([[1], [2]])
        logits = torch.zeros(2, 8, dtype=torch.float64, requires_grad=True)
        total, count = loss_sum_and_count(logits, p.labels)
        self.assertEqual(count, 0)
        self.assertEqual(total.item(), 0)
        total.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        self.assertTrue(torch.isnan(F.cross_entropy(logits, p.labels)).item())

    def test_invalid_sequences(self):
        for seq in ([], [[]], [[-1]], [[0], []]):
            with self.assertRaises(ValueError):
                pack_batch(seq)
        for seq in ([[False]], [[1.0]]):
            with self.assertRaises(TypeError):
                pack_batch(seq)

    def test_real_packed_and_separate_outputs_loss_gradients(self):
        result = compare_models()
        self.assertLess(result["max_logit_abs_error"], 1e-12)
        self.assertLess(result["loss_abs_error"], 1e-12)
        self.assertLess(result["max_parameter_grad_abs_error"], 1e-12)
        self.assertEqual(result["valid_target_count"], 3)

    def test_plain_causal_mask_leaks_between_sequences(self):
        result = compare_models()
        self.assertGreater(result["naive_causal_second_sequence_logit_error"], 1e-5)
        self.assertLess(result["naive_causal_first_sequence_logit_error"], 1e-12)

    def test_absolute_position_reset_matters(self):
        result = compare_models()
        self.assertGreater(result["unreset_position_second_sequence_logit_error"], 1e-5)

    def test_naive_concatenated_shift_adds_a_target(self):
        result = compare_models()
        self.assertEqual(result["naive_concat_target_count"], 4)
        self.assertEqual(result["valid_target_count"], 3)
        self.assertGreater(
            abs(result["naive_boundary_loss"] - result["token_mean_loss"]), 1e-6
        )

    def test_token_and_sequence_mean_are_different_objectives(self):
        sums, counts = [6.0, 6.0], [3, 1]
        token_mean = sum(sums) / sum(counts)
        sequence_mean = sum(s / n for s, n in zip(sums, counts, strict=True)) / 2
        self.assertEqual(token_mean, 3.0)
        self.assertEqual(sequence_mean, 4.0)

    def test_ddp_average_gradient_needs_global_token_denominator(self):
        w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        local_sums = [6 * w, 6 * w]
        scale = ddp_sum_scale([3, 1])
        correct = sum(s * scale for s in local_sums) / 2
        naive = sum(s / n for s, n in zip(local_sums, [3, 1], strict=True)) / 2
        self.assertEqual(
            torch.autograd.grad(correct, w, retain_graph=True)[0].item(), 3
        )
        self.assertEqual(torch.autograd.grad(naive, w)[0].item(), 4)

    def test_zero_local_count_allowed_when_global_count_positive(self):
        self.assertAlmostEqual(ddp_sum_scale([0, 3]), 2 / 3)

    def test_invalid_global_token_counts(self):
        for counts in ([], [0, 0], [-1, 2]):
            with self.assertRaises(ValueError):
                ddp_sum_scale(counts)
        for counts in ([True, 2], [1.5, 2]):
            with self.assertRaises(TypeError):
                ddp_sum_scale(counts)

    def test_loss_shape_value_and_target_validation(self):
        logits = torch.zeros(3, 8, dtype=torch.float64)
        for labels in (
            torch.tensor([0, 1]),
            torch.tensor([0, -1, 2]),
            torch.tensor([0, 8, 2]),
        ):
            with self.assertRaises(ValueError):
                loss_sum_and_count(logits, labels)
        with self.assertRaises(TypeError):
            loss_sum_and_count(logits, torch.zeros(3))
        with self.assertRaises(ValueError):
            loss_sum_and_count(logits.unsqueeze(0), torch.tensor([0, 1, 2]))
        bad = logits.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            loss_sum_and_count(bad, torch.tensor([0, 1, 2]))

    def test_no_double_label_shift_in_cross_entropy(self):
        p = pack_batch([[1, 4, 6], [2, 5]])
        logits = torch.arange(40, dtype=torch.float64).reshape(5, 8) / 17
        total, count = loss_sum_and_count(logits, p.labels)
        expected = -sum(
            logits.log_softmax(-1)[i, c] for i, c in ((0, 4), (1, 6), (3, 5))
        )
        torch.testing.assert_close(total, expected)
        self.assertEqual(count, 3)

    def test_helpers_and_model_do_not_consume_default_rng(self):
        before = torch.get_rng_state().clone()
        pack_batch([[1, 4], [2]])
        first_fit_decreasing([8, 2, 7, 3], 10)
        compare_models()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cpu_reference_under_meta_default_device(self):
        from run_answers import main

        with torch.device("meta"):
            batch = pack_batch([[1, 4, 6], [2, 5]])
            for tensor in (
                batch.tokens,
                batch.positions,
                batch.segments,
                batch.labels,
                batch.cu_seqlens,
                batch.allow,
            ):
                self.assertEqual(tensor.device.type, "cpu")
            self.assertLess(compare_models()["max_parameter_grad_abs_error"], 1e-12)
            with tempfile.TemporaryDirectory(prefix="packing-cpu-") as output:
                stdout = io.StringIO()
                with (
                    patch.object(
                        sys, "argv", ["run_answers.py", "--output-dir", output]
                    ),
                    contextlib.redirect_stdout(stdout),
                ):
                    main()
                result = json.loads(stdout.getvalue())
                self.assertEqual(
                    result, json.loads((Path(output) / "results.json").read_text())
                )


if __name__ == "__main__":
    unittest.main()
