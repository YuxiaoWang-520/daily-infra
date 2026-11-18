"""Behavioral acceptance for causality, lifetime and gradient normalization."""

import copy
import unittest
from itertools import pairwise

import torch
from answers import microbatch_backward, simulate_pipeline
from example_data import make_batch, make_model


class ScheduleTests(unittest.TestCase):
    def test_gpipe_three_stages_four_microbatches(self):
        result = simulate_pipeline(3, 4, "gpipe")
        self.assertEqual(result["makespan_ticks"], 12)
        self.assertEqual(result["peak_saved_microbatches"], [4, 4, 4])
        self.assertAlmostEqual(result["compute_utilization"], 2 / 3)

    def test_1f1b_preserves_time_and_reduces_live_activations(self):
        result = simulate_pipeline(3, 4, "1f1b")
        self.assertEqual(result["makespan_ticks"], 12)
        self.assertEqual(result["peak_saved_microbatches"], [3, 2, 1])

    def test_first_stage_order(self):
        result = simulate_pipeline(3, 4, "1f1b")
        order = [
            (e["phase"], e["microbatch"]) for e in result["events"] if e["stage"] == 0
        ]
        self.assertEqual(
            order,
            [
                ("F", 0),
                ("F", 1),
                ("F", 2),
                ("B", 0),
                ("F", 3),
                ("B", 1),
                ("B", 2),
                ("B", 3),
            ],
        )

    def test_gpipe_backward_is_fifo(self):
        result = simulate_pipeline(3, 4, "gpipe")
        for rank in range(3):
            backward = [
                e["microbatch"]
                for e in result["events"]
                if e["stage"] == rank and e["phase"] == "B"
            ]
            self.assertEqual(backward, [0, 1, 2, 3])

    def test_balanced_formula_for_many_sizes(self):
        for p in range(1, 6):
            for m in range(1, 8):
                for name in ["gpipe", "1f1b"]:
                    with self.subTest(p=p, m=m, schedule=name):
                        result = simulate_pipeline(
                            p, m, name, forward_ticks=[1] * p, backward_ticks=[2] * p
                        )
                        self.assertEqual(result["makespan_ticks"], (m + p - 1) * 3)
                        self.assertAlmostEqual(
                            result["compute_utilization"], m / (m + p - 1)
                        )

    def test_dependencies_and_stage_exclusivity(self):
        for p in range(1, 5):
            for m in range(1, 6):
                for name in ["gpipe", "1f1b"]:
                    result = simulate_pipeline(
                        p,
                        m,
                        name,
                        forward_ticks=list(range(1, p + 1)),
                        backward_ticks=list(range(p, 0, -1)),
                    )
                    events = result["events"]
                    lookup = {
                        (e["stage"], e["phase"], e["microbatch"]): e for e in events
                    }
                    self.assertEqual(len(lookup), 2 * p * m)
                    for r in range(p):
                        local = [e for e in events if e["stage"] == r]
                        for a, b in pairwise(local):
                            self.assertLessEqual(a["end"], b["start"])
                        for i in range(m):
                            f, b = lookup[r, "F", i], lookup[r, "B", i]
                            self.assertLessEqual(f["end"], b["start"])
                            if r:
                                self.assertLessEqual(
                                    lookup[r - 1, "F", i]["end"], f["start"]
                                )
                            if r < p - 1:
                                self.assertLessEqual(
                                    lookup[r + 1, "B", i]["end"], b["start"]
                                )

    def test_fewer_microbatches_than_stages_in_teaching_model(self):
        result = simulate_pipeline(4, 2, "1f1b")
        self.assertEqual(result["peak_saved_microbatches"], [2, 2, 2, 1])
        self.assertEqual(result["makespan_ticks"], 10)

    def test_one_stage_has_no_bubble(self):
        for name, peak in [("gpipe", 4), ("1f1b", 1)]:
            result = simulate_pipeline(1, 4, name)
            self.assertEqual(result["makespan_ticks"], 8)
            self.assertEqual(result["peak_saved_microbatches"], [peak])
            self.assertEqual(result["compute_utilization"], 1)

    def test_slow_middle_stage_increases_time(self):
        balanced = simulate_pipeline(3, 4, "1f1b")
        slow = simulate_pipeline(
            3, 4, "1f1b", forward_ticks=[1, 3, 1], backward_ticks=[1, 3, 1]
        )
        self.assertGreater(slow["makespan_ticks"], balanced["makespan_ticks"])
        self.assertLess(slow["compute_utilization"], balanced["compute_utilization"])

    def test_rejects_invalid_counts(self):
        for value in [0, -1, True, 2.5]:
            for args in [(value, 4), (3, value)]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    simulate_pipeline(*args, "gpipe")

    def test_rejects_invalid_costs(self):
        for value in [[], [1], [1, 0, 1], [1, True, 1], [1, 2.5, 1]]:
            for key in ["forward_ticks", "backward_ticks"]:
                with self.subTest(costs=value, key=key), self.assertRaises(ValueError):
                    simulate_pipeline(3, 4, "gpipe", **{key: value})

    def test_rejects_unknown_schedule(self):
        with self.assertRaises(ValueError):
            simulate_pipeline(3, 4, "unknown")


class GradientTests(unittest.TestCase):
    def compare_reference(self, sizes):
        model = make_model()
        dense = copy.deepcopy(model)
        x, target = make_batch()
        output = dense(x)
        loss = (output - target).square().mean()
        loss.backward()
        result = microbatch_backward(model, x, target, sizes)
        torch.testing.assert_close(result["output"], output, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(result["loss"], loss.item(), places=12)
        for a, b in zip(model.parameters(), dense.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
        return model, dense

    def test_equal_microbatches_match_full_batch(self):
        self.compare_reference([2, 2, 2, 2])

    def test_unequal_microbatches_match_full_batch(self):
        self.compare_reference([1, 3, 4])

    def test_single_full_chunk(self):
        self.compare_reference([8])

    def test_single_row_chunks(self):
        self.compare_reference([1] * 8)

    def test_one_optimizer_update_matches_dense(self):
        model, dense = self.compare_reference([1, 3, 4])
        for net in [model, dense]:
            torch.optim.SGD(net.parameters(), lr=0.05).step()
        for a, b in zip(model.parameters(), dense.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_existing_gradients_are_cleared(self):
        model = make_model()
        x, target = make_batch()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 100
        microbatch_backward(model, x, target, [1, 3, 4])
        dense = make_model()
        (dense(x) - target).square().mean().backward()
        for a, b in zip(model.parameters(), dense.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)

    def test_naive_average_of_unequal_means_is_wrong(self):
        theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0.0, 2.0, 2.0, 2.0], dtype=torch.float64)
        correct = (theta - target).square().mean()
        wrong = (
            (theta - target[:1]).square().mean() + (theta - target[1:]).square().mean()
        ) / 2
        self.assertEqual(correct.item(), 3)
        self.assertEqual(wrong.item(), 2)
        self.assertEqual(torch.autograd.grad(correct, theta)[0].item(), -3)
        self.assertEqual(torch.autograd.grad(wrong, theta)[0].item(), -2)

    def test_rejects_invalid_chunk_partition(self):
        x, target = make_batch()
        for sizes in [[], [7], [9], [0, 8], [True, 7], [1.5, 6.5]]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                microbatch_backward(make_model(), x, target, sizes)

    def test_rejects_invalid_input_target_shapes(self):
        x, target = make_batch()
        for a, b in [
            (x[:0], target[:0]),
            (x, target[:7]),
            (x.flatten(), target),
            (x, target.flatten()),
            (x, target[:, :1]),
        ]:
            with (
                self.subTest(xshape=a.shape, yshape=b.shape),
                self.assertRaises(ValueError),
            ):
                microbatch_backward(make_model(), a, b, [len(a)])


if __name__ == "__main__":
    unittest.main()
