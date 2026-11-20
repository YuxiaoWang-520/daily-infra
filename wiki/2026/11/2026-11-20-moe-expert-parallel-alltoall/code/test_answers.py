"""Independent dense reference and hand-computed routing/traffic checks."""

import math
import unittest

import torch
from answers import (
    capacity_for,
    combine_tokens,
    dispatch_plan,
    evaluate_experts,
    load_balance_loss,
    moe_forward,
    route_tokens,
)
from example_data import example_logits, synthetic_inputs


def dense_reference(x, logits, w1, w2, k=2, *, pre_softmax=False):
    """Compute every expert, then apply a dense gate mask; no packed implementation."""
    selected = torch.topk(logits, k, dim=-1).indices
    if pre_softmax:
        weights = logits.softmax(dim=-1).gather(1, selected)
    else:
        weights = logits.gather(1, selected).softmax(dim=-1)
    gates = torch.zeros_like(logits).scatter(1, selected, weights)
    all_outputs = torch.stack([torch.tanh(x @ a) @ b for a, b in zip(w1, w2)], dim=1)
    return (gates[:, :, None] * all_outputs).sum(dim=1)


def leaves(values):
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


class RoutingTests(unittest.TestCase):
    def test_selected_experts_and_weights(self):
        routes = route_tokens(example_logits(), 2)
        self.assertEqual(routes.token_indices.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(routes.expert_indices.tolist(), [0, 1, 2, 1, 3, 2, 0, 2])
        torch.testing.assert_close(
            routes.weights,
            torch.tensor(
                [5 / 8, 3 / 8, 3 / 4, 1 / 4, 5 / 8, 3 / 8, 8 / 15, 7 / 15],
                dtype=torch.float64,
            ),
        )
        self.assertEqual(routes.expert_counts.tolist(), [2, 2, 3, 1])

    def test_softmax_order_changes_weight_sum(self):
        logits = example_logits()
        after = route_tokens(logits, 2)
        before = route_tokens(logits, 2, pre_softmax=True)
        torch.testing.assert_close(
            after.weights.view(4, 2).sum(1), torch.ones(4, dtype=logits.dtype)
        )
        torch.testing.assert_close(
            before.weights.view(4, 2).sum(1),
            torch.tensor([0.8, 0.8, 0.8, 0.75], dtype=logits.dtype),
        )

    def test_ties_use_lower_expert_id(self):
        routes = route_tokens(torch.zeros(2, 4, dtype=torch.float64), 2)
        self.assertEqual(routes.expert_indices.tolist(), [0, 1, 0, 1])

    def test_capacity_keeps_highest_probability_without_renormalizing(self):
        routes = route_tokens(example_logits(), 2, capacity=2)
        self.assertEqual(routes.expert_counts.tolist(), [2, 2, 2, 1])
        self.assertNotIn(
            (2, 2),
            list(zip(routes.token_indices.tolist(), routes.expert_indices.tolist())),
        )
        self.assertAlmostEqual(
            float(routes.weights[routes.token_indices == 2].sum()), 5 / 8
        )

    def test_zero_capacity_drops_all_assignments(self):
        routes = route_tokens(example_logits(), 2, capacity=0)
        self.assertEqual(routes.weights.numel(), 0)
        self.assertEqual(routes.expert_counts.tolist(), [0, 0, 0, 0])

    def test_capacity_formula_counts_assignments(self):
        self.assertEqual(capacity_for(4, 4, 2, 1.0), 2)
        self.assertEqual(capacity_for(4, 4, 2, 1.25), 3)

    def test_invalid_capacity_factor(self):
        for factor in (0, -1, math.inf, math.nan, True):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                capacity_for(4, 4, 2, factor)

    def test_invalid_route_inputs(self):
        for logits, k, cap in [
            (example_logits(), 0, None),
            (example_logits(), 5, None),
            (example_logits(), True, None),
            (example_logits(), 2, -1),
            (torch.ones(4), 1, None),
            (example_logits().to(torch.int64), 1, None),
        ]:
            with self.subTest(k=k, cap=cap), self.assertRaises((ValueError, TypeError)):
                route_tokens(logits, k, capacity=cap)
        logits = example_logits()
        logits[0, 0] = math.nan
        with self.assertRaises(ValueError):
            route_tokens(logits, 2)

    def test_owner_split_matrix(self):
        plans = [
            dispatch_plan(route_tokens(part, 2), 2)
            for part in example_logits().chunk(2)
        ]
        self.assertEqual(plans[0].send_splits, (3, 1))
        self.assertEqual(plans[1].send_splits, (1, 3))
        self.assertEqual(sum(sum(p.send_splits) for p in plans), 8)

    def test_empty_destination_split(self):
        plan = dispatch_plan(route_tokens(example_logits(skew=True)[:2], 2), 2)
        self.assertEqual(plan.send_splits, (4, 0))
        with self.assertRaises(ValueError):
            dispatch_plan(route_tokens(example_logits(), 2), 3)

    def test_aux_loss_hand_value_and_gradient(self):
        logits = example_logits().requires_grad_(True)
        loss = load_balance_loss(logits, 2, coefficient=0.01)
        self.assertAlmostEqual(float(loss.detach()), 0.01075)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().max()), 0)

    def test_aux_uniform_probability_is_not_balance_certificate(self):
        logits = torch.zeros(4, 4, dtype=torch.float64)
        routes = route_tokens(logits, 2)
        self.assertEqual(routes.expert_counts.tolist(), [4, 4, 0, 0])
        self.assertAlmostEqual(float(load_balance_loss(logits, 2)), 1)

    def test_post_top1_gate_has_zero_task_gradient(self):
        for pre in (False, True):
            logits = example_logits().requires_grad_(True)
            route_tokens(logits, 1, pre_softmax=pre).weights.sum().backward()
            norm = float(logits.grad.abs().sum())
            self.assertGreater(norm, 0.01) if pre else self.assertEqual(norm, 0)


class PackedMoETests(unittest.TestCase):
    def test_scalar_weighted_combine(self):
        routes = route_tokens(example_logits(), 2)
        values = (routes.expert_indices + 1).to(torch.float64) * (
            routes.token_indices + 1
        )
        out = combine_tokens(values[:, None], routes)
        torch.testing.assert_close(
            out[:, 0],
            torch.tensor([11 / 8, 11 / 2, 87 / 8, 116 / 15], dtype=torch.float64),
        )

    def test_capacity_changes_scalar_output(self):
        routes = route_tokens(example_logits(), 2, capacity=2)
        values = (routes.expert_indices + 1).to(torch.float64) * (
            routes.token_indices + 1
        )
        out = combine_tokens(values[:, None], routes)
        self.assertAlmostEqual(float(out[2, 0]), 7.5)

    def test_packed_output_and_all_gradients_match_dense(self):
        for pre in (False, True):
            values = synthetic_inputs()
            x, logits, w1, w2 = leaves(values)
            actual, _ = moe_forward(x, logits, w1, w2, 2, pre_softmax=pre)
            actual.square().mean().backward()
            refs = leaves(values)
            expected = dense_reference(*refs, pre_softmax=pre)
            expected.square().mean().backward()
            for a, b in zip(
                (actual, x.grad, logits.grad, w1.grad, w2.grad),
                (expected, *(r.grad for r in refs)),
            ):
                torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_zero_capacity_keeps_zero_gradients(self):
        inputs = leaves(synthetic_inputs())
        out, _ = moe_forward(*inputs, 2, capacity=0)
        self.assertTrue(bool((out == 0).all()))
        out.sum().backward()
        for tensor in inputs:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool((tensor.grad == 0).all()))

    def test_unused_experts_have_zero_gradients(self):
        inputs = leaves(synthetic_inputs(skew=True))
        out, _ = moe_forward(*inputs, 2)
        out.square().mean().backward()
        for weight in inputs[2:]:
            self.assertTrue(bool((weight.grad[2:] == 0).all()))

    def test_dispatch_order_can_be_inverted(self):
        routes = route_tokens(example_logits(), 2)
        plan = dispatch_plan(routes, 2)
        values = torch.arange(16, dtype=torch.float64).view(8, 2)
        restored = values.index_select(0, plan.order).index_select(
            0, plan.order.argsort()
        )
        torch.testing.assert_close(
            combine_tokens(restored, routes), combine_tokens(values, routes)
        )

    def test_empty_expert_input_remains_differentiable(self):
        _, _, w1, w2 = synthetic_inputs()
        x = torch.empty(0, 2, dtype=torch.float64, requires_grad=True)
        w1 = w1[2:].clone().requires_grad_(True)
        w2 = w2[2:].clone().requires_grad_(True)
        out = evaluate_experts(
            x, torch.empty(0, dtype=torch.int64), w1, w2, expert_offset=2
        )
        self.assertEqual(tuple(out.shape), (0, 2))
        out.sum().backward()
        for tensor in (x, w1, w2):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool((tensor.grad == 0).all()))

    def test_bad_expert_shapes_rejected(self):
        x, logits, w1, w2 = synthetic_inputs()
        for a, b in [(w1[:, :1], w2), (w1, w2[:, :2]), (w1, w2[:3])]:
            with self.assertRaises(ValueError):
                moe_forward(x, logits, a, b, 2)
        with self.assertRaises(ValueError):
            evaluate_experts(x, torch.tensor([0, 1, 2, 7]), w1, w2)

    def test_combination_shape_mismatch_rejected(self):
        routes = route_tokens(example_logits(), 2)
        with self.assertRaises(ValueError):
            combine_tokens(torch.ones(7, 2, dtype=torch.float64), routes)

    def test_finite_difference_gradcheck(self):
        inputs = leaves(synthetic_inputs())
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda *v: moe_forward(*v, 2)[0], inputs, eps=1e-6, atol=1e-5
            )
        )


if __name__ == "__main__":
    unittest.main()
