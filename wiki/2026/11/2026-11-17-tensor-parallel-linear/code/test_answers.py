"""Regression checks for sharded algebra, gradients and traffic accounting."""

import unittest

import torch
from answers import collective_budget, tp_mlp
from example_data import make_inputs


def dense_mlp(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    x, w1, b1, w2, b2 = values
    return torch.relu(x @ w1 + b1) @ w2 + b2


class AlgebraTests(unittest.TestCase):
    def test_two_ranks_match_hand_calculation(self) -> None:
        x, w1, b1, w2, b2, target = make_inputs()
        result = tp_mlp(x, w1, b1, w2, b2, 2)
        torch.testing.assert_close(
            result["output"],
            torch.tensor([[11.0, -5.0], [10.0, 6.0]], dtype=torch.float64),
        )
        expected = [
            torch.tensor([[3.0, 0.0], [1.0, 3.0]], dtype=torch.float64),
            torch.tensor([[7.0, -4.0], [8.0, 4.0]], dtype=torch.float64),
        ]
        for actual, reference in zip(result["partial_outputs"], expected, strict=True):
            torch.testing.assert_close(actual, reference)
        self.assertEqual(((result["output"] - target).square().mean()).item(), 51.25)

    def test_one_rank_matches_dense(self) -> None:
        values = make_inputs()[:5]
        actual = tp_mlp(*values, 1)["output"]
        torch.testing.assert_close(actual, dense_mlp(values))

    def test_four_ranks_keep_one_hidden_feature_each(self) -> None:
        values = make_inputs()[:5]
        result = tp_mlp(*values, 4)
        self.assertEqual(
            [tuple(part.shape) for part in result["hidden_shards"]], [(2, 1)] * 4
        )
        torch.testing.assert_close(result["output"], dense_mlp(values))

    def test_output_bias_is_added_once(self) -> None:
        values = list(make_inputs()[:5])
        values[4] = torch.tensor([7.0, -3.0], dtype=torch.float64)
        without_bias = tp_mlp(*values[:4], torch.zeros_like(values[4]), 2)["output"]
        actual = tp_mlp(*values, 2)["output"]
        torch.testing.assert_close(actual - without_bias, values[4].expand_as(actual))

    def test_all_gradients_match_dense_for_multiple_degrees(self) -> None:
        for degree in (1, 2, 4):
            with self.subTest(tp_size=degree):
                original = make_inputs()
                actual_values = tuple(
                    value.clone().requires_grad_() for value in original[:5]
                )
                expected_values = tuple(
                    value.clone().requires_grad_() for value in original[:5]
                )
                actual = tp_mlp(*actual_values, degree)["output"]
                expected = dense_mlp(expected_values)
                actual_gradients = torch.autograd.grad(
                    (actual - original[5]).square().mean(), actual_values
                )
                expected_gradients = torch.autograd.grad(
                    (expected - original[5]).square().mean(), expected_values
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    torch.testing.assert_close(
                        actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
                    )

    def test_input_gradient_has_two_partial_contributions(self) -> None:
        x, w1, b1, w2, b2, target = make_inputs()
        x.requires_grad_()
        prediction = tp_mlp(x, w1, b1, w2, b2, 2)["output"]
        gradient = torch.autograd.grad((prediction - target).square().mean(), x)[0]
        torch.testing.assert_close(
            gradient, torch.tensor([[19.0, 1.0], [1.5, 15.5]], dtype=torch.float64)
        )

    def test_all_26_derivatives_match_finite_differences(self) -> None:
        values = tuple(value.requires_grad_() for value in make_inputs()[:5])
        target = make_inputs()[5]
        output = tp_mlp(*values, 2)["output"]
        gradients = torch.autograd.grad((output - target).square().mean(), values)
        independent_values = tuple(value.detach().clone() for value in values)
        epsilon = 1e-6
        checked = 0
        for tensor, gradient in zip(independent_values, gradients, strict=True):
            flat = tensor.view(-1)
            for index in range(flat.numel()):
                original = flat[index].item()
                flat[index] = original + epsilon
                positive = (
                    (dense_mlp(independent_values) - target).square().mean().item()
                )
                flat[index] = original - epsilon
                negative = (
                    (dense_mlp(independent_values) - target).square().mean().item()
                )
                flat[index] = original
                self.assertAlmostEqual(
                    (positive - negative) / (2 * epsilon),
                    gradient.view(-1)[index].item(),
                    delta=2e-7,
                )
                checked += 1
        self.assertEqual(checked, 26)

    def test_hidden_input_size_need_not_be_divisible_by_degree(self) -> None:
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        w1 = torch.randn(3, 8, generator=generator, dtype=torch.float64)
        b1 = torch.randn(8, generator=generator, dtype=torch.float64)
        w2 = torch.randn(8, 3, generator=generator, dtype=torch.float64)
        b2 = torch.randn(3, generator=generator, dtype=torch.float64)
        values = (x, w1, b1, w2, b2)
        torch.testing.assert_close(tp_mlp(*values, 4)["output"], dense_mlp(values))

    def test_float32_matches_dense(self) -> None:
        values = make_inputs(torch.float32)[:5]
        torch.testing.assert_close(tp_mlp(*values, 2)["output"], dense_mlp(values))

    def test_rejects_invalid_parallel_degree(self) -> None:
        for degree in (0, -1, True, 1.5):
            with self.subTest(degree=degree), self.assertRaises(ValueError):
                tp_mlp(*make_inputs()[:5], degree)

    def test_rejects_uneven_intermediate_partition(self) -> None:
        with self.assertRaises(ValueError):
            tp_mlp(*make_inputs()[:5], 3)

    def test_rejects_inconsistent_shapes(self) -> None:
        x, w1, b1, w2, b2, _ = make_inputs()
        with self.assertRaises(ValueError):
            tp_mlp(x, w1, b1[:-1], w2, b2, 2)
        with self.assertRaises(ValueError):
            tp_mlp(x, w1, b1, w2[:-1], b2, 2)


class BudgetTests(unittest.TestCase):
    def test_regather_for_weight_gradient_adds_one_allgather(self) -> None:
        result = collective_budget(
            2048,
            8192,
            8,
            2,
            sequence_parallel=True,
            regather_input_for_wgrad=True,
        )
        self.assertEqual(result["all_gather_calls"], 3)
        self.assertEqual(result["reduce_scatter_calls"], 2)
        self.assertEqual(result["sent_bytes_per_rank"], 140 * 2**20)
        self.assertEqual(result["ring_rounds"], 35)

    def test_regather_option_requires_sequence_parallel_mode(self) -> None:
        with self.assertRaises(ValueError):
            collective_budget(8, 8, 2, 4, regather_input_for_wgrad=True)

    def test_two_allreduces_send_112_mib_per_rank(self) -> None:
        result = collective_budget(2048, 8192, 8, 2)
        self.assertEqual(result["payload_bytes"], 32 * 2**20)
        self.assertEqual(result["sent_bytes_per_rank"], 112 * 2**20)
        self.assertEqual(
            result["received_bytes_per_rank"], result["sent_bytes_per_rank"]
        )
        self.assertEqual(result["all_reduce_calls"], 2)
        self.assertEqual(result["ring_rounds"], 28)

    def test_sequence_parallel_has_equal_ideal_ring_bytes(self) -> None:
        base = collective_budget(2048, 8192, 8, 2)
        sequence = collective_budget(2048, 8192, 8, 2, sequence_parallel=True)
        self.assertEqual(sequence["sent_bytes_per_rank"], base["sent_bytes_per_rank"])
        self.assertEqual(sequence["ring_rounds"], base["ring_rounds"])
        self.assertEqual(sequence["all_reduce_calls"], 0)
        self.assertEqual(sequence["all_gather_calls"], 2)
        self.assertEqual(sequence["reduce_scatter_calls"], 2)

    def test_one_rank_has_no_communication(self) -> None:
        for sequence in (False, True):
            result = collective_budget(3, 7, 1, 4, sequence_parallel=sequence)
            self.assertEqual(result["sent_bytes_per_rank"], 0)
            self.assertEqual(result["ring_rounds"], 0)
            self.assertEqual(
                sum(
                    result[key]
                    for key in (
                        "all_reduce_calls",
                        "all_gather_calls",
                        "reduce_scatter_calls",
                    )
                ),
                0,
            )

    def test_rejects_invalid_integer_inputs(self) -> None:
        for arguments in ((0, 8, 2, 4), (2, -1, 2, 4), (2, 8, True, 4), (2, 8, 2, 1.5)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                collective_budget(*arguments)

    def test_rejects_unequal_ring_chunks_in_the_teaching_model(self) -> None:
        with self.assertRaises(ValueError):
            collective_budget(1, 3, 2, 1)

    def test_sequence_partition_requires_even_token_count(self) -> None:
        collective_budget(3, 8, 2, 4)
        with self.assertRaises(ValueError):
            collective_budget(3, 8, 2, 4, sequence_parallel=True)


if __name__ == "__main__":
    unittest.main()
