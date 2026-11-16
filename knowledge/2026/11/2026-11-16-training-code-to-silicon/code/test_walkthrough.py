"""Numerical and observed-operator acceptance criteria for the tutorial."""

import unittest

import torch
from example_data import TARGET, B, W, X
from reference import reference_loss, reference_step
from walkthrough import forward, run_step


class ReferenceTests(unittest.TestCase):
    def test_forward_values(self) -> None:
        result = reference_step()
        self.assertEqual(
            result["preactivation"], [[5.0, -6.0, 7.0, -6.0], [1.0, 4.0, -3.0, 2.0]]
        )
        self.assertEqual(
            result["prediction"], [[5.0, 0.0, 7.0, 0.0], [1.0, 4.0, 0.0, 2.0]]
        )
        self.assertEqual(result["loss_before"], 7.25)

    def test_known_gradients(self) -> None:
        result = reference_step()
        self.assertEqual(
            result["grad_w"],
            [
                [0.75, -0.75, 1.25, -0.25],
                [-1.5, 1.5, -2.5, 0.5],
                [3.25, 0.75, 3.75, 0.25],
            ],
        )
        self.assertEqual(result["grad_b"], [1.25, 0.75, 1.25, 0.25])
        self.assertEqual(result["grad_x"], [[3.5, -1.25, 2.25], [-0.5, 1.75, -0.25]])

    def test_sgd_update(self) -> None:
        result = reference_step()
        self.assertAlmostEqual(result["updated_w"][0][0], 0.925)
        self.assertAlmostEqual(result["updated_w"][2][2], 0.625)
        self.assertAlmostEqual(result["loss_after"], 3.725390625)

    def test_finite_differences_for_all_trainable_and_input_values(self) -> None:
        x, w = [list(row) for row in X], [list(row) for row in W]
        b = list(B)
        reference = reference_step()
        epsilon = 1e-6
        parameters = [(x, reference["grad_x"]), (w, reference["grad_w"])]
        for matrix, gradient in parameters:
            for row_index, row in enumerate(matrix):
                for column_index, original in enumerate(row):
                    row[column_index] = original + epsilon
                    positive = reference_loss(x, w, b, TARGET)
                    row[column_index] = original - epsilon
                    negative = reference_loss(x, w, b, TARGET)
                    row[column_index] = original
                    numerical = (positive - negative) / (2 * epsilon)
                    self.assertAlmostEqual(
                        numerical, gradient[row_index][column_index], delta=1e-7
                    )
        for index, original in enumerate(b):
            b[index] = original + epsilon
            positive = reference_loss(x, w, b, TARGET)
            b[index] = original - epsilon
            negative = reference_loss(x, w, b, TARGET)
            b[index] = original
            self.assertAlmostEqual(
                (positive - negative) / (2 * epsilon),
                reference["grad_b"][index],
                delta=1e-7,
            )


class PyTorchTests(unittest.TestCase):
    def test_float32_matches_scalar_reference(self) -> None:
        self.check_dtype("float32", 1e-5)

    def test_float64_matches_scalar_reference(self) -> None:
        self.check_dtype("float64", 1e-12)

    def check_dtype(self, dtype_name: str, tolerance: float) -> None:
        actual, expected = run_step(dtype_name), reference_step()
        for key in (
            "preactivation",
            "prediction",
            "grad_x",
            "grad_w",
            "grad_b",
            "updated_w",
            "updated_b",
            "loss_before",
            "loss_after",
        ):
            with self.subTest(dtype=dtype_name, field=key):
                torch.testing.assert_close(
                    torch.tensor(actual[key], dtype=torch.float64),
                    torch.tensor(expected[key], dtype=torch.float64),
                    rtol=tolerance,
                    atol=tolerance,
                )

    def test_cpu_profile_observes_three_matrix_multiplications(self) -> None:
        result = run_step()
        multiplies = [
            event for event in result["cpu_operators"] if event["name"] == "aten::mm"
        ]
        self.assertEqual(sum(event["count"] for event in multiplies), 3)
        shapes = {
            tuple(tuple(dimensions) for dimensions in event["input_shapes"])
            for event in multiplies
        }
        self.assertEqual(shapes, {((2, 3), (3, 4)), ((2, 4), (4, 3)), ((3, 2), (2, 4))})
        self.assertIn("MmBackward0", result["autograd_node_types"])

    def test_fx_forward_matches_the_example(self) -> None:
        graph = torch.fx.symbolic_trace(forward)
        tensors = [
            torch.tensor(value, dtype=torch.float64) for value in (X, W, B, TARGET)
        ]
        preactivation, prediction, loss = graph(*tensors)
        torch.testing.assert_close(
            prediction,
            torch.tensor(
                [[5.0, 0.0, 7.0, 0.0], [1.0, 4.0, 0.0, 2.0]], dtype=torch.float64
            ),
        )
        self.assertEqual(loss.item(), 7.25)
        self.assertEqual(tuple(preactivation.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
