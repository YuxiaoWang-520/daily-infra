"""Scalar Python reference for the tutorial's fixed synthetic example."""

from collections.abc import Sequence

from example_data import LEARNING_RATE, TARGET, B, W, X

Matrix = Sequence[Sequence[float]]


def _forward_values(
    x: Matrix, w: Matrix, b: Sequence[float]
) -> tuple[list[list[float]], list[list[float]]]:
    preactivation = []
    for row in x:
        output_row = []
        for column in range(len(b)):
            value = sum(row[k] * w[k][column] for k in range(len(w))) + b[column]
            output_row.append(value)
        preactivation.append(output_row)
    prediction = [[max(value, 0.0) for value in row] for row in preactivation]
    return preactivation, prediction


def reference_loss(x: Matrix, w: Matrix, b: Sequence[float], target: Matrix) -> float:
    """Evaluate the fixed-shape example using scalar Python arithmetic."""
    _, prediction = _forward_values(x, w, b)
    squared_error = 0.0
    for row_index, row in enumerate(prediction):
        for column_index, value in enumerate(row):
            squared_error += (value - target[row_index][column_index]) ** 2
    return squared_error / (len(x) * len(b))


def reference_step() -> dict[str, object]:
    """Compute forward, analytic gradients and one SGD update for the example."""
    preactivation, prediction = _forward_values(X, W, B)
    gradient = []
    element_count = len(X) * len(B)
    for row_index, row in enumerate(preactivation):
        gradient_row = []
        for column_index, value in enumerate(row):
            derivative = 0.0
            if value > 0:
                derivative = (
                    2
                    * (
                        prediction[row_index][column_index]
                        - TARGET[row_index][column_index]
                    )
                    / element_count
                )
            gradient_row.append(derivative)
        gradient.append(gradient_row)

    grad_w = [
        [sum(X[i][k] * gradient[i][j] for i in range(len(X))) for j in range(len(B))]
        for k in range(len(W))
    ]
    grad_b = [sum(row[j] for row in gradient) for j in range(len(B))]
    grad_x = [
        [sum(gradient[i][j] * W[k][j] for j in range(len(B))) for k in range(len(W))]
        for i in range(len(X))
    ]
    updated_w = [
        [value - LEARNING_RATE * grad_w[k][j] for j, value in enumerate(row)]
        for k, row in enumerate(W)
    ]
    updated_b = [value - LEARNING_RATE * grad_b[j] for j, value in enumerate(B)]
    return {
        "preactivation": preactivation,
        "prediction": prediction,
        "loss_before": reference_loss(X, W, B, TARGET),
        "grad_x": grad_x,
        "grad_w": grad_w,
        "grad_b": grad_b,
        "updated_w": updated_w,
        "updated_b": updated_b,
        "loss_after": reference_loss(X, updated_w, updated_b, TARGET),
    }
