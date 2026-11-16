"""Trace a synthetic PyTorch CPU step without exporting private file paths."""

import json
import platform

import torch
from example_data import LEARNING_RATE, TARGET, B, W, X
from reference import reference_step


def forward(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preactivation = x @ w + b
    prediction = torch.relu(preactivation)
    loss = (prediction - target).square().mean()
    return preactivation, prediction, loss


def _autograd_node_types(loss: torch.Tensor) -> list[str]:
    pending = [loss.grad_fn]
    visited = set()
    names = set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        names.add(type(node).__name__)
        pending.extend(child for child, _ in node.next_functions)
    return sorted(names)


def run_step(dtype_name: str = "float32") -> dict[str, object]:
    """Run a CPU-only step and retain operator names, counts and shapes."""
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    x, w, b = [
        torch.tensor(value, dtype=dtype, device="cpu", requires_grad=True)
        for value in (X, W, B)
    ]
    target = torch.tensor(TARGET, dtype=dtype, device="cpu")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as profile:
        with torch.profiler.record_function("tutorial::forward"):
            preactivation, prediction, loss = forward(x, w, b, target)
            node_types = _autograd_node_types(loss)
        with torch.profiler.record_function("tutorial::backward"):
            loss.backward()
        if x.grad is None or w.grad is None or b.grad is None:
            raise RuntimeError("The example must produce all three gradients")
        gradients = {
            "grad_x": x.grad.tolist(),
            "grad_w": w.grad.tolist(),
            "grad_b": b.grad.tolist(),
        }
        with torch.profiler.record_function("tutorial::sgd"), torch.no_grad():
            w.add_(w.grad, alpha=-LEARNING_RATE)
            b.add_(b.grad, alpha=-LEARNING_RATE)

    # The verification forward is outside the measured one-step operator scope.
    with torch.no_grad():
        _, _, updated_loss = forward(x, w, b, target)
    operators = [
        {"name": event.key, "count": event.count, "input_shapes": event.input_shapes}
        for event in profile.key_averages(group_by_input_shape=True)
    ]
    operators.sort(key=lambda event: (event["name"], str(event["input_shapes"])))
    return {
        "dtype": dtype_name,
        "device": "cpu",
        "preactivation": preactivation.detach().tolist(),
        "prediction": prediction.detach().tolist(),
        "loss_before": loss.detach().item(),
        **gradients,
        "updated_w": w.detach().tolist(),
        "updated_b": b.detach().tolist(),
        "loss_after": updated_loss.item(),
        "autograd_node_types": node_types,
        "cpu_operators": operators,
    }


def main() -> None:
    graph = torch.fx.symbolic_trace(forward)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_build_commit": torch.version.git_version,
            "os": platform.system(),
            "architecture": platform.machine(),
            "cuda_available": torch.cuda.is_available(),
        },
        "scalar_reference": reference_step(),
        "cpu_runs": [run_step("float32"), run_step("float64")],
        "fx_forward_code": graph.code,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
