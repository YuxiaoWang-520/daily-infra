"""Two local CPU/Gloo ranks validating actual PyTorch TP layout transitions."""

import copy
import datetime
import json

import torch
import torch.distributed as dist
from example_data import make_inputs
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def _model() -> nn.Sequential:
    _, w1, b1, w2, b2, _ = make_inputs()
    model = nn.Sequential(
        nn.Linear(2, 4, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(4, 2, dtype=torch.float64),
    )
    with torch.no_grad():
        model[0].weight.copy_(w1.T)
        model[0].bias.copy_(b1)
        model[2].weight.copy_(w2.T)
        model[2].bias.copy_(b2)
    return model


def _counts(mode: CommDebugMode) -> dict[str, int]:
    return {
        str(operation): count for operation, count in mode.get_comm_counts().items()
    }


def run_case(mesh, *, gather_hidden: bool) -> dict[str, object]:
    """Compare a paired layout or an explicitly gathered hidden activation."""
    rank, size = dist.get_rank(), dist.get_world_size()
    dense = _model()
    model = copy.deepcopy(dense)
    x, _, _, _, _, target = make_inputs()
    baseline_x = x.clone().requires_grad_()
    parallel_x = x.clone().requires_grad_()
    baseline = dense(baseline_x)
    (baseline - target).square().mean().backward()

    column = (
        ColwiseParallel(output_layouts=Replicate())
        if gather_hidden
        else ColwiseParallel()
    )
    row = (
        RowwiseParallel(input_layouts=Replicate())
        if gather_hidden
        else RowwiseParallel()
    )
    parallelize_module(model, mesh, {"0": column, "2": row})
    hidden_shape = []
    model[1].register_forward_hook(
        lambda module, inputs, output: hidden_shape.extend(output.shape)
    )
    forward_mode, backward_mode = CommDebugMode(), CommDebugMode()
    with forward_mode:
        output = model(parallel_x)
    loss = (output - target).square().mean()
    with backward_mode:
        loss.backward()
    torch.testing.assert_close(output, baseline, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(parallel_x.grad, baseline_x.grad, rtol=1e-12, atol=1e-12)

    maximum_gradient_error = 0.0
    shapes = {}
    for name, parameter in model.named_parameters():
        expected = dict(dense.named_parameters())[name].grad
        if name in ("0.weight", "0.bias"):
            expected = expected.chunk(size, dim=0)[rank]
        elif name == "2.weight":
            expected = expected.chunk(size, dim=1)[rank]
        actual = parameter.grad.to_local()
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        maximum_gradient_error = max(
            maximum_gradient_error, (actual - expected).abs().max().item()
        )
        shapes[name] = list(parameter.to_local().shape)

    return {
        "rank": rank,
        "case": "gathered_hidden" if gather_hidden else "paired_shards",
        "backend": "gloo",
        "device": "cpu",
        "world_size": size,
        "output": output.detach().tolist(),
        "loss": loss.item(),
        "hidden_local_shape": hidden_shape,
        "parameter_local_shapes": shapes,
        "input_gradient": parallel_x.grad.tolist(),
        "max_parameter_gradient_abs_error": maximum_gradient_error,
        "forward_collective_counts": _counts(forward_mode),
        "backward_collective_counts": _counts(backward_mode),
    }


def main() -> None:
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=45))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("Run this fixed example with exactly two local ranks")
        mesh = init_device_mesh("cpu", (2,))
        for gather_hidden in (False, True):
            result = run_case(mesh, gather_hidden=gather_hidden)
            print("TP_RESULT " + json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
