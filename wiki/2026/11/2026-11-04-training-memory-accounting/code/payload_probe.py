"""Real CPU Adam snapshots and alias storage evidence, without GPU claims."""

import json
import math
from contextlib import nullcontext

import torch


def tensor_info(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
    }


def snapshot(layer: torch.nn.Linear, optimizer: torch.optim.Adam) -> dict[str, object]:
    parameters = list(layer.parameters())
    groups: dict[str, list[torch.Tensor]] = {
        "parameters": parameters,
        "gradients": [p.grad for p in parameters if p.grad is not None],
        "first_moment": [],
        "second_moment": [],
        "step": [],
    }
    state_keys = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        state_keys.append(sorted(state))
        for key, group in (
            ("exp_avg", "first_moment"),
            ("exp_avg_sq", "second_moment"),
            ("step", "step"),
        ):
            if key in state:
                groups[group].append(state[key])
    tensors = [t for group in groups.values() for t in group]
    # Addresses identify simultaneous live storages internally; never publish them.
    unique = {
        (str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
        for t in tensors
    }
    return {
        "groups": {key: [tensor_info(t) for t in ts] for key, ts in groups.items()},
        "group_bytes": {
            key: sum(t.numel() * t.element_size() for t in ts)
            for key, ts in groups.items()
        },
        "optimizer_state_keys": state_keys,
        "logical_payload_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "unique_storage_bytes": sum(unique.values()),
        "unique_storage_count": len(unique),
    }


def profile(name: str, dtype: torch.dtype, use_autocast: bool) -> dict[str, object]:
    layer = torch.nn.Linear(4, 4, bias=False, dtype=dtype, device="cpu")
    with torch.no_grad():
        layer.weight.fill_(0.25)
    optimizer = torch.optim.Adam(
        layer.parameters(), lr=0.001, foreach=False, fused=False, capturable=False
    )
    observations = {"constructed": snapshot(layer, optimizer)}
    inputs = torch.arange(8, dtype=dtype).reshape(2, 4) / 8
    context = (
        torch.autocast("cpu", dtype=torch.bfloat16) if use_autocast else nullcontext()
    )
    with context:
        output = layer(inputs)
        loss = 0.5 * output.float().square().mean()
    output_dtype = str(output.dtype)
    loss.backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    observations["after_backward"] = snapshot(layer, optimizer)
    optimizer.step()
    observations["after_step"] = snapshot(layer, optimizer)
    optimizer.zero_grad(set_to_none=True)
    observations["after_zero_grad_none"] = snapshot(layer, optimizer)
    width = layer.weight.element_size()
    expected = (16 * width, 32 * width, 64 * width + 4, 48 * width + 4)
    assert (
        tuple(row["logical_payload_bytes"] for row in observations.values()) == expected
    )
    state = optimizer.state[layer.weight]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == dtype
    assert state["step"].numel() == 1 and state["step"].element_size() == 4
    assert state["step"].device.type == "cpu"
    assert output_dtype == str(torch.bfloat16 if use_autocast else dtype)
    assert layer.weight.dtype == dtype and layer.weight.grad is None
    assert all(
        row["logical_payload_bytes"] == row["unique_storage_bytes"]
        for row in observations.values()
    )
    return {
        "name": name,
        "parameter_elements": 16,
        "output_dtype": output_dtype,
        "autocast": use_autocast,
        "snapshots": observations,
    }


def alias_probe() -> dict[str, object]:
    base = torch.arange(16, dtype=torch.float32)
    view = base[:8]
    copy = view.clone()
    assert base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
    assert base.untyped_storage().data_ptr() != copy.untyped_storage().data_ptr()
    tensors = {"base": base, "view": view, "clone": copy}
    logical = sum(t.numel() * t.element_size() for t in tensors.values())
    unique = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in tensors.values()
    }
    assert logical == 128 and sum(unique.values()) == 96
    return {
        "tensors": {name: tensor_info(t) for name, t in tensors.items()},
        "logical_sum_bytes": logical,
        "unique_storage_bytes": sum(unique.values()),
        "base_and_view_share_storage": True,
    }


def unused_parameter_probe() -> dict[str, object]:
    used = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = torch.optim.Adam([used, unused], foreach=False, fused=False)
    used.square().sum().backward()
    optimizer.step()
    assert used in optimizer.state and unused not in optimizer.state
    assert unused.grad is None
    state = optimizer.state[used]
    first = state["exp_avg"].item()
    second = state["exp_avg_sq"].item()
    assert math.isclose(first, 0.2, rel_tol=1e-12)
    assert math.isclose(second, 0.004, rel_tol=1e-12)
    expected_weight = 1 - 0.001 * 2 / (math.sqrt(4) + 1e-8)
    assert math.isclose(used.item(), expected_weight, rel_tol=1e-12)
    return {
        "used_has_state": True,
        "unused_has_state": False,
        "unused_grad_is_none": True,
        "scalar_adam_check": {
            "dtype": "float64",
            "gradient": used.grad.item(),
            "first_moment": first,
            "second_moment": second,
            "weight_after_step": used.item(),
            "expected_weight": expected_weight,
        },
    }


def main() -> None:
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    result = {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "scope": "selected CPU tensor payload snapshots; not RSS or a GPU peak",
        "profiles": [
            profile("fp32", torch.float32, False),
            profile("fp32_parameters_cpu_autocast_bf16", torch.float32, True),
            profile("explicit_bf16_parameters", torch.bfloat16, False),
        ],
        "alias": alias_probe(),
        "lazy_unused_parameter": unused_parameter_probe(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
