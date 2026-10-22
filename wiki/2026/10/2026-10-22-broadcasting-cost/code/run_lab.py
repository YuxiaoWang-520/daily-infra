"""Print reproducible semantic observations; no performance claims."""

import json

import torch

from answers import broadcast_shape, pairwise_sqdist


def main() -> None:
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([10.0, 20.0, 30.0])
    expanded = bias.expand(2, 3)
    repeated = bias.repeat(2, 1)
    output = x + bias
    leaf = bias.clone().requires_grad_()
    (x + leaf).sum().backward()
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    report = {
        "runtime": {"torch": torch.__version__, "device": "cpu"},
        "bias_shape": list(bias.shape),
        "expanded_stride": list(expanded.stride()),
        "logical_elements": expanded.numel(),
        "bias_storage_bytes": bias.untyped_storage().nbytes(),
        "expanded_storage_bytes": expanded.untyped_storage().nbytes(),
        "repeated_storage_bytes": repeated.untyped_storage().nbytes(),
        "expand_shares_known_storage": expanded.untyped_storage().data_ptr()
        == bias.untyped_storage().data_ptr(),
        "output": output.tolist(),
        "bias_gradient_of_sum": leaf.grad.tolist(),
        "zero_dimension_broadcast": list(broadcast_shape((0, 3), (1, 3))),
        "pairwise_squared_distance": pairwise_sqdist(points, points, 2, 2).tolist(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
