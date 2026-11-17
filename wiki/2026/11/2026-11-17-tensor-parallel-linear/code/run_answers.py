"""Print numerical teaching results and clearly labeled ring estimates."""

import json
import platform

import torch
from answers import collective_budget, tp_mlp
from example_data import make_inputs


def main() -> None:
    values = make_inputs()
    trainable = tuple(value.requires_grad_() for value in values[:5])
    result = tp_mlp(*trainable, 2)
    loss = (result["output"] - values[5]).square().mean()
    gradients = torch.autograd.grad(loss, trainable)
    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_build_commit": torch.version.git_version,
            "os": platform.system(),
            "architecture": platform.machine(),
            "cuda_available": torch.cuda.is_available(),
            "gloo_available": torch.distributed.is_gloo_available(),
        },
        "single_process_tp_algebra": {
            "tp_size": 2,
            "preactivations": [
                part.detach().tolist() for part in result["preactivations"]
            ],
            "hidden_shards": [
                part.detach().tolist() for part in result["hidden_shards"]
            ],
            "partial_outputs": [
                part.detach().tolist() for part in result["partial_outputs"]
            ],
            "output": result["output"].detach().tolist(),
            "loss": loss.item(),
            "gradients": {
                name: gradient.tolist()
                for name, gradient in zip(
                    ("x", "w1", "b1", "w2", "b2"), gradients, strict=True
                )
            },
        },
        "ideal_equal_chunk_ring_estimates": {
            "inputs": {
                "tokens": 2048,
                "hidden": 8192,
                "tp_size": 8,
                "element_bytes": 2,
            },
            "two_allreduces": collective_budget(2048, 8192, 8, 2),
            "sequence_layout_transforms_only": collective_budget(
                2048, 8192, 8, 2, sequence_parallel=True
            ),
            "sequence_plus_input_regather": collective_budget(
                2048, 8192, 8, 2, sequence_parallel=True, regather_input_for_wgrad=True
            ),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
