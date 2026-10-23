"""Confirm values and show how an assumed penalty reverses a toy cost choice."""

import json

import torch

from answers import chunked_affine_relu, pipeline_model


def main() -> None:
    x = torch.tensor([[-2.0, 0.0, 2.0], [1.0, -1.0, 3.0]])
    bias = torch.tensor([1.0, 2.0, -1.0])
    actual = chunked_affine_relu(x, bias, 2.0, 1)
    torch.testing.assert_close(actual, torch.relu((x + bias) * 2.0), rtol=0, atol=0)
    good = pipeline_model(1024, 1024, 4)
    bad = pipeline_model(1024, 1024, 4, fused_slowdown=5.0)
    assert good["fused_seconds"] < good["staged_seconds"]
    assert bad["fused_seconds"] > bad["staged_seconds"]
    print(
        json.dumps(
            {
                "scope": "CPU semantics and synthetic cost assumptions; no kernel fusion measured",
                "hand_result": actual.tolist(),
                "hypothetical_parameters": {"bandwidth_gbs": 1000, "launch_us": 5},
                "no_penalty": good,
                "assumed_fused_slowdown5": bad,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
