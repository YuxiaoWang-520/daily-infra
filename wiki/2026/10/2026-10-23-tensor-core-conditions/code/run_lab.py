"""Observe CPU algebra and rounding; no claim of matrix hardware usage."""

import json

import torch

from answers import padding_plan, quantized_padded_matmul


def main() -> None:
    a = torch.tensor([[1.001]], dtype=torch.float64)
    b = torch.ones(1, 1, dtype=torch.float64)
    half = quantized_padded_matmul(a, b, torch.float16).item()
    bf16 = quantized_padded_matmul(a, b, torch.bfloat16).item()
    assert half == 1.0009765625 and bf16 == 1.0
    plan = padding_plan(17, 9, 17)
    assert plan["padded_shape"] == (32, 16, 32)
    print(
        json.dumps(
            {
                "scope": "CPU padding and input quantization only; no Tensor Core execution",
                "chosen_tile": [16, 8, 16],
                "plan_17_9_17": plan,
                "original_real_target": 1.001,
                "half_storage_fp32_cpu_result": half,
                "bf16_storage_fp32_cpu_result": bf16,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
