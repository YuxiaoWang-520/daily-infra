"""Verify a tiny attention example; large-capacity numbers are integer models."""

import json

import torch

from answers import dense_attention, online_softmax, tiled_attention


def main() -> None:
    q = torch.tensor([[1.0]], dtype=torch.float64)
    k = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    v = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float64)
    actual = tiled_attention(q, k, v, 1, 2)
    torch.testing.assert_close(actual, dense_attention(q, k, v), rtol=1e-12, atol=1e-12)
    probabilities = online_softmax(k[:, 0], 2)
    m0 = 2.0
    l0 = 1 + torch.exp(torch.tensor(-1.0, dtype=torch.float64)).item()
    u0 = 20 + 10 * torch.exp(torch.tensor(-1.0, dtype=torch.float64)).item()
    alpha = torch.exp(torch.tensor(-1.0, dtype=torch.float64)).item()
    l1, u1 = l0 * alpha + 1, u0 * alpha + 30
    assert abs(u1 / l1 - actual.item()) < 1e-12
    print(
        json.dumps(
            {
                "scope": "CPU FP64 algebra, no CUDA/FlashAttention execution or performance",
                "probabilities": probabilities.tolist(),
                "first_block": {"m": m0, "l": l0, "u": u0},
                "second_block": {"m": 3.0, "alpha": alpha, "l": l1, "u": u1},
                "attention_result": actual.tolist(),
                "capacity_model_only": {
                    "length": 8192,
                    "one_head_one_fp16_score_matrix_bytes": 8192**2 * 2,
                    "32_heads_two_fp16_matrices_bytes": 32 * 2 * 8192**2 * 2,
                    "score_tile_elements_64x128": 64 * 128,
                    "full_dense_matmul_flops_d128_per_head": 4 * 8192**2 * 128,
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
