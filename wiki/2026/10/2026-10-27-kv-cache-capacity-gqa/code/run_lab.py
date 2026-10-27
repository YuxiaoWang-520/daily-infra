"""Print reproducible CPU semantics and analytical capacities, never GPU timings."""

import json

import torch
from answers import grouped_attention, kv_cache_bytes, kv_heads_per_rank
from torch.nn import functional as F


def main() -> None:
    capacities = {}
    tp = {}
    for name, heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
        total = kv_cache_bytes(
            [8192], layers=32, kv_heads=heads, head_dim=128, element_bytes=2
        )
        local_heads = kv_heads_per_rank(heads, 4)
        local = kv_cache_bytes(
            [8192], layers=32, kv_heads=local_heads, head_dim=128, element_bytes=2
        )
        capacities[name] = {
            "kv_heads": heads,
            "one_request_bytes": total,
            "eight_requests_bytes": total * 8,
        }
        tp[name] = {
            "ranks": 4,
            "heads_per_rank": local_heads,
            "bytes_per_rank": local,
            "cluster_bytes": local * 4,
        }

    generator = torch.Generator().manual_seed(194)
    q = torch.randn(1, 4, 2, 3, dtype=torch.float64, generator=generator, device="cpu")
    k = torch.randn(1, 2, 5, 3, dtype=torch.float64, generator=generator, device="cpu")
    v = torch.randn(1, 2, 5, 3, dtype=torch.float64, generator=generator, device="cpu")
    actual = grouped_attention(q, k, v, query_start=3)
    expanded_k = k.repeat_interleave(2, dim=1)
    expanded_v = v.repeat_interleave(2, dim=1)
    mask = (
        torch.arange(5, device="cpu")[None, :]
        <= torch.arange(3, 5, device="cpu")[:, None]
    )
    expected = F.scaled_dot_product_attention(
        q, expanded_k, expanded_v, attn_mask=mask, dropout_p=0.0
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    compact_bytes = (k.numel() + v.numel()) * k.element_size()
    repeated_bytes = (expanded_k.numel() + expanded_v.numel()) * k.element_size()
    assert compact_bytes == 480 and repeated_bytes == 960
    print(
        json.dumps(
            {
                "scope": "CPU semantic check and analytical byte counts; no GPU allocation or timing",
                "capacity_assumptions": {
                    "layers": 32,
                    "head_dim": 128,
                    "element_bytes": 2,
                    "tokens_per_request": 8192,
                },
                "capacities": capacities,
                "tensor_parallel_4": tp,
                "gqa_check": {
                    "dtype": str(q.dtype),
                    "q_shape": list(q.shape),
                    "kv_shape": list(k.shape),
                    "output_shape": list(actual.shape),
                    "logical_compact_kv_bytes": compact_bytes,
                    "logical_expanded_kv_bytes": repeated_bytes,
                    "max_abs_difference_from_sdpa": (actual - expected)
                    .abs()
                    .max()
                    .item(),
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
