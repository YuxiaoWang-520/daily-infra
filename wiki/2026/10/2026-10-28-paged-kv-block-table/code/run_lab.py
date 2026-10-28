"""Show page counts, noncontiguous slots, exhaustion, reuse, and CPU semantics."""

import json

import torch
from answers import PagedKVCache, page_usage, slot_for_token
from torch.nn import functional as F


def main() -> None:
    cache = PagedKVCache(num_blocks=4, block_size=4, head_dim=2)
    key = torch.arange(18, dtype=torch.float64, device="cpu").reshape(9, 2) / 9
    value = torch.arange(20, 38, dtype=torch.float64, device="cpu").reshape(9, 2) / 11
    cache.append("a", key[:3], value[:3])
    cache.append("b", key[:5] + 10, value[:5] + 10)
    cache.append("a", key[3:7], value[3:7])
    initial = {
        "a_table": cache.block_table("a"),
        "b_table": cache.block_table("b"),
        "a_length": cache.length("a"),
        "free_blocks": cache.free_block_count,
    }
    before_k, before_v = cache.read("a")
    try:
        cache.append("a", key[7:], value[7:])
    except MemoryError:
        rejected = True
    else:
        raise AssertionError("two new rows cross a page boundary and must fail")
    torch.testing.assert_close(cache.read("a")[0], before_k, rtol=0, atol=0)
    torch.testing.assert_close(cache.read("a")[1], before_v, rtol=0, atol=0)
    cache.append("a", key[7:8], value[7:8])
    self_contained = {
        "length": cache.length("a"),
        "table": cache.block_table("a"),
        "free_blocks": cache.free_block_count,
    }
    cache.release("b")
    reused_slots = cache.append("a", key[8:], value[8:])
    gathered_k, gathered_v = cache.read("a")
    torch.testing.assert_close(gathered_k, key, rtol=0, atol=0)
    torch.testing.assert_close(gathered_v, value, rtol=0, atol=0)
    q = torch.tensor([[[[0.3, -0.4]]]], dtype=torch.float64, device="cpu")
    actual = F.scaled_dot_product_attention(
        q, gathered_k[None, None], gathered_v[None, None], dropout_p=0.0
    )
    expected = F.scaled_dot_product_attention(
        q, key[None, None], value[None, None], dropout_p=0.0
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    print(
        json.dumps(
            {
                "scope": "CPU logical paging and analytical bytes; not GPU memory or vLLM execution",
                "address_example": [
                    slot_for_token((3, 5), i, block_size=4, length=7, num_blocks=8)
                    for i in range(7)
                ],
                "page_accounting": page_usage([5, 3], block_size=4, bytes_per_token=32),
                "initial_interleaved_requests": initial,
                "failed_two_row_append_preserved_prefix": rejected,
                "one_row_append_without_free_blocks": self_contained,
                "after_release_and_reuse": {
                    "a_table": cache.block_table("a"),
                    "a_length": cache.length("a"),
                    "new_slots": reused_slots,
                    "free_blocks": cache.free_block_count,
                },
                "decode_sdpa_max_abs_difference": (actual - expected)
                .abs()
                .max()
                .item(),
                "no_gpu_timing_or_memory_measurement": True,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
