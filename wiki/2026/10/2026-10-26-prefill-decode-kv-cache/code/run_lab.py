"""Print small, asserted CPU examples; work counts are analytical."""

import json

import torch

from answers import KVCache, TinyDecoder, cached_attention, greedy_generate


def main() -> None:
    old = KVCache(torch.zeros(2, 1), torch.tensor([[10.0], [20.0]]))
    output, cache = cached_attention(
        torch.zeros(2, 1), torch.zeros(2, 1), torch.tensor([[30.0], [40.0]]), old
    )
    torch.testing.assert_close(output, torch.tensor([[20.0], [25.0]]))
    model = TinyDecoder(vocab=13, width=8, layers=2, max_length=16, seed=11)
    prompt = torch.tensor([2, 5, 1])
    cached = greedy_generate(model, prompt, 4)
    full = greedy_generate(model, prompt, 4, use_cache=False)
    assert torch.equal(cached.tokens, full.tokens)
    assert [c.length for c in cached.caches] == [6, 6]
    _, recomputed = model.full(cached.tokens[:-1])
    differences = []
    for actual, expected in zip(cached.caches, recomputed):
        torch.testing.assert_close(actual.k, expected.k, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(actual.v, expected.v, rtol=1e-11, atol=1e-11)
        differences.append(
            {
                "k_max_abs": (actual.k - expected.k).abs().max().item(),
                "v_max_abs": (actual.v - expected.v).abs().max().item(),
            }
        )
    p, g = 3, 4
    work = {
        "full_forward_positions_per_layer": sum(p + i for i in range(g)),
        "cached_forward_positions_per_layer": p + g - 1,
        "full_valid_causal_pairs_per_layer_head": sum((p + i) * (p + i + 1) // 2 for i in range(g)),
        "cached_valid_causal_pairs_per_layer_head": p * (p + 1) // 2
        + sum(p + i for i in range(1, g)),
    }
    assert list(work.values()) == [18, 6, 52, 21]
    print(
        json.dumps(
            {
                "scope": "CPU semantic reference and analytical counts; no vLLM execution or GPU timing",
                "uniform_attention_two_new_queries": output.tolist(),
                "updated_cache_length": cache.length,
                "prompt_ids": prompt.tolist(),
                "generated_full_ids": full.tokens.tolist(),
                "generated_cached_ids": cached.tokens.tolist(),
                "full_forward_lengths": full.forward_lengths,
                "cached_forward_lengths": cached.forward_lengths,
                "final_cache_lengths": [c.length for c in cached.caches],
                "full_state_max_absolute_errors": differences,
                "work_model_not_measured_flops_or_speedup": work,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
