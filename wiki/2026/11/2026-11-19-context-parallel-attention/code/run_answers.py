"""Print reproducible numeric examples; bytes are estimates, not measured peaks."""

import json

import torch
from answers import (
    KVBlock,
    causal_pair_counts,
    cp_attention,
    sequence_shards,
    streaming_attention,
)
from example_data import scalar_example, synthetic_inputs


def main() -> None:
    q, k, v = scalar_example()
    blocks = [
        KVBlock(k[:2], v[:2], torch.arange(2)),
        KVBlock(k[2:], v[2:], torch.arange(2, 4)),
    ]
    out, lse = streaming_attention(q, blocks, torch.arange(4))
    layouts = {}
    qs, ks, vs = synthetic_inputs()
    for layout in ("contiguous", "zigzag"):
        shards = sequence_shards(8, 2, layout=layout)
        layouts[layout] = {
            "positions": shards,
            "causal_pair_counts": causal_pair_counts(shards),
            "output": cp_attention(qs, ks, vs, 2, layout=layout).tolist(),
        }
    # One forward, one sequence, single MHA layer; K and V each [S,Hkv,D].
    seq, heads_kv, dim, cp, bytes_per_element = 32768, 32, 128, 4, 2
    local_kv = 2 * (seq // cp) * heads_kv * dim * bytes_per_element
    print(
        json.dumps(
            {
                "scalar_causal_output": out[:, 0].tolist(),
                "scalar_logsumexp": lse.tolist(),
                "naive_local_output_average": (50 / 3 + 250 / 7) / 2,
                "layouts": layouts,
                "assumed_forward_kv_budget": {
                    "scope": "one sequence, one layer, full unpruned ring; send and receive counted separately",
                    "S": seq,
                    "Hkv": heads_kv,
                    "D": dim,
                    "cp_size": cp,
                    "bytes_per_element": bytes_per_element,
                    "owned_kv_MiB": local_kv / 2**20,
                    "fully_gathered_kv_MiB": local_kv * cp / 2**20,
                    "forward_ring_send_per_rank_MiB": local_kv * (cp - 1) / 2**20,
                    "forward_ring_receive_per_rank_MiB": local_kv * (cp - 1) / 2**20,
                    "memory_peak_measured": False,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
