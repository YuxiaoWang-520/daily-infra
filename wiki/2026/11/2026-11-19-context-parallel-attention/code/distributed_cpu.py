"""Two real CPU/Gloo ranks: differentiable gather baseline, then P2P forward.

This is a correctness check, not a memory/speed benchmark. Each process also
constructs full synthetic inputs for its independent dense reference. The P2P
loop waits between steps and does not implement distributed ring backward.
"""

import json
import tempfile
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from answers import KVBlock, ring_order, streaming_attention
from example_data import synthetic_inputs
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], is_causal=True
    )[0, 0]


def error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    return float((actual.detach() - expected.detach()).abs().max())


def ring_blocks(
    k: torch.Tensor, v: torch.Tensor, rank: int, size: int
) -> Iterator[KVBlock]:
    """Forward-only K/V rotation; metadata derives from equal contiguous shards."""
    current = torch.stack((k, v)).contiguous()
    width = k.shape[0]
    for step, owner in enumerate(ring_order(rank, size)):
        yield KVBlock(
            current[0], current[1], torch.arange(owner * width, (owner + 1) * width)
        )
        if step == size - 1:
            break
        received = torch.empty_like(current)
        operations = [
            dist.P2POp(dist.isend, current, (rank + 1) % size),
            dist.P2POp(dist.irecv, received, (rank - 1) % size),
        ]
        for request in dist.batch_isend_irecv(operations):
            request.wait()
        current = received


def worker(rank: int, store_path: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size != 2:
            raise ValueError("launch exactly two processes for this check")
        full = synthetic_inputs()
        width = full[0].shape[0] // size
        owned = slice(rank * width, (rank + 1) * width)
        q, k, v = [x[owned].detach().clone().requires_grad_(True) for x in full]
        qp = torch.arange(rank * width, (rank + 1) * width)
        # Local losses sum to the dense global mean, without a second division
        # by CP size. all_gather's backward sums remote query contributions.
        gathered_k = torch.cat(all_gather(k))
        gathered_v = torch.cat(all_gather(v))
        out, _ = streaming_attention(
            q, [KVBlock(gathered_k, gathered_v, torch.arange(full[0].shape[0]))], qp
        )
        (out.square().sum() / full[2].numel()).backward()
        qr, kr, vr = [x.detach().clone().requires_grad_(True) for x in full]
        expected = reference(qr, kr, vr)
        expected.square().mean().backward()
        errors = {"output": error(out, expected[owned])}
        for name, local, whole in (("dq", q, qr), ("dk", k, kr), ("dv", v, vr)):
            if local.grad is None or whole.grad is None:
                raise RuntimeError(f"missing gradient: {name}")
            errors[name] = error(local.grad, whole.grad[owned])
        print(
            "CP_RESULT "
            + json.dumps(
                {
                    "rank": rank,
                    "mode": "autograd_all_gather",
                    "status": "passed",
                    "shape": [8, 2],
                    "dtype": "float64",
                    "positions": qp.tolist(),
                    "max_abs_errors": errors,
                }
            ),
            flush=True,
        )
        dist.barrier()
        with torch.no_grad():
            ring_out, _ = streaming_attention(q, ring_blocks(k, v, rank, size), qp)
            ring_error = error(ring_out, expected[owned])
        print(
            "CP_RESULT "
            + json.dumps(
                {
                    "rank": rank,
                    "mode": "p2p_forward_only",
                    "status": "passed",
                    "owner_order": ring_order(rank, size),
                    "max_abs_errors": {"output": ring_error},
                    "distributed_ring_backward_tested": False,
                    "compute_communication_overlap_tested": False,
                }
            ),
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    # A local FileStore avoids torchrun's separate TCP rendezvous. Gloo still
    # performs real communication between the two spawned CPU processes.
    with tempfile.TemporaryDirectory(prefix="cp-teaching-") as directory:
        store_path = str(Path(directory) / "rendezvous")
        mp.spawn(worker, args=(store_path,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
