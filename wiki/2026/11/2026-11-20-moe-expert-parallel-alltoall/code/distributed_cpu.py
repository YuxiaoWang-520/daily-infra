"""Two CPU/Gloo expert owners: variable-split dispatch, combine and backward.

This is a correctness check. Each process holds full synthetic inputs solely for
an independent dense reference. No GPU, TP, capacity dropping or overlap test.
"""

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from answers import combine_tokens, dispatch_plan, evaluate_experts, route_tokens
from example_data import synthetic_inputs


def exchange(
    x: torch.Tensor, output_splits: tuple[int, ...], input_splits: tuple[int, ...]
) -> torch.Tensor:
    output = x.new_empty((sum(output_splits), *x.shape[1:]))
    dist.all_to_all_single(
        output, x.contiguous(), list(output_splits), list(input_splits)
    )
    return output


class DifferentiableAllToAll(torch.autograd.Function):
    """First-order teaching wrapper; backward exchanges gradients in reverse."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits):
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        return exchange(x, output_splits, input_splits)

    @staticmethod
    def backward(ctx, grad_output):
        return exchange(grad_output, ctx.input_splits, ctx.output_splits), None, None


def dense_reference(x, logits, w1, w2):
    selected = logits.topk(2, dim=-1).indices
    weights = logits.gather(1, selected).softmax(dim=-1)
    gates = torch.zeros_like(logits).scatter(1, selected, weights)
    all_outputs = torch.stack([torch.tanh(x @ a) @ b for a, b in zip(w1, w2)], dim=1)
    return (gates[:, :, None] * all_outputs).sum(dim=1)


def error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    return float((actual.detach() - expected.detach()).abs().max())


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
        for skew in (False, True):
            full = synthetic_inputs(skew=skew)
            token_slice = slice(2 * rank, 2 * rank + 2)
            expert_slice = slice(2 * rank, 2 * rank + 2)
            x = full[0][token_slice].clone().requires_grad_(True)
            logits = full[1][token_slice].clone().requires_grad_(True)
            w1 = full[2][expert_slice].clone().requires_grad_(True)
            w2 = full[3][expert_slice].clone().requires_grad_(True)
            routes = route_tokens(logits, 2)
            plan = dispatch_plan(routes, 2)
            counts = torch.tensor(plan.send_splits, dtype=torch.int64)
            received_counts = torch.empty_like(counts)
            dist.all_to_all_single(received_counts, counts)
            receive_splits = tuple(received_counts.tolist())
            payload = x.index_select(
                0, routes.token_indices.index_select(0, plan.order)
            )
            expert_ids = routes.expert_indices.index_select(0, plan.order)
            received = DifferentiableAllToAll.apply(
                payload, receive_splits, plan.send_splits
            )
            received_ids = exchange(expert_ids, receive_splits, plan.send_splits)
            values = evaluate_experts(
                received, received_ids, w1, w2, expert_offset=2 * rank
            )
            returned = DifferentiableAllToAll.apply(
                values, plan.send_splits, receive_splits
            )
            original_assignment_order = returned.index_select(0, plan.order.argsort())
            out = combine_tokens(original_assignment_order, routes)
            # Both token owners use the same global denominator: 4 tokens * 2 outputs.
            (out.square().sum() / 8).backward()
            references = tuple(t.clone().requires_grad_(True) for t in full)
            expected = dense_reference(*references)
            expected.square().mean().backward()
            errors = {"output": error(out, expected[token_slice])}
            for name, local, whole, selection in (
                ("dx", x, references[0], token_slice),
                ("dlogits", logits, references[1], token_slice),
                ("dw1", w1, references[2], expert_slice),
                ("dw2", w2, references[3], expert_slice),
            ):
                if local.grad is None or whole.grad is None:
                    raise RuntimeError(f"missing gradient {name}")
                errors[name] = error(local.grad, whole.grad[selection])
            with torch.no_grad():
                errors["sgd_w1"] = error(
                    w1 - 0.05 * w1.grad,
                    references[2][expert_slice]
                    - 0.05 * references[2].grad[expert_slice],
                )
                errors["sgd_w2"] = error(
                    w2 - 0.05 * w2.grad,
                    references[3][expert_slice]
                    - 0.05 * references[3].grad[expert_slice],
                )
            print(
                "EP_RESULT "
                + json.dumps(
                    {
                        "rank": rank,
                        "case": "empty_owner" if skew else "mixed",
                        "status": "passed",
                        "send_splits": plan.send_splits,
                        "receive_splits": receive_splits,
                        "received_assignments": received.shape[0],
                        "local_expert_counts": torch.bincount(
                            received_ids - 2 * rank, minlength=2
                        ).tolist(),
                        "max_abs_errors": errors,
                        "global_loss_denominator": 8,
                        "capacity_dropping_tested": False,
                        "auxiliary_loss_included": False,
                        "weights_applied_on_source": True,
                        "gpu_or_overlap_tested": False,
                    }
                ),
                flush=True,
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ep-teaching-") as directory:
        mp.spawn(
            worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True
        )


if __name__ == "__main__":
    main()
