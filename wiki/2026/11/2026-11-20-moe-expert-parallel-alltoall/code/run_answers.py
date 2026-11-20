"""Hand-checkable routing outputs and theoretical payload counts, not timings."""

import json

from answers import combine_tokens, dispatch_plan, load_balance_loss, route_tokens
from example_data import example_logits


def main() -> None:
    logits = example_logits()
    cases = {}
    for cap in (None, 2):
        routes = route_tokens(logits, 2, capacity=cap)
        values = (routes.expert_indices + 1).to(logits.dtype) * (
            routes.token_indices + 1
        )
        cases[str(cap)] = {
            "expert_counts": routes.expert_counts.tolist(),
            "kept_assignments": routes.weights.numel(),
            "scalar_output": combine_tokens(values[:, None], routes)[:, 0].tolist(),
        }
    split_matrix = [
        dispatch_plan(route_tokens(part, 2), 2).send_splits for part in logits.chunk(2)
    ]
    remote = sum(split_matrix[r][1 - r] for r in (0, 1))
    print(
        json.dumps(
            {
                "cases": cases,
                "source_destination_split_matrix": split_matrix,
                "auxiliary_loss_coefficient_0_01": float(
                    load_balance_loss(logits, 2, coefficient=0.01)
                ),
                "assumed_payload": {
                    "assignment_rows": 8,
                    "remote_assignment_rows": remote,
                    "hidden_dimension": 4096,
                    "bytes_per_element": 2,
                    "forward_dispatch_and_return_remote_bytes": 2 * remote * 4096 * 2,
                    "scope": "all ranks combined, remote hidden-state payload only; receive not counted twice; no gradients or metadata",
                    "measured": False,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
