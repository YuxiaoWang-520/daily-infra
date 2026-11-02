"""Print real CPU hash/metadata results; no model output or latency measured."""

import json
from dataclasses import asdict

from answers import PrefixPool, block_hashes, reusable_tokens
from models import PrefixContext


def main() -> None:
    context = PrefixContext("toy-model@revision-1/position-v1/fp16", "base", "group-a")
    original = list(range(1, 11))
    cached = set(block_hashes(original, 4, context))
    comparisons = []
    for name, tokens, request_context in (
        ("eight_common_positions", list(range(1, 9)) + [90, 91], context),
        ("seven_common_positions", list(range(1, 8)) + [90, 91, 92], context),
        ("changed_first_position", [90] + list(range(2, 11)), context),
        (
            "different_trust_group",
            original,
            PrefixContext(context.namespace, "base", "group-b"),
        ),
        ("identical_eight_position_prompt", list(range(1, 9)), context),
    ):
        hit = reusable_tokens(tokens, 4, request_context, cached)
        comparisons.append(
            dict(
                case=name,
                prompt_length=len(tokens),
                hit_tokens=hit,
                new_positions=len(tokens) - hit,
            )
        )
    assert [case["hit_tokens"] for case in comparisons] == [8, 4, 0, 0, 4]

    pool = PrefixPool(3)
    states = []

    def record(label: str) -> None:
        states.append(dict(event=label, state=asdict(pool.snapshot())))

    record("empty")
    assert pool.acquire("r1", ("a", "b")) == (0, 1)
    record("r1_pending")
    pool.complete("r1")
    record("r1_ready")
    pool.release("r1")
    assert pool.snapshot().free_order == (2, 1, 0)
    record("r1_released_but_warm")
    assert pool.acquire("r2", ("a", "b", "c")) == (0, 1, 2)
    record("r2_reused_a_b_and_allocated_c")
    before = pool.snapshot()
    try:
        pool.acquire("blocked", ("d",))
    except RuntimeError:
        assert pool.snapshot() == before
    else:
        raise AssertionError("pinned capacity must not be evicted")
    record("capacity_failure_without_mutation")
    pool.complete("r2")
    pool.release("r2")
    record("r2_released")
    assert pool.acquire("r3", ("d",)) == (2,)
    assert "c" not in dict(pool.snapshot().cache_entries)
    record("c_evicted_before_d_is_ready")
    print(
        json.dumps(
            dict(
                scope="Actual CPU hash and metadata execution only. No KV tensors, model accuracy or performance measurement.",
                comparisons=comparisons,
                lifecycle=states,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
