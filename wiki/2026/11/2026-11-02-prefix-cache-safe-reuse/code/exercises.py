"""Student interfaces. Contracts and expected states are in code/README.md."""

from models import PoolView, PrefixContext


def block_hashes(
    tokens: list[int] | tuple[int, ...], block_size: int, context: PrefixContext
) -> tuple[str, ...]:
    raise NotImplementedError("Exercise 1: hash complete blocks with their prefix")


def reusable_tokens(
    tokens: list[int] | tuple[int, ...],
    block_size: int,
    context: PrefixContext,
    available: set[str] | frozenset[str],
) -> int:
    raise NotImplementedError("Exercise 1: probe a contiguous aligned prefix")


class PrefixPool:
    def __init__(self, capacity: int) -> None:
        raise NotImplementedError("Exercise 2: initialize a fixed metadata pool")

    def acquire(self, request_id: str, keys: tuple[str, ...]) -> tuple[int, ...]:
        raise NotImplementedError("Exercise 2: protect hits before allocating misses")

    def complete(self, request_id: str) -> None:
        raise NotImplementedError("Exercise 2: make completed full blocks reusable")

    def release(self, request_id: str) -> None:
        raise NotImplementedError(
            "Exercise 2: release references without deleting warm KV"
        )

    def snapshot(self) -> PoolView:
        raise NotImplementedError("Exercise 2: return an immutable state snapshot")
