"""Immutable records shared by the two CPU metadata exercises."""

from dataclasses import dataclass


def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def require_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PrefixContext:
    namespace: str
    adapter: str
    salt: str

    def __post_init__(self) -> None:
        for name in ("namespace", "adapter", "salt"):
            require_text(getattr(self, name), name)


@dataclass(frozen=True)
class BlockView:
    block_id: int
    key: str | None
    ref_count: int
    ready: bool


@dataclass(frozen=True)
class PoolView:
    blocks: tuple[BlockView, ...]
    free_order: tuple[int, ...]
    cache_entries: tuple[tuple[str, tuple[int, ...]], ...]
    requests: tuple[tuple[str, tuple[int, ...]], ...]
