"""CPU enumeration of CUDA-style 1D ownership and byte-sector geometry."""

from collections.abc import Iterable
from dataclasses import dataclass


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ThreadWork:
    block: int
    thread: int
    index: int
    active: bool
    warp: int
    lane: int


def launch_map(n: int, block_size: int) -> list[ThreadWork]:
    """Enumerate ownership, not scheduling; assumes NVIDIA 32-thread warps.

    Positive block sizes here are mathematical inputs, not validated device limits.
    n=0 returns no rows; a CUDA caller should skip a zero-sized launch.
    """
    _integer(n, "n")
    _integer(block_size, "block_size", 1)
    blocks = (n + block_size - 1) // block_size
    return [
        ThreadWork(b, t, b * block_size + t, b * block_size + t < n, t // 32, t % 32)
        for b in range(blocks)
        for t in range(block_size)
    ]


def sector_ids(
    addresses: Iterable[int], width: int = 4, sector_bytes: int = 32
) -> frozenset[int]:
    """Union of aligned sectors touched by byte intervals [a,a+width).

    No cache, instruction splitting, hardware transactions or timing is simulated.
    Call once per conceptual instruction and active lane group, not across kernels.
    """
    _integer(width, "width", 1)
    _integer(sector_bytes, "sector_bytes", 1)
    sectors = set()
    for address in addresses:
        _integer(address, "address")
        first = address // sector_bytes
        last = (address + width - 1) // sector_bytes
        sectors.update(range(first, last + 1))
    return frozenset(sectors)
