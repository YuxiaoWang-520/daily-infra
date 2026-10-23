"""CPU-only tiled GEMM reference and a logical input-load model."""

import torch


def _integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _tiles(bm: int, bn: int, bk: int) -> None:
    for name, value in (("bm", bm), ("bn", bn), ("bk", bk)):
        _integer(value, name, 1)


@torch.inference_mode()
def blocked_matmul(
    a: torch.Tensor, b: torch.Tensor, bm: int, bn: int, bk: int
) -> torch.Tensor:
    """C=A@B, CPU float32/64; each tile product calls PyTorch's CPU matmul.

    Inference-only: result is an inference tensor and no gradient graph is built.
    This verifies block algebra, not a CUDA/shared-memory/Tensor Core kernel.
    Summation grouping changes across bk; bitwise agreement is not promised.
    """
    _tiles(bm, bn, bk)
    for value in (a, b):
        if not isinstance(value, torch.Tensor):
            raise ValueError("inputs must be tensors")
        if value.layout != torch.strided or value.ndim != 2:
            raise ValueError("inputs must be strided 2D tensors")
        if value.device.type != "cpu" or value.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("inputs must be CPU float32 or float64")
    if a.dtype != b.dtype or a.shape[1] != b.shape[0]:
        raise ValueError("input dtypes and contraction dimensions must match")
    m, k = a.shape
    n = b.shape[1]
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    for i in range(0, m, bm):
        for j in range(0, n, bn):
            accumulator = torch.zeros((min(bm, m - i), min(bn, n - j)), dtype=a.dtype)
            for p in range(0, k, bk):
                accumulator.add_(a[i : i + bm, p : p + bk] @ b[p : p + bk, j : j + bn])
            out[i : i + bm, j : j + bn] = accumulator
    return out


def tile_traffic(
    m: int, n: int, k: int, bm: int, bn: int, bk: int, itemsize: int
) -> dict[str, int]:
    """Valid logical loads: no inter-output-tile cache reuse or padded loads.

    Each output tile accumulates across all K, reads its A/B panels once per K
    tile, then writes C once. Footprints count valid elements, not allocation.
    """
    for name, value in (("m", m), ("n", n), ("k", k)):
        _integer(value, name, 0)
    _integer(itemsize, "itemsize", 1)
    _tiles(bm, bn, bk)
    rows, cols = (m + bm - 1) // bm, (n + bn - 1) // bn
    reads = m * k * cols + k * n * rows
    writes = m * n
    programs = rows * cols
    return {
        "flops": 2 * m * n * k,
        "read_elements": reads,
        "write_elements": writes,
        "bytes": (reads + writes) * itemsize,
        "program_count": programs,
        "k_steps": (k + bk - 1) // bk if programs else 0,
        "peak_valid_input_elements": (
            min(m, bm) * min(k, bk) + min(k, bk) * min(n, bn) if programs else 0
        ),
        "accumulator_elements": min(m, bm) * min(n, bn),
    }
