"""CPU teaching answers: broadcast geometry and inference-only tiled distances."""

from collections.abc import Sequence

import torch


def broadcast_shape(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """Return the concrete broadcast shape, including zero-size dimensions."""
    left, right = tuple(left), tuple(right)
    if any(type(size) is not int or size < 0 for size in (*left, *right)):
        raise ValueError("shape dimensions must be non-negative integers")
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + left
    right = (1,) * (rank - len(right)) + right
    result = []
    for a, b in zip(left, right):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"incompatible dimensions: {a} and {b}")
    return tuple(result)


@torch.inference_mode()
def pairwise_sqdist(
    x: torch.Tensor,
    y: torch.Tensor,
    block_rows: int = 64,
    block_cols: int = 64,
) -> torch.Tensor:
    """Return squared L2 distances with each difference tile <= rows*cols*D.

    Inference only. Inputs are ordinary dense 2D float32/float64 tensors on
    the same device with equal feature width and dtype. Output still uses N*M
    elements. A reduced tile also uses up to rows*cols elements, even for D=0.
    This is an educational decomposition, not a tuned GPU kernel.
    """
    if any(type(size) is not int or size <= 0 for size in (block_rows, block_cols)):
        raise ValueError("tile dimensions must be positive integers")
    if x.layout != torch.strided or y.layout != torch.strided:
        raise ValueError("inputs must have strided layout")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("inputs must be [N,D] and [M,D] with equal D")
    if x.dtype not in (torch.float32, torch.float64) or y.dtype != x.dtype:
        raise ValueError("inputs must have the same float32/float64 dtype")
    if x.device != y.device:
        raise ValueError("inputs must be on the same device")
    n = x.shape[0]
    m = y.shape[0]
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    for row in range(0, n, block_rows):
        for col in range(0, m, block_cols):
            diff = (
                x[row : row + block_rows, None, :] - y[None, col : col + block_cols, :]
            )
            # Safe: diff is fresh and the whole function is inference-only.
            diff.square_()
            out[row : row + block_rows, col : col + block_cols] = diff.sum(dim=-1)
            # Drop the Python reference before allocating the next diff tile.
            del diff
    return out
