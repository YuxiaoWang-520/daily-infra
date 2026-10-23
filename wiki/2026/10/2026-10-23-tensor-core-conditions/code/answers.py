"""CPU demonstrations of explicit storage rounding and chosen tile padding."""

import torch


def padding_plan(
    m: int, n: int, k: int, tile: tuple[int, int, int] = (16, 8, 16)
) -> dict[str, tuple[int, int, int] | int | float | None]:
    """A caller-chosen geometric tile, NOT a universal Tensor Core eligibility rule."""
    if not isinstance(tile, tuple) or len(tile) != 3:
        raise ValueError("tile must be a three-integer tuple")
    for name, value, minimum in (
        ("m", m, 0),
        ("n", n, 0),
        ("k", k, 0),
        ("tile_m", tile[0], 1),
        ("tile_n", tile[1], 1),
        ("tile_k", tile[2], 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    padded = tuple(((d + t - 1) // t) * t for d, t in zip((m, n, k), tile))
    original_work = 2 * m * n * k
    padded_work = 2 * padded[0] * padded[1] * padded[2]
    return {
        "padded_shape": padded,
        "original_flops": original_work,
        "padded_flops": padded_work,
        "work_ratio": padded_work / original_work if original_work else None,
    }


@torch.inference_mode()
def quantized_padded_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    storage_dtype: torch.dtype,
    tile: tuple[int, int, int] = (16, 8, 16),
) -> torch.Tensor:
    """Round via FP16/BF16 storage, promote, pad, then use CPU FP32 matmul.

    Inference-only, small finite inputs; NOT an emulator of Tensor Core rounding,
    instructions, tile layout, accumulators or performance. FP32 sum may overflow.
    """
    if storage_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("storage_dtype must be float16 or bfloat16")
    for value in (a, b):
        if not isinstance(value, torch.Tensor):
            raise ValueError("inputs must be tensors")
        if (
            value.ndim != 2
            or value.layout != torch.strided
            or value.device.type != "cpu"
        ):
            raise ValueError("inputs must be CPU strided 2D tensors")
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError("inputs must start as float32 or float64")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("inputs must be finite")
    if a.dtype != b.dtype or a.shape[1] != b.shape[0]:
        raise ValueError("input dtypes and contraction dimensions must match")
    m, k = a.shape
    n = b.shape[1]
    plan = padding_plan(m, n, k, tile)
    mp, np, kp = plan["padded_shape"]
    aq = a.to(storage_dtype).float()
    bq = b.to(storage_dtype).float()
    if not bool(torch.isfinite(aq).all()) or not bool(torch.isfinite(bq).all()):
        raise ValueError("storage conversion produced nonfinite inputs")
    ap = torch.zeros((mp, kp), dtype=torch.float32)
    bp = torch.zeros((kp, np), dtype=torch.float32)
    ap[:m, :k] = aq
    bp[:k, :n] = bq
    return (ap @ bp)[:m, :n].clone(memory_format=torch.contiguous_format)
