"""Reference answers for head-layout transforms using real PyTorch tensors."""

import torch


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """Map [B,S,H*D] to [B,H,S,D] without copying under a strict input contract.

    Accept nonempty, unnamed, nonquantized, default-contiguous strided tensors.
    A production caller should specify its own supported layout contract.
    """
    if type(heads) is not int or heads <= 0:
        raise ValueError("heads must be a positive integer")
    if x.layout != torch.strided or x.ndim != 3:
        raise ValueError("expected a strided tensor with shape [B,S,M]")
    if x.is_quantized or any(name is not None for name in x.names):
        raise ValueError("this exercise requires unnamed, nonquantized tensors")
    if any(size == 0 for size in x.shape):
        raise ValueError("this exercise requires positive dimensions")
    if not x.is_contiguous():
        raise ValueError("the no-copy contract requires default-contiguous x")
    batch, sequence, width = x.shape
    if width % heads:
        raise ValueError("M must be divisible by heads")
    return x.view(batch, sequence, heads, width // heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Map [B,H,S,D] to [B,S,H*D], copying if the input strides require it.

    Inputs must be ordinary unnamed, nonquantized tensors. Correct element order
    is guaranteed; independent storage is not promised.
    """
    if x.layout != torch.strided or x.ndim != 4:
        raise ValueError("expected a strided tensor with shape [B,H,S,D]")
    if x.is_quantized or any(name is not None for name in x.names):
        raise ValueError("this exercise requires unnamed, nonquantized tensors")
    if any(size == 0 for size in x.shape):
        raise ValueError("this exercise requires positive dimensions")
    batch, heads, sequence, width = x.shape
    return x.transpose(1, 2).reshape(batch, sequence, heads * width)
