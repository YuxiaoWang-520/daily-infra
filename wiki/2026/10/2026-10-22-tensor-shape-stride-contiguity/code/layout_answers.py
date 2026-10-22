"""A small address model for nonnegative-stride, concrete tensor metadata.

This is a teaching model, not a replacement for PyTorch storage or view logic.
All offsets and strides are in elements. Negative indices are out of scope.
"""

from collections.abc import Sequence
from itertools import product
from typing import TypeVar

T = TypeVar("T")


def _validate_layout(shape: Sequence[int], strides: Sequence[int]) -> None:
    if len(shape) != len(strides):
        raise ValueError("shape and strides must have the same rank")
    for name, values in (("shape", shape), ("strides", strides)):
        if any(type(value) is not int for value in values):
            raise TypeError(f"{name} entries must be integers")
        if any(value < 0 for value in values):
            raise ValueError(f"{name} entries must be nonnegative")


def storage_index(
    shape: Sequence[int],
    strides: Sequence[int],
    index: Sequence[int],
    offset: int = 0,
) -> int:
    """Return the element index from the start of storage; do not read it."""
    _validate_layout(shape, strides)
    if type(offset) is not int:
        raise TypeError("offset must be an integer")
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    if len(index) != len(shape):
        raise ValueError("index must match the tensor rank")
    if any(type(value) is not int for value in index):
        raise TypeError("indices must be integers")
    if any(i < 0 or i >= size for i, size in zip(index, shape)):
        raise IndexError("index is outside the logical shape")
    return offset + sum(i * stride for i, stride in zip(index, strides))


def is_c_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Check default row-major contiguity, including empty and size-one axes."""
    _validate_layout(shape, strides)
    if 0 in shape:
        return True
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


def pack_logical(
    storage: Sequence[T],
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int = 0,
) -> list[T]:
    """Materialize logical row-major iteration into a fresh Python list.

    A scalar reads one element; an empty tensor reads none. This deliberately
    uses an explicit logical-index loop, not PyTorch's optimized copy kernel.
    """
    _validate_layout(shape, strides)
    if type(offset) is not int:
        raise TypeError("offset must be an integer")
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    if 0 in shape:
        return []
    result = []
    for index in product(*(range(size) for size in shape)):
        position = storage_index(shape, strides, index, offset)
        if position >= len(storage):
            raise IndexError("layout addresses data outside the given storage")
        result.append(storage[position])
    return result
