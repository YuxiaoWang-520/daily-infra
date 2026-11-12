"""CPU teaching exercise for retained storage and borrowed outputs."""

import math

import torch


class StaticAffine:
    """Keep input/output storage for forward-only y = scale*x + bias."""

    def __init__(self, example: torch.Tensor, scale: float, bias: float) -> None:
        _validate_example(example)
        self._scale = _scalar(scale)
        self._bias = _scalar(bias)
        self._input = example.detach().contiguous().clone()
        self._output = torch.empty_like(self._input)

    @property
    def input_buffer(self) -> torch.Tensor:
        return self._input

    @property
    def output_buffer(self) -> torch.Tensor:
        return self._output

    def copy_input(self, source: torch.Tensor) -> None:
        _validate_example(source)
        if source.shape != self._input.shape or source.dtype != self._input.dtype:
            raise ValueError("source shape and dtype must match the retained input")
        if (
            source.untyped_storage().data_ptr()
            == self._input.untyped_storage().data_ptr()
        ):
            source = source.detach().clone()
        with torch.no_grad():
            self._input.copy_(source)

    def replay(self) -> torch.Tensor:
        with torch.no_grad():
            torch.mul(self._input, self._scale, out=self._output)
            self._output.add_(self._bias)
        return self._output


def _validate_example(example: torch.Tensor) -> None:
    if not isinstance(example, torch.Tensor):
        raise ValueError("input must be a Tensor")  # noqa: TRY004 - exercise contract
    if example.device.type != "cpu" or example.layout != torch.strided:
        raise ValueError("input must be a dense, strided CPU tensor")
    if example.ndim != 2 or example.numel() == 0:
        raise ValueError("input must be a nonempty matrix")
    if example.dtype not in (torch.float32, torch.float64):
        raise ValueError("input dtype must be float32 or float64")


def _scalar(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("scale and bias must be finite numbers")  # noqa: TRY004 - exercise contract
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("scalar is outside the finite float range") from error
    if not math.isfinite(result):
        raise ValueError("scale and bias must be finite numbers")
    return result
