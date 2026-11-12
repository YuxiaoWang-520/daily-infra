"""CPU teaching exercise for retained storage and borrowed outputs."""

import torch


class StaticAffine:
    """Keep input/output storage for forward-only y = scale*x + bias."""

    def __init__(self, example: torch.Tensor, scale: float, bias: float) -> None:
        raise NotImplementedError("Allocate independent, retained CPU buffers")

    @property
    def input_buffer(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_buffer(self) -> torch.Tensor:
        raise NotImplementedError

    def copy_input(self, source: torch.Tensor) -> None:
        raise NotImplementedError

    def replay(self) -> torch.Tensor:
        raise NotImplementedError
