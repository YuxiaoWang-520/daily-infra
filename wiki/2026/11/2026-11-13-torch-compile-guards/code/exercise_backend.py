"""Observe FX graph delivery without running an optimizing code generator."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GraphRecord:
    code: str
    operators: tuple[str, ...]
    placeholders: tuple[str, ...]
    input_kinds: tuple[str, ...]


class CountingBackend:
    """Record one entry per backend invocation and return equivalent FX execution."""

    def __init__(self) -> None:
        self.records: list[GraphRecord] = []

    def __call__(
        self, graph_module: torch.fx.GraphModule, example_inputs: list[Any]
    ) -> Callable[..., Any]:
        raise NotImplementedError(
            "Record graph structure and return its forward callable"
        )
