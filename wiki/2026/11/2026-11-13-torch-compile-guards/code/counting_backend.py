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
        nodes = tuple(graph_module.graph.nodes)
        self.records.append(
            GraphRecord(
                code=graph_module.code,
                operators=tuple(
                    str(node.target)
                    for node in nodes
                    if node.op in ("call_function", "call_method", "call_module")
                ),
                placeholders=tuple(
                    str(node.target) for node in nodes if node.op == "placeholder"
                ),
                input_kinds=tuple(type(value).__name__ for value in example_inputs),
            )
        )
        return graph_module.forward
