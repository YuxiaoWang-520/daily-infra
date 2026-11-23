"""Learner starting points. See code/README.md for contracts and acceptance criteria."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def overlap_summary(
    compute: Iterable[tuple[float, float]],
    communication: Iterable[tuple[float, float]],
    window: tuple[float, float],
) -> dict[str, float]:
    """Exercise 1: clip, merge, intersect; distinguish summed and union time."""
    raise NotImplementedError("Implement interval accounting without double counting")


def profile_training(output_dir: Path) -> dict[str, Any]:
    """Exercise 2: annotate and profile steps 2/3, verify baseline equality."""
    raise NotImplementedError("Use the documented CPU-only schedule and trace contract")
