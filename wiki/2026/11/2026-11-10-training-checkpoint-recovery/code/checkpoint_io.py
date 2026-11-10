"""Single-writer, same-directory publication of an already frozen payload.

Only caller-owned paths are accepted by this teaching interface. A completed
os.replace gives atomic visibility on the same filesystem. Directory fsync,
power-loss durability, distributed coordination and concurrent writers are
outside this exercise.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    """Publish a snapshot; keep the previous file if pre-publication I/O fails."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read tensors and primitive state from a checkpoint owned by the caller."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a dictionary")
    return payload
