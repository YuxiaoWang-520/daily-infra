"""CPU teaching models; this is not a production DataLoader checkpoint API."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int (not bool)")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _boolean(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")


def _sampling_arguments(
    sample_count: int,
    world_size: int,
    rank: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    drop_last: bool,
) -> None:
    for name, value, lower in (
        ("sample_count", sample_count, 0),
        ("world_size", world_size, 1),
        ("rank", rank, 0),
        ("seed", seed, 0),
        ("epoch", epoch, 0),
    ):
        _integer(name, value, lower)
    if rank >= world_size:
        raise ValueError("rank must be < world_size")
    if seed + epoch >= 2**63:
        raise ValueError("teaching contract requires seed + epoch < 2**63")
    _boolean("shuffle", shuffle)
    _boolean("drop_last", drop_last)


def rank_indices(
    sample_count: int,
    world_size: int,
    rank: int,
    *,
    seed: int = 0,
    epoch: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> list[int]:
    """Reproduce the checked DistributedSampler order for a fixed-size dataset."""
    _sampling_arguments(sample_count, world_size, rank, seed, epoch, shuffle, drop_last)
    if sample_count == 0:
        return []
    if shuffle:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(sample_count, generator=generator).tolist()
    else:
        order = list(range(sample_count))
    per_rank = (
        sample_count // world_size
        if drop_last
        else (sample_count + world_size - 1) // world_size
    )
    total = per_rank * world_size
    if drop_last:
        order = order[:total]
    else:
        original = order.copy()
        order.extend(original[i % sample_count] for i in range(total - sample_count))
    return order[rank:total:world_size]


def batch_indices(
    indices: Sequence[int],
    batch_size: int,
    *,
    drop_last: bool = False,
) -> list[list[int]]:
    """Batch one rank's ordered IDs, preserving the distinction between two tails."""
    _integer("batch_size", batch_size, 1)
    _boolean("drop_last", drop_last)
    for index in indices:
        _integer("sample index", index)
    end = len(indices)
    if drop_last:
        end -= end % batch_size
    return [
        list(indices[i : min(i + batch_size, end)]) for i in range(0, end, batch_size)
    ]


def assigned_shards(
    shard_count: int,
    world_size: int,
    rank: int,
    workers: int,
    worker: int,
) -> list[int]:
    """Assign whole shards before reading; assume equal worker counts on all ranks."""
    for name, value, lower in (
        ("shard_count", shard_count, 0),
        ("world_size", world_size, 1),
        ("rank", rank, 0),
        ("workers", workers, 1),
        ("worker", worker, 0),
    ):
        _integer(name, value, lower)
    if rank >= world_size or worker >= workers:
        raise ValueError("rank or worker outside its group")
    return list(range(rank * workers + worker, shard_count, world_size * workers))


def sample_value(sample_id: int, epoch: int, seed: int = 911) -> int:
    """Stateless synthetic augmentation; no image transform or worker RNG is modeled."""
    for name, value in (("sample_id", sample_id), ("epoch", epoch), ("seed", seed)):
        _integer(name, value)
    key = f"augmentation-v1:{seed}:{epoch}:{sample_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 10000


@dataclass(frozen=True)
class ResumeToken:
    """The number of completed batches belonging to the checkpoint's training state."""

    plan_digest: str
    committed_batches: int

    def __post_init__(self) -> None:
        if not isinstance(self.plan_digest, str):
            raise TypeError("plan_digest must be str")
        if len(self.plan_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.plan_digest
        ):
            raise ValueError("plan_digest must be a lowercase SHA-256 hex string")
        _integer("committed_batches", self.committed_batches)


@dataclass(frozen=True)
class DataPlan:
    """O(N) replay plan for immutable indexed data and fixed training topology."""

    sample_count: int
    world_size: int
    rank: int
    batch_size: int
    seed: int = 0
    epoch: int = 0
    shuffle: bool = True
    sampler_drop_last: bool = False
    loader_drop_last: bool = False
    dataset_revision: str = "synthetic-manifest-v1"

    def __post_init__(self) -> None:
        _sampling_arguments(
            self.sample_count,
            self.world_size,
            self.rank,
            self.seed,
            self.epoch,
            self.shuffle,
            self.sampler_drop_last,
        )
        _integer("batch_size", self.batch_size, 1)
        _boolean("loader_drop_last", self.loader_drop_last)
        if not isinstance(self.dataset_revision, str):
            raise TypeError("dataset_revision must be str")
        if not self.dataset_revision.strip():
            raise ValueError("dataset_revision must not be blank")

    @property
    def batches(self) -> list[list[int]]:
        ids = rank_indices(
            self.sample_count,
            self.world_size,
            self.rank,
            seed=self.seed,
            epoch=self.epoch,
            shuffle=self.shuffle,
            drop_last=self.sampler_drop_last,
        )
        return batch_indices(ids, self.batch_size, drop_last=self.loader_drop_last)

    @property
    def digest(self) -> str:
        # Include actual IDs too: a changed RNG implementation must not silently resume.
        value = {"schema": 1, "config": asdict(self), "batches": self.batches}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def checkpoint(self, committed_batches: int) -> ResumeToken:
        _integer("committed_batches", committed_batches)
        if committed_batches > len(self.batches):
            raise ValueError("cursor exceeds this rank's epoch length")
        return ResumeToken(self.digest, committed_batches)

    def resume(self, token: ResumeToken) -> list[list[int]]:
        if not isinstance(token, ResumeToken):
            raise TypeError("token must be ResumeToken")
        if token.plan_digest != self.digest:
            raise ValueError("dataset, configuration, or concrete sample order changed")
        batches = self.batches
        if token.committed_batches > len(batches):
            raise ValueError("cursor exceeds this rank's epoch length")
        return batches[token.committed_batches :]
