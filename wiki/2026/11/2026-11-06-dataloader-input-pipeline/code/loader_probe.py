"""Synthetic JSON input fixture with real CPU DataLoader and spawn workers."""

import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import torch
from models import BatchObservation, TrialReport
from torch.utils.data import DataLoader, Dataset, get_worker_info


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _pause(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1000
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and in [0, 1000] milliseconds")


class JsonDataset(Dataset):
    """Read known synthetic records; worker classes stay importable for spawn."""

    def __init__(self, root: Path, count: int, source_pause_ms: float) -> None:
        _integer(count, "count")
        _pause(source_pause_ms, "source_pause_ms")
        self.root = Path(root)
        self.count = count
        self.source_pause_ms = source_pause_ms

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, int | float]:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.count
        ):
            raise IndexError("fixture index is out of range")
        started = time.perf_counter_ns()
        time.sleep(self.source_pause_ms / 1000)
        data = (self.root / f"{index:05d}.json").read_bytes()
        read_done = time.perf_counter_ns()
        record = json.loads(data)
        if (
            not isinstance(record, dict)
            or type(record.get("id")) is not int
            or record["id"] != index
            or not isinstance(record.get("values"), list)
            or any(type(value) is not int for value in record["values"])
            or record["values"] != list(range(index, index + 4))
        ):
            raise ValueError(f"invalid synthetic record at index {index}")
        decode_done = time.perf_counter_ns()
        checksum = float(
            torch.tensor(record["values"], dtype=torch.float64).square().sum()
        )
        transform_done = time.perf_counter_ns()
        info = get_worker_info()
        return {
            "id": index,
            "checksum": checksum,
            "source_wall_ms": (read_done - started) / 1e6,
            "decode_ms": (decode_done - read_done) / 1e6,
            "transform_ms": (transform_done - decode_done) / 1e6,
            "worker_id": info.id if info else -1,
        }


def run_trial(
    workers: int,
    prefetch_factor: int = 2,
    sample_count: int = 40,
    batch_size: int = 4,
    source_pause_ms: float = 5,
    consumer_pause_ms: float = 20,
    warmup_batches: int = 2,
) -> TrialReport:
    """Measure one CPU iteration, keeping startup and steady waits separate.

    Files belong to an exclusive TemporaryDirectory and are opened in 'x' mode.
    The source phase includes an artificial sleep and tiny cached-file reads.
    The consumer performs a checksum reduction and an artificial sleep.
    Timings therefore describe this fixture, not device or storage bandwidth.
    """
    for name, value, minimum in (
        ("workers", workers, 0),
        ("prefetch_factor", prefetch_factor, 1),
        ("sample_count", sample_count, 1),
        ("batch_size", batch_size, 1),
        ("warmup_batches", warmup_batches, 0),
    ):
        _integer(value, name, minimum)
    _pause(source_pause_ms, "source_pause_ms")
    _pause(consumer_pause_ms, "consumer_pause_ms")
    batch_count = (sample_count + batch_size - 1) // batch_size
    if warmup_batches >= batch_count:
        raise ValueError("warmup_batches must leave at least one measured batch")

    with tempfile.TemporaryDirectory(prefix="daily-infra-loader-") as directory:
        root = Path(directory)
        for index in range(sample_count):
            with (root / f"{index:05d}.json").open("x", encoding="utf-8") as stream:
                json.dump(
                    {"id": index, "values": list(range(index, index + 4))}, stream
                )
        fixture_bytes = sum(path.stat().st_size for path in root.glob("*.json"))
        options = {}
        if workers:
            options = {
                "prefetch_factor": prefetch_factor,
                "multiprocessing_context": "spawn",
                "timeout": 30,
            }
        loader = DataLoader(
            JsonDataset(root, sample_count, source_pause_ms),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=False,
            in_order=True,
            generator=torch.Generator(device="cpu").manual_seed(2026),
            **options,
        )
        ids: list[int] = []
        checksums: list[float] = []
        observations: list[BatchObservation] = []
        started = time.perf_counter_ns()
        iterator = iter(loader)
        created = time.perf_counter_ns()
        try:
            while True:
                before_next = time.perf_counter_ns()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                after_next = time.perf_counter_ns()
                float(batch["checksum"].sum())
                time.sleep(consumer_pause_ms / 1000)
                after_consume = time.perf_counter_ns()
                batch_ids = batch["id"].tolist()
                ids.extend(batch_ids)
                checksums.extend(batch["checksum"].tolist())
                observations.append(
                    BatchObservation(
                        index=len(observations),
                        samples=len(batch_ids),
                        next_wait_ms=(after_next - before_next) / 1e6,
                        consumer_ms=(after_consume - after_next) / 1e6,
                        source_wall_ms=float(batch["source_wall_ms"].sum()),
                        decode_ms=float(batch["decode_ms"].sum()),
                        transform_ms=float(batch["transform_ms"].sum()),
                        worker_ids=tuple(batch["worker_id"].tolist()),
                    )
                )
        finally:
            # This lab pins PyTorch 2.15.0 and uses its idempotent private cleanup.
            # Tracebacks can keep an iterator alive after next() raises.
            if workers:
                iterator._shutdown_workers()
            del iterator
        finished = time.perf_counter_ns()
        return TrialReport(
            workers=workers,
            sample_ids=tuple(ids),
            checksums=tuple(checksums),
            fixture_bytes=fixture_bytes,
            iterator_creation_ms=(created - started) / 1e6,
            total_iteration_ms=(finished - started) / 1e6,
            warmup_batches=warmup_batches,
            steady_next_wait_median_ms=statistics.median(
                observation.next_wait_ms
                for observation in observations[warmup_batches:]
            ),
            batches=tuple(observations),
        )
