"""Real CPU DataLoader probes; run from a file for spawn-compatible workers."""

import json
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from answers import DataPlan, ResumeToken, assigned_shards, sample_value
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class SampleDataset(Dataset):
    def __init__(self, count: int, epoch: int) -> None:
        self.count = count
        self.epoch = epoch

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> list[int]:
        return [index, sample_value(index, self.epoch)]


def keep_rows(rows: list[list[int]]) -> list[list[int]]:
    """Avoid tensor IPC to isolate ordering from shared-memory transport details."""
    return rows


class CountingBatchSampler:
    def __init__(self, batches: list[list[int]]) -> None:
        self.batches = batches
        self.dispatched = 0

    def __iter__(self) -> Iterator[list[int]]:
        self.dispatched = 0
        for batch in self.batches:
            self.dispatched += 1
            yield batch.copy()

    def __len__(self) -> int:
        return len(self.batches)


class ShardedRange(IterableDataset):
    """Each integer is a synthetic whole shard; ownership is decided before yield."""

    def __iter__(self) -> Iterator[int]:
        info = get_worker_info()
        workers, worker = (1, 0) if info is None else (info.num_workers, info.id)
        yield from assigned_shards(6, 1, 0, workers, worker)


def run_runtime_probes() -> dict:
    p = DataPlan(sample_count=40, world_size=2, rank=0, batch_size=2, seed=17, epoch=3)
    dataset = SampleDataset(p.sample_count, p.epoch)
    counter = CountingBatchSampler(p.batches)
    loader = DataLoader(
        dataset,
        batch_sampler=counter,
        num_workers=2,
        prefetch_factor=2,
        multiprocessing_context="spawn",
        persistent_workers=False,
        in_order=True,
        collate_fn=keep_rows,
        timeout=30,
    )
    iterator = iter(loader)
    dispatched_at_start = counter.dispatched
    prefix = [next(iterator), next(iterator)]
    dispatched_at_checkpoint = counter.dispatched
    token = p.checkpoint(len(prefix))
    # Drain the original iterator as an uninterrupted oracle and for public-API cleanup.
    uninterrupted_suffix = list(iterator)
    with tempfile.TemporaryDirectory(prefix="data-plan-") as tmp:
        checkpoint = Path(tmp) / "data-progress.json"
        checkpoint.write_text(json.dumps(asdict(token)), encoding="utf-8")
        restored = ResumeToken(**json.loads(checkpoint.read_text(encoding="utf-8")))
        # A different worker count is safe here because sample_value is stateless.
        resumed = list(
            DataLoader(
                dataset,
                batch_sampler=p.resume(restored),
                num_workers=0,
                collate_fn=keep_rows,
            )
        )
    expected = [[[i, sample_value(i, p.epoch)] for i in b] for b in p.batches]
    if prefix + uninterrupted_suffix != expected or resumed != uninterrupted_suffix:
        raise RuntimeError("sample ID or stateless value replay mismatch")
    if dispatched_at_start != 4 or dispatched_at_checkpoint != 6:
        raise RuntimeError("checked DataLoader prefetch behavior changed")
    wrong = p.resume(p.checkpoint(dispatched_at_checkpoint))
    skipped_ids = [
        i for b in p.batches[len(prefix) : dispatched_at_checkpoint] for i in b
    ]

    iterable_multi = list(
        DataLoader(
            ShardedRange(),
            batch_size=4,
            drop_last=True,
            num_workers=2,
            multiprocessing_context="spawn",
            timeout=30,
        )
    )
    iterable_single = [
        b.tolist()
        for b in DataLoader(ShardedRange(), batch_size=4, drop_last=True, num_workers=0)
    ]
    if iterable_multi != [] or iterable_single != [[0, 1, 2, 3]]:
        raise RuntimeError("per-worker IterableDataset tail behavior changed")
    return {
        "kind": "real CPU DataLoader functional probe",
        "rank": 0,
        "world_size_in_plan": 2,
        "distributed_process_group": False,
        "worker_start_method": "spawn",
        "num_workers_before": 2,
        "num_workers_resume": 0,
        "prefetch_factor": 2,
        "in_order": True,
        "total_batches": len(p.batches),
        "dispatched_at_start": dispatched_at_start,
        "dispatched_at_checkpoint": dispatched_at_checkpoint,
        "committed_batches": len(prefix),
        "token": asdict(token),
        "prefix": prefix,
        "uninterrupted_suffix": uninterrupted_suffix,
        "resumed_suffix": resumed,
        "exact_rows_match": resumed == uninterrupted_suffix,
        "wrong_resume_batch_count": len(wrong),
        "wrong_resume_skipped_ids": skipped_ids,
        "iterable_worker_shards": [[0, 2, 4], [1, 3, 5]],
        "iterable_batch_size": 4,
        "iterable_drop_last": True,
        "iterable_two_worker_batches": iterable_multi,
        "iterable_zero_worker_batches": iterable_single,
        "limitations": [
            "No distributed training or GPU execution",
            "No throughput benchmark",
            "Immutable synthetic data; fixed rank and world size",
            "No worker RNG or open streaming connection recovery",
            "Original loader drained after checkpoint; no process-crash injection",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_runtime_probes(), indent=2))
