"""Data correctness and observation scope, without timing speed assertions."""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

impl = importlib.import_module(os.environ.get("PROBE_IMPL", "loader_probe"))


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        options = {
            "sample_count": 10,
            "batch_size": 4,
            "source_pause_ms": 0,
            "consumer_pause_ms": 0,
            "warmup_batches": 0,
        }
        cls.single = impl.run_trial(0, **options)
        cls.parallel = impl.run_trial(2, **options)

    def test_every_id_arrives_once_and_in_order(self):
        for report in (self.single, self.parallel):
            self.assertEqual(report.sample_ids, tuple(range(10)))
            self.assertEqual(len(set(report.sample_ids)), 10)

    def test_payload_checksums_match_hand_calculation(self):
        for report in (self.single, self.parallel):
            self.assertEqual(report.checksums[0], 14)
            self.assertEqual(sum(report.checksums), 1820)
            self.assertEqual(report.checksums, self.single.checksums)

    def test_tail_batch_is_retained(self):
        self.assertEqual([batch.samples for batch in self.parallel.batches], [4, 4, 2])

    def test_single_process_work_happens_in_the_main_process(self):
        worker_ids = {
            worker for batch in self.single.batches for worker in batch.worker_ids
        }
        self.assertEqual(worker_ids, {-1})

    def test_spawn_workers_actually_fetch_records(self):
        worker_ids = {
            worker for batch in self.parallel.batches for worker in batch.worker_ids
        }
        self.assertEqual(worker_ids, {0, 1})

    def test_phase_measurements_are_nonnegative(self):
        for report in (self.single, self.parallel):
            self.assertGreaterEqual(report.iterator_creation_ms, 0)
            self.assertGreaterEqual(report.total_iteration_ms, 0)
            self.assertGreaterEqual(report.steady_next_wait_median_ms, 0)
            self.assertGreater(report.fixture_bytes, 0)
            for batch in report.batches:
                for duration in (
                    batch.next_wait_ms,
                    batch.consumer_ms,
                    batch.source_wall_ms,
                    batch.decode_ms,
                    batch.transform_ms,
                ):
                    self.assertGreaterEqual(duration, 0)


class BoundaryTests(unittest.TestCase):
    def test_one_sample_and_warmup_exclusion(self):
        report = impl.run_trial(
            0,
            sample_count=1,
            batch_size=4,
            source_pause_ms=0,
            consumer_pause_ms=0,
            warmup_batches=0,
        )
        self.assertEqual(report.sample_ids, (0,))
        self.assertEqual(report.checksums, (14,))
        self.assertEqual(len(report.batches), 1)

    def test_invalid_configuration_fails_before_iteration(self):
        for changes in (
            {"workers": -1},
            {"workers": True},
            {"prefetch_factor": 0},
            {"sample_count": 0},
            {"batch_size": 0},
            {"source_pause_ms": -1},
            {"source_pause_ms": 1001},
            {"consumer_pause_ms": float("nan")},
            {"consumer_pause_ms": True},
            {"warmup_batches": 10},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                impl.run_trial(**{"workers": 0, **changes})

    def test_global_cpu_rng_is_preserved(self):
        before = torch.get_rng_state().clone()
        impl.run_trial(
            0, sample_count=1, source_pause_ms=0, consumer_pause_ms=0, warmup_batches=0
        )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_iteration_failure_closes_workers_with_iterator_still_referenced(self):
        iterators = []

        def fail_next(iterator):
            iterators.append(iterator)
            raise RuntimeError("injected next failure")

        with patch(
            "torch.utils.data.dataloader._MultiProcessingDataLoaderIter._next_data",
            fail_next,
        ):
            try:
                impl.run_trial(
                    2,
                    sample_count=4,
                    batch_size=2,
                    source_pause_ms=0,
                    consumer_pause_ms=0,
                    warmup_batches=0,
                )
            except RuntimeError as error:
                retained_error = error
            else:
                self.fail("the injected failure must propagate")
        iterator = iterators[0]
        try:
            self.assertIsNotNone(retained_error.__traceback__)
            for worker in iterator._workers:
                self.assertFalse(worker.is_alive())
        finally:
            iterator._shutdown_workers()

    def test_dataset_rejects_a_mismatched_record_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00000.json").write_text(
                json.dumps({"id": 1, "values": [0, 1, 2, 3]})
            )
            dataset = impl.JsonDataset(root, 1, 0)
            with self.assertRaises(ValueError):
                dataset[0]

    def test_dataset_rejects_malformed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00000.json").write_text(
                json.dumps({"id": 0, "values": [True, 1, 2, 3]})
            )
            dataset = impl.JsonDataset(root, 1, 0)
            with self.assertRaises(ValueError):
                dataset[0]

    def test_dataset_index_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = impl.JsonDataset(Path(directory), 1, 0)
            for index in (-1, 1, True, 0.5):
                with self.subTest(index=index), self.assertRaises(IndexError):
                    dataset[index]


if __name__ == "__main__":
    unittest.main()
