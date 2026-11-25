"""Compare teaching models with PyTorch and independent small oracles."""

import dataclasses
import hashlib
import json
import unittest

import torch
from answers import (
    DataPlan,
    ResumeToken,
    assigned_shards,
    batch_indices,
    rank_indices,
    sample_value,
)
from torch.utils.data import DataLoader, DistributedSampler


def plan(**overrides):
    config = {
        "sample_count": 40,
        "world_size": 2,
        "rank": 0,
        "batch_size": 2,
        "seed": 17,
        "epoch": 3,
        "dataset_revision": "synthetic-manifest-v1",
    }
    config.update(overrides)
    return DataPlan(**config)


class ShardingTests(unittest.TestCase):
    def test_padding_hand_example(self):
        actual = [rank_indices(10, 3, r, shuffle=False) for r in range(3)]
        self.assertEqual(actual, [[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 1]])

    def test_truncation_hand_example(self):
        actual = [
            rank_indices(10, 3, r, shuffle=False, drop_last=True) for r in range(3)
        ]
        self.assertEqual(actual, [[0, 3, 6], [1, 4, 7], [2, 5, 8]])

    def test_matches_official_sampler_grid(self):
        for n in (0, 1, 2, 3, 7, 10, 31):
            for ranks in (1, 2, 3, 5):
                for shuffle in (False, True):
                    for drop in (False, True):
                        for epoch in (0, 3):
                            for rank in range(ranks):
                                sampler = DistributedSampler(
                                    range(n),
                                    num_replicas=ranks,
                                    rank=rank,
                                    shuffle=shuffle,
                                    seed=17,
                                    drop_last=drop,
                                )
                                sampler.set_epoch(epoch)
                                with self.subTest(
                                    n=n,
                                    ranks=ranks,
                                    rank=rank,
                                    shuffle=shuffle,
                                    drop=drop,
                                    epoch=epoch,
                                ):
                                    self.assertEqual(
                                        rank_indices(
                                            n,
                                            ranks,
                                            rank,
                                            shuffle=shuffle,
                                            seed=17,
                                            epoch=epoch,
                                            drop_last=drop,
                                        ),
                                        list(sampler),
                                    )

    def test_global_rng_not_consumed(self):
        before = torch.get_rng_state().clone()
        rank_indices(30, 2, 0, seed=17, epoch=3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_epoch_is_explicit_and_repeatable(self):
        a = rank_indices(40, 2, 0, seed=17, epoch=3)
        self.assertEqual(a, rank_indices(40, 2, 0, seed=17, epoch=3))
        self.assertNotEqual(a, rank_indices(40, 2, 0, seed=17, epoch=4))

    def test_empty_and_too_few_samples(self):
        self.assertEqual(rank_indices(0, 3, 2), [])
        self.assertEqual(rank_indices(1, 4, 3), [0])
        self.assertEqual(rank_indices(1, 4, 3, drop_last=True), [])

    def test_two_drop_last_levels(self):
        for sampler_drop, loader_drop, expected in (
            (False, False, [[0, 3, 6], [9]]),
            (False, True, [[0, 3, 6]]),
            (True, False, [[0, 3, 6]]),
            (True, True, [[0, 3, 6]]),
        ):
            p = plan(
                sample_count=10,
                world_size=3,
                rank=0,
                batch_size=3,
                shuffle=False,
                sampler_drop_last=sampler_drop,
                loader_drop_last=loader_drop,
            )
            self.assertEqual(p.batches, expected)
            official = DistributedSampler(
                range(10), num_replicas=3, rank=0, shuffle=False, drop_last=sampler_drop
            )
            actual = [
                b.tolist()
                for b in DataLoader(
                    range(10),
                    sampler=official,
                    batch_size=3,
                    drop_last=loader_drop,
                    num_workers=0,
                )
            ]
            self.assertEqual(actual, expected)

    def test_batch_tail_and_no_input_mutation(self):
        ids = [4, 1, 3, 0, 2]
        self.assertEqual(batch_indices(ids, 2), [[4, 1], [3, 0], [2]])
        self.assertEqual(batch_indices(ids, 2, drop_last=True), [[4, 1], [3, 0]])
        self.assertEqual(ids, [4, 1, 3, 0, 2])

    def test_invalid_rank_arguments(self):
        for args in ((-1, 2, 0), (3, 0, 0), (3, 2, -1), (3, 2, 2)):
            with self.assertRaises(ValueError):
                rank_indices(*args)
        for args in ((True, 2, 0), (3, 2.0, 0), (3, 2, False)):
            with self.assertRaises(TypeError):
                rank_indices(*args)
        with self.assertRaises(ValueError):
            rank_indices(3, 2, 0, seed=2**63 - 1, epoch=1)
        with self.assertRaises(TypeError):
            rank_indices(3, 2, 0, shuffle=1)

    def test_shard_assignment_example(self):
        owners = [assigned_shards(6, 2, r, 2, w) for r in range(2) for w in range(2)]
        self.assertEqual(owners, [[0, 4], [1, 5], [2], [3]])

    def test_shard_coverage_and_disjointness(self):
        for n in range(12):
            for ranks in (1, 2, 4):
                for workers in (1, 2, 3):
                    groups = [
                        assigned_shards(n, ranks, r, workers, w)
                        for r in range(ranks)
                        for w in range(workers)
                    ]
                    merged = [s for group in groups for s in group]
                    self.assertEqual(sorted(merged), list(range(n)))
                    self.assertEqual(len(merged), len(set(merged)))

    def test_invalid_batch_and_shard_arguments(self):
        for size in (0, -1):
            with self.assertRaises(ValueError):
                batch_indices([0], size)
        with self.assertRaises(TypeError):
            batch_indices([0], True)
        with self.assertRaises(ValueError):
            batch_indices([-1], 1)
        for args in (
            (6, 0, 0, 2, 0),
            (6, 2, 2, 2, 0),
            (6, 2, 0, 0, 0),
            (6, 2, 0, 2, 2),
            (-1, 2, 0, 2, 0),
        ):
            with self.assertRaises(ValueError):
                assigned_shards(*args)


class ResumeTests(unittest.TestCase):
    def test_json_token_round_trip(self):
        p = plan()
        token = ResumeToken(
            **json.loads(json.dumps(dataclasses.asdict(p.checkpoint(2))))
        )
        self.assertEqual(p.resume(token), p.batches[2:])

    def test_every_commit_boundary(self):
        for drop in (False, True):
            p = plan(sample_count=19, batch_size=3, loader_drop_last=drop)
            for cursor in range(len(p.batches) + 1):
                self.assertEqual(
                    p.batches[:cursor] + p.resume(p.checkpoint(cursor)), p.batches
                )

    def test_configuration_changes_rejected(self):
        p = plan()
        changes = [
            {"sample_count": 41},
            {"world_size": 3},
            {"rank": 1},
            {"batch_size": 3},
            {"seed": 18},
            {"epoch": 4},
            {"dataset_revision": "synthetic-manifest-v2"},
            {"shuffle": False},
            {"sampler_drop_last": True},
            {"loader_drop_last": True},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan(**change).resume(p.checkpoint(2))

    def test_cursor_validation(self):
        p = plan()
        for cursor in (-1, len(p.batches) + 1):
            with self.assertRaises(ValueError):
                p.checkpoint(cursor)
        with self.assertRaises(TypeError):
            p.checkpoint(True)
        with self.assertRaises(TypeError):
            p.resume({"committed_batches": 2})
        with self.assertRaises(ValueError):
            ResumeToken("x" * 64, 2)
        with self.assertRaises(ValueError):
            p.resume(ResumeToken(p.digest, len(p.batches) + 1))

    def test_prefetch_cursor_skips_uncommitted_data(self):
        p = plan()
        committed, dispatched = 2, 6
        correct = p.resume(p.checkpoint(committed))
        wrong = p.resume(p.checkpoint(dispatched))
        self.assertEqual(correct[:4], p.batches[2:6])
        self.assertEqual(wrong, correct[4:])
        self.assertEqual(len([v for b in correct[:4] for v in b]), 8)

    def test_stateless_value_reproducible_by_sample_key(self):
        expected = (
            int.from_bytes(
                hashlib.sha256(b"augmentation-v1:911:3:7").digest()[:4], "big"
            )
            % 10000
        )
        self.assertEqual(sample_value(7, 3), expected)
        a = [sample_value(i, 3) for i in (9, 7, 1)]
        self.assertEqual(a, list(reversed([sample_value(i, 3) for i in (1, 7, 9)])))
        self.assertNotEqual(sample_value(7, 3), sample_value(7, 4))

    def test_plan_validation_and_empty_resume(self):
        for change in (
            {"dataset_revision": ""},
            {"epoch": -1},
            {"seed": -1},
            {"batch_size": 0},
        ):
            with self.assertRaises(ValueError):
                plan(**change)
        with self.assertRaises(TypeError):
            plan(loader_drop_last=1)
        p = plan(sample_count=0)
        self.assertEqual(p.resume(p.checkpoint(0)), [])

    def test_digest_is_stable_and_content_sensitive(self):
        p = plan()
        self.assertEqual(p.digest, plan().digest)
        self.assertEqual(len(p.digest), 64)
        self.assertNotEqual(p.digest, plan(dataset_revision="different").digest)


if __name__ == "__main__":
    unittest.main()
