"""Behavioral checks for page accounting, address translation, and cache reuse."""

import random
import unittest

import torch
from answers import PagedKVCache, page_usage, slot_for_token
from torch.nn import functional as F


def rows(count: int, start: int = 0) -> torch.Tensor:
    return torch.arange(
        start, start + 2 * count, dtype=torch.float64, device="cpu"
    ).reshape(count, 2)


class AddressTests(unittest.TestCase):
    def test_noncontiguous_table(self) -> None:
        slots = [
            slot_for_token((3, 5), i, block_size=4, length=7, num_blocks=8)
            for i in range(7)
        ]
        self.assertEqual(slots, [12, 13, 14, 15, 20, 21, 22])

    def test_unfilled_tail_is_not_readable(self) -> None:
        with self.assertRaises(IndexError):
            slot_for_token((3, 5), 7, block_size=4, length=7, num_blocks=8)

    def test_page_boundary(self) -> None:
        self.assertEqual(
            slot_for_token((5, 2), 4, block_size=4, length=5, num_blocks=6), 8
        )

    def test_unused_table_entries_are_not_dereferenced(self) -> None:
        self.assertEqual(
            slot_for_token((3, -1), 0, block_size=4, length=1, num_blocks=8), 12
        )

    def test_bad_addresses(self) -> None:
        for table in ((), (-1,), (8,), (True,)):
            with self.subTest(table=table), self.assertRaises(ValueError):
                slot_for_token(table, 0, block_size=4, length=1, num_blocks=8)
        for position in (-1, True, 0.5):
            with self.subTest(position=position), self.assertRaises(ValueError):
                slot_for_token((1,), position, block_size=4, length=1, num_blocks=8)

    def test_invalid_dimensions(self) -> None:
        for field in ("block_size", "num_blocks"):
            for value in (0, -1, True, 1.5):
                kwargs = dict(block_size=4, length=1, num_blocks=8)
                kwargs[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    slot_for_token((1,), 0, **kwargs)
        with self.assertRaises(ValueError):
            slot_for_token((1,), 0, block_size=4, length=-1, num_blocks=8)

    def test_empty_history(self) -> None:
        with self.assertRaises(IndexError):
            slot_for_token((), 0, block_size=4, length=0, num_blocks=8)


class AccountingTests(unittest.TestCase):
    def test_unequal_lengths(self) -> None:
        result = page_usage([5, 3], block_size=4, bytes_per_token=32)
        self.assertEqual(
            result,
            dict(
                tokens=8,
                blocks=3,
                unused_slots=4,
                content_bytes=256,
                allocated_bytes=384,
            ),
        )

    def test_zero_exact_and_empty(self) -> None:
        self.assertEqual(
            page_usage([0, 4, 8], block_size=4, bytes_per_token=2)["unused_slots"], 0
        )
        self.assertEqual(page_usage([], block_size=4, bytes_per_token=2)["blocks"], 0)

    def test_tail_bound_for_many_lengths(self) -> None:
        for size in (1, 3, 4, 16):
            for length in range(40):
                result = page_usage([length], block_size=size, bytes_per_token=7)
                self.assertEqual(
                    result["allocated_bytes"],
                    result["content_bytes"] + 7 * result["unused_slots"],
                )
                self.assertTrue(0 <= result["unused_slots"] < size)

    def test_invalid_accounting_inputs(self) -> None:
        for lengths in ([-1], [True], [1.5]):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                page_usage(lengths, block_size=4, bytes_per_token=2)
        for field in ("block_size", "bytes_per_token"):
            for value in (0, -1, True, 1.5):
                kwargs = dict(block_size=4, bytes_per_token=2)
                kwargs[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    page_usage([1], **kwargs)


class CacheTests(unittest.TestCase):
    def make_cache(self, blocks: int = 4, size: int = 4) -> PagedKVCache:
        return PagedKVCache(num_blocks=blocks, block_size=size, head_dim=2)

    def test_append_across_pages_and_keep_other_request(self) -> None:
        cache = self.make_cache()
        ka, va, kb, vb = rows(7), rows(7, 100), rows(5, 200), rows(5, 300)
        cache.append("a", ka[:3], va[:3])
        cache.append("b", kb, vb)
        cache.append("a", ka[3:], va[3:])
        self.assertEqual(cache.block_table("a"), (0, 3))
        self.assertEqual(cache.block_table("b"), (1, 2))
        self.assertEqual(cache.length("a"), 7)
        self.assertEqual(cache.free_block_count, 0)
        for actual, expected in zip(cache.read("a"), (ka, va)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(cache.read("b"), (kb, vb)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_append_inside_last_page_needs_no_free_block(self) -> None:
        cache = self.make_cache(blocks=1)
        cache.append("a", rows(3), rows(3, 20))
        cache.append("a", rows(1, 6), rows(1, 26))
        self.assertEqual(cache.length("a"), 4)
        self.assertEqual(cache.block_table("a"), (0,))

    def test_out_of_memory_does_not_partly_append(self) -> None:
        cache = self.make_cache(blocks=1)
        cache.append("a", rows(3), rows(3, 20))
        before = cache.read("a")
        with self.assertRaises(MemoryError):
            cache.append("a", rows(2, 6), rows(2, 26))
        self.assertEqual(cache.length("a"), 3)
        self.assertEqual(cache.block_table("a"), (0,))
        self.assertEqual(cache.free_block_count, 0)
        for actual, expected in zip(cache.read("a"), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(MemoryError):
            cache.append("new", rows(1), rows(1))
        with self.assertRaises(KeyError):
            cache.block_table("new")

    def test_release_reuse_does_not_read_stale_tail(self) -> None:
        cache = self.make_cache(blocks=1)
        cache.append("old", rows(4, 100), rows(4, 200))
        cache.release("old")
        self.assertEqual(cache.free_block_count, 1)
        cache.append("new", rows(1), rows(1, 20))
        k, v = cache.read("new")
        self.assertEqual(k.shape, (1, 2))
        torch.testing.assert_close(k, rows(1), rtol=0, atol=0)
        torch.testing.assert_close(v, rows(1, 20), rtol=0, atol=0)
        with self.assertRaises(KeyError):
            cache.read("old")
        with self.assertRaises(KeyError):
            cache.release("old")
        self.assertEqual(cache.free_block_count, 0)

    def test_fifo_keeps_existing_free_pages_before_released_pages(self) -> None:
        cache = self.make_cache(blocks=4)
        cache.append("old", rows(5), rows(5))
        self.assertEqual(cache.block_table("old"), (0, 1))
        cache.release("old")
        cache.append("new", rows(9), rows(9))
        self.assertEqual(cache.block_table("new"), (2, 3, 0))
        self.assertEqual(cache.free_block_count, 1)

    def test_input_and_read_result_do_not_alias_cache(self) -> None:
        cache = self.make_cache()
        k, v = rows(2), rows(2, 20)
        cache.append("a", k, v)
        k.fill_(999)
        v.fill_(999)
        read_k, read_v = cache.read("a")
        read_k.fill_(-1)
        read_v.fill_(-1)
        actual_k, actual_v = cache.read("a")
        torch.testing.assert_close(actual_k, rows(2), rtol=0, atol=0)
        torch.testing.assert_close(actual_v, rows(2, 20), rtol=0, atol=0)
        self.assertIsInstance(cache.block_table("a"), tuple)

    def test_paged_gather_matches_independent_sdpa(self) -> None:
        cache = self.make_cache()
        k, v = rows(7) / 9, rows(7, 20) / 11
        cache.append("a", k[:3], v[:3])
        cache.append("b", rows(5), rows(5))
        cache.append("a", k[3:], v[3:])
        packed_k, packed_v = cache.read("a")
        q = torch.tensor([[[[0.3, -0.4]]]], dtype=torch.float64, device="cpu")
        actual = F.scaled_dot_product_attention(
            q, packed_k[None, None], packed_v[None, None], dropout_p=0.0
        )
        expected = F.scaled_dot_product_attention(
            q, k[None, None], v[None, None], dropout_p=0.0
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_invalid_input_cannot_consume_blocks(self) -> None:
        cache = self.make_cache()
        invalid = [
            (rows(2).float(), rows(2)),
            (rows(2), rows(1)),
            (torch.zeros(2, 3, dtype=torch.float64, device="cpu"), rows(2)),
            (rows(2)[None], rows(2)[None]),
            (rows(0), rows(0)),
            (torch.empty(1, 2, dtype=torch.float64, device="meta"), rows(1)),
        ]
        for k, v in invalid:
            with (
                self.subTest(shape=k.shape, dtype=k.dtype),
                self.assertRaises(ValueError),
            ):
                cache.append("a", k, v)
            self.assertEqual(cache.free_block_count, 4)
        for target in (0, 1):
            for value in (float("nan"), float("inf"), -float("inf")):
                pair = [rows(1), rows(1)]
                pair[target][0, 0] = value
                with (
                    self.subTest(target=target, value=value),
                    self.assertRaises(ValueError),
                ):
                    cache.append("a", *pair)
                self.assertEqual(cache.free_block_count, 4)

    def test_request_id_and_configuration(self) -> None:
        for field in ("num_blocks", "block_size", "head_dim"):
            for value in (0, -1, True, 1.5):
                kwargs = dict(num_blocks=2, block_size=4, head_dim=2)
                kwargs[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    PagedKVCache(**kwargs)
        cache = self.make_cache()
        for request_id in ("", 1, None):
            with self.subTest(request_id=request_id), self.assertRaises(ValueError):
                cache.append(request_id, rows(1), rows(1))
        for operation in (cache.read, cache.block_table, cache.length, cache.release):
            with self.assertRaises(KeyError):
                operation("missing")

    def test_noncontiguous_input_and_no_autograd_history(self) -> None:
        cache = self.make_cache()
        k = (
            torch.arange(12, dtype=torch.float64, device="cpu")
            .reshape(2, 6)
            .t()
            .requires_grad_()
        )
        self.assertFalse(k.is_contiguous())
        cache.append("a", k, k)
        result, _ = cache.read("a")
        torch.testing.assert_close(result, k, rtol=0, atol=0)
        self.assertFalse(result.requires_grad)

    def test_default_device_does_not_change_cpu_contract(self) -> None:
        k = rows(2)
        with torch.device("meta"):
            cache = self.make_cache()
            cache.append("a", k, k)
            actual, _ = cache.read("a")
        self.assertEqual(actual.device.type, "cpu")
        torch.testing.assert_close(actual, k, rtol=0, atol=0)

    def test_randomized_append_release_matches_dense_oracle(self) -> None:
        rng = random.Random(391)
        cache = self.make_cache(blocks=5, size=3)
        dense = {}
        for step in range(100):
            name = rng.choice(("a", "b", "c"))
            if name in dense and rng.random() < 0.25:
                cache.release(name)
                del dense[name]
            else:
                count = rng.randint(1, 4)
                k, v = rows(count, step * 10), rows(count, step * 10 + 1000)
                try:
                    cache.append(name, k, v)
                except MemoryError:
                    pass
                else:
                    if name in dense:
                        old_k, old_v = dense[name]
                        dense[name] = (torch.cat((old_k, k)), torch.cat((old_v, v)))
                    else:
                        dense[name] = (k.clone(), v.clone())
            owned = []
            for request, expected_pair in dense.items():
                owned.extend(cache.block_table(request))
                for actual, expected in zip(cache.read(request), expected_pair):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(owned), len(set(owned)))
            self.assertEqual(len(owned) + cache.free_block_count, 5)


if __name__ == "__main__":
    unittest.main()
