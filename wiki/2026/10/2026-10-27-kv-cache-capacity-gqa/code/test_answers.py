"""CPU checks for a byte ledger and grouped-query attention semantics."""

import unittest

import torch
from answers import grouped_attention, kv_cache_bytes, kv_heads_per_rank
from torch.nn import functional as F


class CapacityTests(unittest.TestCase):
    def test_one_token_counts_both_key_and_value(self):
        self.assertEqual(
            kv_cache_bytes([1], layers=1, kv_heads=1, head_dim=1, element_bytes=2),
            4,
        )

    def test_variable_lengths_are_summed_without_padding(self):
        lengths = [2, 5, 0]
        self.assertEqual(
            kv_cache_bytes(lengths, layers=3, kv_heads=2, head_dim=4, element_bytes=2),
            672,
        )
        self.assertEqual(lengths, [2, 5, 0])

    def test_mha_gqa_mqa_reference_capacities(self):
        for heads, expected in [(32, 4 * 2**30), (8, 2**30), (1, 128 * 2**20)]:
            with self.subTest(kv_heads=heads):
                self.assertEqual(
                    kv_cache_bytes(
                        [8192], layers=32, kv_heads=heads, head_dim=128, element_bytes=2
                    ),
                    expected,
                )

    def test_empty_and_zero_lengths(self):
        for lengths in [[], [0, 0]]:
            self.assertEqual(
                kv_cache_bytes(
                    lengths, layers=2, kv_heads=2, head_dim=3, element_bytes=2
                ),
                0,
            )

    def test_invalid_capacity_inputs(self):
        config = dict(layers=2, kv_heads=2, head_dim=3, element_bytes=2)
        for field in config:
            for invalid in [0, -1, True, 1.5]:
                with (
                    self.subTest(field=field, value=invalid),
                    self.assertRaises(ValueError),
                ):
                    kv_cache_bytes([1], **{**config, field: invalid})
        for invalid in [-1, True, 1.5]:
            with self.subTest(length=invalid), self.assertRaises(ValueError):
                kv_cache_bytes([invalid], **config)

    def test_tp_partition_and_replication(self):
        for heads, ranks, local in [(32, 1, 32), (8, 4, 2), (2, 4, 1), (1, 4, 1)]:
            with self.subTest(heads=heads, ranks=ranks):
                self.assertEqual(kv_heads_per_rank(heads, ranks), local)

    def test_invalid_tp_layout(self):
        for heads, ranks in [(8, 3), (3, 8), (0, 4), (2, 0), (True, 2), (2, 1.5)]:
            with self.subTest(heads=heads, ranks=ranks), self.assertRaises(ValueError):
                kv_heads_per_rank(heads, ranks)


class GroupedAttentionTests(unittest.TestCase):
    def inputs(self, query_heads=4, kv_heads=2, queries=2, keys=5, dtype=torch.float64):
        generator = torch.Generator().manual_seed(194)
        q = torch.randn(
            2, query_heads, queries, 3, generator=generator, dtype=dtype, device="cpu"
        )
        k = torch.randn(
            2, kv_heads, keys, 3, generator=generator, dtype=dtype, device="cpu"
        )
        v = torch.randn(
            2, kv_heads, keys, 3, generator=generator, dtype=dtype, device="cpu"
        )
        return q, k, v

    def assert_matches_sdpa(self, q, k, v, start):
        groups = q.shape[1] // k.shape[1]
        mask = torch.arange(k.shape[2], device="cpu")[None, :] <= (
            start + torch.arange(q.shape[2], device="cpu")[:, None]
        )
        expected = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(groups, dim=1),
            v.repeat_interleave(groups, dim=1),
            attn_mask=mask,
            dropout_p=0.0,
        )
        actual = grouped_attention(q, k, v, query_start=start)
        tolerance = 1e-12 if q.dtype == torch.float64 else 2e-6
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_mha(self):
        self.assert_matches_sdpa(*self.inputs(kv_heads=4), start=3)

    def test_gqa(self):
        self.assert_matches_sdpa(*self.inputs(kv_heads=2), start=3)

    def test_mqa(self):
        self.assert_matches_sdpa(*self.inputs(kv_heads=1), start=3)

    def test_full_prefill(self):
        self.assert_matches_sdpa(*self.inputs(queries=4, keys=4), start=0)

    def test_single_token_decode(self):
        self.assert_matches_sdpa(*self.inputs(queries=1, keys=5), start=4)

    def test_float32(self):
        self.assert_matches_sdpa(*self.inputs(dtype=torch.float32), start=3)

    def test_cpu_inputs_under_a_different_default_device(self):
        q, k, v = self.inputs()
        expected = grouped_attention(q, k, v, query_start=3)
        with torch.device("meta"):
            actual = grouped_attention(q, k, v, query_start=3)
        self.assertEqual(actual.device.type, "cpu")
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_contiguous_head_groups_not_interleaved(self):
        q = torch.zeros(1, 4, 1, 1, dtype=torch.float64, device="cpu")
        k = torch.zeros(1, 2, 1, 1, dtype=torch.float64, device="cpu")
        v = torch.tensor([2.0, 11.0], dtype=torch.float64, device="cpu").reshape(
            1, 2, 1, 1
        )
        actual = grouped_attention(q, k, v, query_start=0)
        torch.testing.assert_close(
            actual.flatten(),
            torch.tensor([2.0, 2.0, 11.0, 11.0], dtype=torch.float64, device="cpu"),
        )

    def test_shared_kv_does_not_force_shared_outputs(self):
        q = torch.tensor([0.0, 2.0], dtype=torch.float64, device="cpu").reshape(
            1, 2, 1, 1
        )
        k = torch.tensor([0.0, 1.0], dtype=torch.float64, device="cpu").reshape(
            1, 1, 2, 1
        )
        v = torch.tensor([0.0, 10.0], dtype=torch.float64, device="cpu").reshape(
            1, 1, 2, 1
        )
        actual = grouped_attention(q, k, v, query_start=1)
        self.assertEqual(actual[0, 0, 0, 0].item(), 5.0)
        self.assertGreater(actual[0, 1, 0, 0].item(), 8.0)

    def test_future_tokens_cannot_affect_earlier_queries(self):
        q, k, v = self.inputs(queries=3, keys=5)
        before = grouped_attention(q, k, v, query_start=0)
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[:, :, 3:] = 1000.0
        changed_v[:, :, 3:] = -1000.0
        after = grouped_attention(q, changed_k, changed_v, query_start=0)
        torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)

    def test_noncontiguous_inputs_and_input_ownership(self):
        q, k, v = (
            tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
            for tensor in self.inputs()
        )
        self.assertFalse(q.is_contiguous())
        originals = [tensor.clone() for tensor in (q, k, v)]
        self.assert_matches_sdpa(q, k, v, start=3)
        for tensor, original in zip((q, k, v), originals):
            self.assertTrue(torch.equal(tensor, original))

    def test_invalid_shapes(self):
        q, k, v = self.inputs()
        cases = [
            (q[:, :3], k, v),
            (q, k, v[:, :, :-1]),
            (q[0], k, v),
            (q, k[:1], v),
            (q, k[:, :, :, :2], v),
            (q[:, :, :0], k, v),
        ]
        for values in cases:
            with (
                self.subTest(shapes=[tuple(t.shape) for t in values]),
                self.assertRaises(ValueError),
            ):
                grouped_attention(*values, query_start=0)

    def test_invalid_positions(self):
        q, k, v = self.inputs()
        for start in [-1, True, 0.5, 4]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                grouped_attention(q, k, v, query_start=start)

    def test_invalid_dtype_and_nonfinite_input(self):
        q, k, v = self.inputs()
        for values in [(q.float(), k, v), (q.half(), k.half(), v.half())]:
            with self.assertRaises(ValueError):
                grouped_attention(*values, query_start=0)
        for index in range(3):
            for nonfinite in [float("nan"), float("inf"), -float("inf")]:
                inputs = [t.clone() for t in (q, k, v)]
                inputs[index][0, 0, 0, 0] = nonfinite
                with (
                    self.subTest(input_index=index, value=nonfinite),
                    self.assertRaises(ValueError),
                ):
                    grouped_attention(*inputs, query_start=0)

    def test_score_overflow_is_not_silently_accepted(self):
        q, k, v = self.inputs(dtype=torch.float32)
        q.fill_(torch.finfo(torch.float32).max)
        k.fill_(torch.finfo(torch.float32).max)
        with self.assertRaises(ValueError):
            grouped_attention(q, k, v, query_start=0)


if __name__ == "__main__":
    unittest.main()
