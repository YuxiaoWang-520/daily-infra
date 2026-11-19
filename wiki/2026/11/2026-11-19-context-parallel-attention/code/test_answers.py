"""Independent SDPA and finite-difference checks for the two CP exercises."""

import math
import unittest

import torch
from answers import (
    KVBlock,
    causal_pair_counts,
    cp_attention,
    ring_order,
    sequence_shards,
    streaming_attention,
)
from torch.nn import functional as F


def dense(q, k, v, query_positions=None, key_positions=None, causal=True):
    """Use PyTorch SDPA, independently of the online-softmax implementation."""
    qp = torch.arange(q.shape[0]) if query_positions is None else query_positions
    kp = torch.arange(k.shape[0]) if key_positions is None else key_positions
    mask = qp[:, None] >= kp[None, :] if causal else None
    return F.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], attn_mask=mask
    )[0, 0]


def data(seq=8, dim=3, value_dim=2):
    generator = torch.Generator().manual_seed(19)
    return tuple(
        torch.randn(seq, width, generator=generator, dtype=torch.float64)
        for width in (dim, dim, value_dim)
    )


def blocks(k, v, order=(0, 1, 2, 3)):
    return [
        KVBlock(
            k[2 * i : 2 * i + 2], v[2 * i : 2 * i + 2], torch.arange(2 * i, 2 * i + 2)
        )
        for i in order
    ]


def leaves(values):
    return tuple(value.detach().clone().requires_grad_(True) for value in values)


class StreamingTests(unittest.TestCase):
    def test_numeric_causal_example(self):
        q = torch.ones(4, 1, dtype=torch.float64)
        k = torch.log(torch.arange(1, 5, dtype=torch.float64))[:, None]
        v = torch.arange(10, 50, 10, dtype=torch.float64)[:, None]
        out, lse = streaming_attention(q, blocks(k, v, (0, 1)), torch.arange(4))
        torch.testing.assert_close(
            out[:, 0], torch.tensor([10, 50 / 3, 140 / 6, 30], dtype=q.dtype)
        )
        torch.testing.assert_close(
            lse, torch.log(torch.tensor([1, 3, 6, 10], dtype=q.dtype))
        )

    def test_local_softmax_average_is_wrong(self):
        q = torch.ones(1, 1, dtype=torch.float64)
        k = torch.log(torch.arange(1, 5, dtype=torch.float64))[:, None]
        v = torch.arange(10, 50, 10, dtype=torch.float64)[:, None]
        local = [dense(q, k[s], v[s], causal=False) for s in (slice(0, 2), slice(2, 4))]
        self.assertAlmostEqual(float((local[0] + local[1]) / 2), 550 / 21)
        self.assertGreater(abs(float((local[0] + local[1]) / 2) - 30), 3)

    def test_noncausal_matches_sdpa(self):
        q, k, v = data()
        actual, _ = streaming_attention(
            q, iter(blocks(k, v)), torch.arange(8), causal=False
        )
        torch.testing.assert_close(
            actual, dense(q, k, v, causal=False), rtol=1e-12, atol=1e-12
        )

    def test_causal_output_and_all_gradients_match_sdpa(self):
        q, k, v = leaves(data())
        actual, _ = streaming_attention(q, blocks(k, v), torch.arange(8))
        actual.square().mean().backward()
        qr, kr, vr = leaves(data())
        expected = dense(qr, kr, vr)
        expected.square().mean().backward()
        for a, b in zip(
            (actual, q.grad, k.grad, v.grad), (expected, qr.grad, kr.grad, vr.grad)
        ):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_reordered_blocks_keep_output_and_gradients(self):
        results = []
        for order in [(0, 1, 2, 3), (3, 1, 0, 2)]:
            q, k, v = leaves(data())
            out, lse = streaming_attention(q, blocks(k, v, order), torch.arange(8))
            out.sin().sum().backward()
            results.append((out, lse, q.grad, k.grad, v.grad))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_fully_masked_first_block_has_finite_gradients(self):
        q, k, v = leaves(data())
        out, _ = streaming_attention(q[:2], blocks(k, v, (3, 2, 1, 0)), torch.arange(2))
        out.square().sum().backward()
        torch.testing.assert_close(out, dense(q[:2], k, v), rtol=1e-12, atol=1e-12)
        for grad in (q.grad, k.grad, v.grad):
            self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_fully_masked_query_returns_zero_not_nan(self):
        q, k, v = leaves(data())
        out, lse = streaming_attention(q[:2], blocks(k, v, (3,)), torch.arange(2))
        self.assertTrue(bool((out == 0).all()))
        self.assertTrue(bool(torch.isneginf(lse).all()))
        torch.testing.assert_close(
            out, dense(q[:2], k[6:8], v[6:8], torch.arange(2), torch.arange(6, 8))
        )
        out.sum().backward()
        for grad in (q.grad, k.grad, v.grad):
            self.assertTrue(bool((grad == 0).all()))

    def test_global_positions_define_mask(self):
        q, k, v = data()
        qp = torch.tensor([100, 103])
        kp = torch.arange(100, 108)
        out, _ = streaming_attention(q[:2], [KVBlock(k, v, kp)], qp)
        expected = dense(q[:2], k, v, qp, kp)
        torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(float((out - dense(q[:2], k, v)).abs().max()), 0.01)

    def test_large_logits_stay_finite(self):
        q, k, v = data()
        q = q * 500
        k = k * 500
        out, lse = streaming_attention(q, blocks(k, v), torch.arange(8))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool(torch.isfinite(lse).all()))
        torch.testing.assert_close(out, dense(q, k, v), rtol=1e-10, atol=1e-10)

    def test_empty_key_block_is_neutral(self):
        q, k, v = data()
        empty = KVBlock(k[:0], v[:0], torch.arange(0))
        out, _ = streaming_attention(q, [empty, *blocks(k, v), empty], torch.arange(8))
        torch.testing.assert_close(out, dense(q, k, v), rtol=1e-12, atol=1e-12)

    def test_only_empty_keys_allow_zero_gradients(self):
        q, k, v = leaves(data())
        out, lse = streaming_attention(
            q, [KVBlock(k[:0], v[:0], torch.arange(0))], torch.arange(8)
        )
        self.assertTrue(bool((out == 0).all()))
        self.assertTrue(bool(torch.isneginf(lse).all()))
        out.sum().backward()
        for grad in (q.grad, k.grad, v.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(bool((grad == 0).all()))

    def test_finite_inputs_with_overflowing_scores_rejected(self):
        q, k, v = data()
        with self.assertRaises(ValueError):
            streaming_attention(q * 1e300, blocks(k * 1e300, v), torch.arange(8))

    def test_missing_blocks_rejected(self):
        q, _, _ = data()
        with self.assertRaises(ValueError):
            streaming_attention(q, [], torch.arange(8))

    def test_bad_shapes_dtype_and_positions_rejected(self):
        q, k, v = data()
        cases = [
            (q[:, 0], [KVBlock(k, v, torch.arange(8))], torch.arange(8)),
            (q, [KVBlock(k[:, :2], v, torch.arange(8))], torch.arange(8)),
            (q, [KVBlock(k, v[:7], torch.arange(8))], torch.arange(8)),
            (q, [KVBlock(k.float(), v, torch.arange(8))], torch.arange(8)),
            (q, [KVBlock(k, v, torch.arange(8).float())], torch.arange(8)),
            (q, [KVBlock(k, v, torch.arange(8))], torch.arange(7)),
            (q, [KVBlock(k, v, torch.arange(7))], torch.arange(8)),
            (q, [KVBlock(k, v[:, :0], torch.arange(8))], torch.arange(8)),
        ]
        for aq, ab, ap in cases:
            with (
                self.subTest(q=aq.shape, k=ab[0].k.shape, positions=ap.shape),
                self.assertRaises((TypeError, ValueError)),
            ):
                streaming_attention(aq, ab, ap)

    def test_nonfinite_input_rejected(self):
        for which in range(3):
            values = list(data())
            values[which][0, 0] = math.inf
            q, k, v = values
            with self.subTest(which=which), self.assertRaises(ValueError):
                streaming_attention(q, blocks(k, v), torch.arange(8))

    def test_finite_difference_autograd_check(self):
        q, k, v = leaves(data(seq=4, dim=2, value_dim=2))

        def function(aq, ak, av):
            out, _ = streaming_attention(aq, blocks(ak, av, (1, 0)), torch.arange(4))
            return out

        self.assertTrue(
            torch.autograd.gradcheck(function, (q, k, v), eps=1e-6, atol=1e-5)
        )

    def test_logsumexp_matches_full_scores(self):
        q, k, v = data()
        _, lse = streaming_attention(q, blocks(k, v), torch.arange(8))
        scores = q @ k.T / math.sqrt(q.shape[1])
        scores = scores.masked_fill(
            torch.arange(8)[None, :] > torch.arange(8)[:, None], -math.inf
        )
        torch.testing.assert_close(
            lse, torch.logsumexp(scores, dim=-1), rtol=1e-12, atol=1e-12
        )


class ContextParallelTests(unittest.TestCase):
    def test_contiguous_positions_and_pair_counts(self):
        shards = sequence_shards(8, 2, layout="contiguous")
        self.assertEqual(shards, ((0, 1, 2, 3), (4, 5, 6, 7)))
        self.assertEqual(causal_pair_counts(shards), (10, 26))

    def test_zigzag_balances_causal_pairs(self):
        shards = sequence_shards(8, 2, layout="zigzag")
        self.assertEqual(shards, ((0, 1, 6, 7), (2, 3, 4, 5)))
        self.assertEqual(causal_pair_counts(shards), (18, 18))
        self.assertEqual(
            causal_pair_counts(sequence_shards(16, 4, layout="zigzag")), (34,) * 4
        )

    def test_layout_invalid_inputs(self):
        for seq, cp, layout in [
            (7, 2, "contiguous"),
            (10, 2, "zigzag"),
            (8, 0, "zigzag"),
            (8, True, "zigzag"),
            (8, 2, "unknown"),
            (0, 2, "contiguous"),
        ]:
            with (
                self.subTest(seq=seq, cp=cp, layout=layout),
                self.assertRaises(ValueError),
            ):
                sequence_shards(seq, cp, layout=layout)

    def test_ring_visits_every_owner_once(self):
        self.assertEqual(ring_order(0, 4), (0, 3, 2, 1))
        for cp in (1, 2, 4):
            for rank in range(cp):
                order = ring_order(rank, cp)
                self.assertEqual(order[0], rank)
                self.assertEqual(sorted(order), list(range(cp)))
        for rank, cp in [(-1, 2), (2, 2), (0, 0), (True, 2)]:
            with self.assertRaises(ValueError):
                ring_order(rank, cp)

    def test_cp_one_with_odd_length(self):
        q, k, v = data(seq=7)
        for layout in ("contiguous", "zigzag"):
            torch.testing.assert_close(
                cp_attention(q, k, v, 1, layout=layout), dense(q, k, v)
            )

    def test_cp_two_restores_global_output_order(self):
        q, k, v = data()
        for layout in ("contiguous", "zigzag"):
            actual = cp_attention(q, k, v, 2, layout=layout)
            torch.testing.assert_close(actual, dense(q, k, v), rtol=1e-12, atol=1e-12)

    def test_cp_four_zigzag_all_gradients(self):
        values = data(seq=16)
        q, k, v = leaves(values)
        actual = cp_attention(q, k, v, 4, layout="zigzag")
        actual.square().mean().backward()
        qr, kr, vr = leaves(values)
        expected = dense(qr, kr, vr)
        expected.square().mean().backward()
        for a, b in zip(
            (actual, q.grad, k.grad, v.grad), (expected, qr.grad, kr.grad, vr.grad)
        ):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_cp_noncausal(self):
        q, k, v = data()
        for layout in ("contiguous", "zigzag"):
            out = cp_attention(q, k, v, 2, layout=layout, causal=False)
            torch.testing.assert_close(
                out, dense(q, k, v, causal=False), rtol=1e-12, atol=1e-12
            )

    def test_cp_shape_mismatch_rejected(self):
        q, k, v = data()
        with self.assertRaises(ValueError):
            cp_attention(q, k[:7], v, 2)


if __name__ == "__main__":
    unittest.main()
