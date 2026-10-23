"""CPU comparisons against independent PyTorch softmax/SDPA references."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from answers import dense_attention, online_softmax, tiled_attention


class OnlineTests(unittest.TestCase):
    def test_masked_large_logit_is_removed_before_exp(self):
        original_exp = torch.exp

        def finite_exp(value):
            result = original_exp(value)
            if not bool(torch.isfinite(result).all()):
                raise AssertionError("masked logits must not overflow exp internally")
            return result

        with patch("answers.torch.exp", side_effect=finite_exp):
            out = online_softmax(
                torch.tensor([10000.0, 1.0, 2.0], dtype=torch.float64),
                1,
                torch.tensor([False, True, True]),
            )
        torch.testing.assert_close(
            out[1:], torch.softmax(torch.tensor([1.0, 2.0], dtype=torch.float64), 0)
        )

    def test_increasing_maximum_and_large_shift(self):
        x = torch.tensor([1000.0, 1001.0, 999.0, 1002.0], dtype=torch.float64)
        for block in [1, 2, 9]:
            torch.testing.assert_close(
                online_softmax(x, block), torch.softmax(x, 0), rtol=1e-12, atol=1e-12
            )

    def test_initial_masked_block_and_fully_masked(self):
        x = torch.tensor([9.0, 8.0, 1.0, 2.0], dtype=torch.float64)
        keep = torch.tensor([False, False, True, True])
        expected = torch.tensor(
            [0.0, 0.0, 0.2689414213699951, 0.7310585786300049], dtype=torch.float64
        )
        torch.testing.assert_close(
            online_softmax(x, 2, keep), expected, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(
            online_softmax(x, 2, torch.zeros(4, dtype=torch.bool)).count_nonzero(), 0
        )

    def test_equal_and_empty(self):
        torch.testing.assert_close(
            online_softmax(torch.zeros(5), 2),
            torch.full((5,), 0.2, dtype=torch.float64),
        )
        self.assertEqual(online_softmax(torch.empty(0), 2).numel(), 0)

    def test_invalid_vector_mask_and_block(self):
        for value, block, mask in [
            (torch.ones(2, 2), 2, None),
            (torch.ones(3), 0, None),
            (torch.ones(3), 2, torch.ones(3)),
            (torch.tensor([float("nan")]), 1, None),
        ]:
            with self.assertRaises(ValueError):
                online_softmax(value, block, mask)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.q = torch.randn(5, 3, dtype=torch.float64)
        self.k = torch.randn(7, 3, dtype=torch.float64)
        self.v = torch.randn(7, 4, dtype=torch.float64)

    def reference(self, q, k, v, mask=None):
        return F.scaled_dot_product_attention(
            q[None, None].double(),
            k[None, None].double(),
            v[None, None].double(),
            attn_mask=mask,
            dropout_p=0.0,
        )[0, 0]

    def test_hand_weighted_average(self):
        q = torch.tensor([[1.0]], dtype=torch.float64)
        k = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        v = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float64)
        expected = (torch.softmax(k[:, 0], 0) * v[:, 0]).sum()
        actual = tiled_attention(q, k, v, 1, 2)
        torch.testing.assert_close(actual[0, 0], expected, rtol=1e-12, atol=1e-12)

    def test_rectangular_edges_and_multiple_tiles(self):
        expected = self.reference(self.q, self.k, self.v)
        for bq, bk in [(1, 1), (2, 3), (8, 16)]:
            torch.testing.assert_close(
                tiled_attention(self.q, self.k, self.v, bq, bk),
                expected,
                rtol=1e-12,
                atol=1e-12,
            )
        torch.testing.assert_close(
            dense_attention(self.q, self.k, self.v), expected, rtol=1e-12, atol=1e-12
        )

    def test_top_left_causal_contract(self):
        keep = torch.arange(7)[None, :] <= torch.arange(5)[:, None]
        expected = self.reference(self.q, self.k, self.v, keep)
        torch.testing.assert_close(
            tiled_attention(self.q, self.k, self.v, 2, 3, causal=True),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_masked_first_tile_all_masked_row_and_mixed_rows(self):
        keep = torch.rand(5, 7) > 0.4
        keep[0] = False
        keep[1, :3] = False
        keep[1, 3:] = True
        keep[2] = False
        keep[2, -1] = True
        actual = tiled_attention(self.q, self.k, self.v, 2, 3, keep_mask=keep)
        torch.testing.assert_close(
            actual, self.reference(self.q, self.k, self.v, keep), rtol=1e-12, atol=1e-12
        )
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(actual[0].count_nonzero(), 0)
        torch.testing.assert_close(actual[2], self.v[-1], rtol=0, atol=0)

    def test_causal_and_explicit_mask_intersect(self):
        keep = torch.ones(5, 7, dtype=torch.bool)
        keep[:, 0] = False
        expected_mask = keep & (torch.arange(7)[None, :] <= torch.arange(5)[:, None])
        actual = tiled_attention(
            self.q, self.k, self.v, 2, 2, keep_mask=keep, causal=True
        )
        torch.testing.assert_close(
            actual,
            self.reference(self.q, self.k, self.v, expected_mask),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_large_finite_logits_are_stable(self):
        q = torch.tensor([[1000.0]], dtype=torch.float64)
        k = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        v = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        actual = tiled_attention(q, k, v, 1, 1)
        self.assertEqual(actual.item(), 3.0)

    def test_empty_queries_keys_and_values_width(self):
        for nq, nk, dv in [(0, 3, 2), (3, 0, 2), (2, 3, 0)]:
            q, k, v = torch.ones(nq, 4), torch.ones(nk, 4), torch.ones(nk, dv)
            actual = tiled_attention(q, k, v, 2, 2)
            self.assertEqual(actual.shape, (nq, dv))
            torch.testing.assert_close(actual, dense_attention(q, k, v), rtol=0, atol=0)

    def test_noncontiguous_float32_inputs_and_inference(self):
        q = self.q.float().t().contiguous().t().requires_grad_()
        k = self.k.float().t().contiguous().t()
        v = self.v.float().t().contiguous().t()
        before = q.clone()
        actual = tiled_attention(q, k, v, 2, 3)
        torch.testing.assert_close(
            actual, self.reference(q, k, v), rtol=1e-12, atol=1e-12
        )
        self.assertEqual(actual.dtype, torch.float64)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(q, before, rtol=0, atol=0)

    def test_invalid_contracts(self):
        for options in [
            {"block_q": 0},
            {"block_k": True},
            {"causal": 1},
            {"keep_mask": torch.ones(5, 7)},
            {"keep_mask": torch.ones(5, 6, dtype=torch.bool)},
        ]:
            with self.assertRaises(ValueError):
                tiled_attention(self.q, self.k, self.v, **options)
        with self.assertRaises(ValueError):
            tiled_attention(self.q, self.k[:, :2], self.v)
        with self.assertRaises(ValueError):
            tiled_attention(self.q.half(), self.k.half(), self.v.half())

    def test_reject_nonfinite_and_intermediate_overflow(self):
        for value in [float("inf"), float("nan")]:
            q = self.q.clone()
            q[0, 0] = value
            with self.assertRaises(ValueError):
                tiled_attention(q, self.k, self.v)
        huge = torch.tensor([[1e308]], dtype=torch.float64)
        with self.assertRaises(ValueError):
            tiled_attention(huge, huge, torch.ones(1, 1, dtype=torch.float64))
        with self.assertRaises(ValueError):
            tiled_attention(
                torch.zeros(1, 1, dtype=torch.float64),
                torch.zeros(2, 1, dtype=torch.float64),
                torch.full((2, 1), 1e308, dtype=torch.float64),
                1,
                2,
            )


if __name__ == "__main__":
    unittest.main()
