"""Executable acceptance criteria for the two broadcasting exercises."""

import unittest

import torch

from answers import broadcast_shape, pairwise_sqdist


class ShapeTests(unittest.TestCase):
    def test_valid_shapes_match_pytorch(self):
        pairs = [
            ((2, 3), (3,)),
            ((5, 1, 4), (3, 1)),
            ((), (2, 3)),
            ((0, 3), (1, 3)),
            ((1,), (0,)),
            ((), ()),
            ((2, 1), (1, 3)),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    broadcast_shape(left, right),
                    tuple(torch.broadcast_shapes(left, right)),
                )

    def test_conflicting_shapes(self):
        for left, right in [((2, 3), (2,)), ((0, 3), (2, 3)), ((2, 3), (3, 2))]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                broadcast_shape(left, right)

    def test_invalid_dimensions(self):
        for shape in [(-1, 3), (True, 3), (2.5, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                broadcast_shape(shape, (3,))


class BroadcastBehaviorTests(unittest.TestCase):
    def test_expand_is_a_zero_stride_view(self):
        bias = torch.tensor([10.0, 20.0, 30.0])
        expanded = bias.expand(2, 3)
        self.assertEqual(expanded.stride(), (0, 1))
        self.assertEqual(
            expanded.untyped_storage().data_ptr(), bias.untyped_storage().data_ptr()
        )
        self.assertEqual(expanded.numel(), 6)
        self.assertEqual(expanded.untyped_storage().nbytes(), 12)
        torch.testing.assert_close(
            expanded, torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        )

    def test_repeat_allocates_and_add_materializes(self):
        bias = torch.tensor([10.0, 20.0, 30.0])
        repeated = bias.repeat(2, 1)
        self.assertNotEqual(
            repeated.untyped_storage().data_ptr(), bias.untyped_storage().data_ptr()
        )
        out = torch.zeros(2, 3) + bias
        self.assertNotEqual(
            out.untyped_storage().data_ptr(), bias.untyped_storage().data_ptr()
        )
        self.assertEqual(out.untyped_storage().nbytes(), 24)

    def test_overlapping_bulk_write_rejected(self):
        expanded = torch.ones(3).expand(2, 3)
        with self.assertRaises(RuntimeError):
            expanded.add_(1)

    def test_bias_gradient_reduces_reused_axes(self):
        x = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
        bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        ((x + bias) ** 2).sum().backward()
        torch.testing.assert_close(bias.grad, (2 * (x + bias)).sum(dim=0))


class DistanceTests(unittest.TestCase):
    def test_rectangular_odd_tiles_and_noncontiguous_inputs(self):
        generator = torch.Generator().manual_seed(7)
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(7, 8, generator=generator, dtype=dtype)[:, ::2]
            y = torch.randn(5, 8, generator=generator, dtype=dtype)[:, ::2]
            expected = (x[:, None, :] - y[None, :, :]).square().sum(-1)
            actual = pairwise_sqdist(x, y, block_rows=3, block_cols=2)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, (7, 5))

    def test_equal_vectors_and_inference_contract(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        actual = pairwise_sqdist(x, x, block_rows=1, block_cols=1)
        torch.testing.assert_close(actual, torch.tensor([[0.0, 8.0], [8.0, 0.0]]))
        self.assertFalse(actual.requires_grad)

    def test_empty_rows_columns_and_features(self):
        for n, m, d in [(0, 3, 2), (2, 0, 3), (2, 3, 0)]:
            actual = pairwise_sqdist(
                torch.empty(n, d), torch.empty(m, d), block_rows=2, block_cols=2
            )
            self.assertEqual(actual.shape, (n, m))
            if d == 0:
                torch.testing.assert_close(actual, torch.zeros(n, m))

    def test_inference_does_not_propagate_forward_ad(self):
        with torch.autograd.forward_ad.dual_level():
            x = torch.ones(2, 3)
            dual = torch.autograd.forward_ad.make_dual(x, torch.ones_like(x))
            output = pairwise_sqdist(dual, torch.zeros(1, 3), 1, 1)
            self.assertIsNone(torch.autograd.forward_ad.unpack_dual(output).tangent)

    def test_reject_invalid_inputs(self):
        good = torch.ones(2, 3)
        for bad in [0, -1, True, 1.5]:
            with self.subTest(tile=bad), self.assertRaises(ValueError):
                pairwise_sqdist(good, good, block_rows=bad)
        for bad in [
            torch.ones(3),
            torch.ones(2, 4),
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(2, 3, dtype=torch.int64),
        ]:
            with (
                self.subTest(shape=bad.shape, dtype=bad.dtype),
                self.assertRaises(ValueError),
            ):
                pairwise_sqdist(good, bad)


if __name__ == "__main__":
    unittest.main()
