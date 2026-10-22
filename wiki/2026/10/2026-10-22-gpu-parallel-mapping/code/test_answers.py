"""Address coverage and sector geometry, independently of a GPU scheduler."""

import unittest

from answers import launch_map, sector_ids


class MappingTests(unittest.TestCase):
    def test_ten_elements_and_tail(self):
        rows = launch_map(10, 4)
        self.assertEqual(len(rows), 12)
        self.assertEqual([r.index for r in rows if r.active], list(range(10)))
        self.assertEqual(
            [(r.block, r.thread) for r in rows[-4:]], [(2, i) for i in range(4)]
        )
        self.assertEqual([r.active for r in rows[-4:]], [True, True, False, False])

    def test_coverage_many_boundaries(self):
        for n in [0, 1, 31, 32, 33, 65, 127]:
            for block in [1, 4, 32, 33, 64]:
                rows = launch_map(n, block)
                active = [r.index for r in rows if r.active]
                self.assertEqual(active, list(range(n)))
                self.assertEqual(len(rows), ((n + block - 1) // block) * block)

    def test_warps_restart_at_each_block(self):
        rows = launch_map(66, 33)
        self.assertEqual((rows[32].warp, rows[32].lane), (1, 0))
        self.assertEqual((rows[33].warp, rows[33].lane), (0, 0))

    def test_invalid_launch(self):
        for n, block in [(-1, 4), (1, 0), (True, 4), (1, 1.5)]:
            with self.assertRaises(ValueError):
                launch_map(n, block)


class SectorTests(unittest.TestCase):
    def test_aligned_contiguous_and_strided(self):
        self.assertEqual(
            sector_ids([4 * lane for lane in range(32)]), frozenset(range(4))
        )
        self.assertEqual(len(sector_ids([128 * lane for lane in range(32)])), 32)

    def test_misaligned_contiguous(self):
        self.assertEqual(
            sector_ids([4 + 4 * lane for lane in range(32)]), frozenset(range(5))
        )

    def test_cross_boundary_duplicate_and_empty(self):
        self.assertEqual(sector_ids([31], width=4), frozenset([0, 1]))
        self.assertEqual(sector_ids([0] * 32), frozenset([0]))
        self.assertEqual(sector_ids([]), frozenset())

    def test_masked_tail_uses_only_valid_addresses(self):
        tail = launch_map(10, 4)[-4:]
        self.assertEqual(
            sector_ids([r.index * 4 for r in tail if r.active]), frozenset([1])
        )

    def test_invalid_addresses_and_widths(self):
        for addresses in [[-1], [True], [1.2]]:
            with self.assertRaises(ValueError):
                sector_ids(addresses)
        for width, sector in [(0, 32), (4, 0), (True, 32)]:
            with self.assertRaises(ValueError):
                sector_ids([0], width, sector)


if __name__ == "__main__":
    unittest.main()
