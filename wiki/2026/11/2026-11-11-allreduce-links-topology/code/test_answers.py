"""Check arithmetic correctness, trace invariants and resource accounting."""

import copy
import math
import unittest

from costs import binomial_tree_cost, ring_cost
from ring import ring_allreduce


def flat_routes(ranks: int, bandwidth: float = 50e9):
    routes = {(r, (r + 1) % ranks): (f"link-{r}",) for r in range(ranks)}
    capacities = {f"link-{r}": bandwidth for r in range(ranks)}
    return routes, capacities


def two_node_routes():
    routes = {}
    capacities = {"A-tx": 50e9, "A-rx": 50e9, "B-tx": 50e9, "B-rx": 50e9}
    for source in range(4):
        for destination in range(4):
            if source == destination:
                continue
            a, b = source // 2, destination // 2
            if a == b:
                resource = f"local-{source}-{destination}"
                routes[source, destination] = (resource,)
                capacities[resource] = 200e9
            else:
                routes[source, destination] = (
                    "A-tx" if a == 0 else "B-tx",
                    "A-rx" if b == 0 else "B-rx",
                )
    return routes, capacities


class RingTests(unittest.TestCase):
    def test_four_rank_sum_and_reduce_scatter_ownership(self):
        inputs = [[rank * 10 + i for i in range(8)] for rank in range(4)]
        frozen = copy.deepcopy(inputs)
        result = ring_allreduce(inputs, element_bytes=4)
        expected = tuple(60 + 4 * i for i in range(8))
        self.assertEqual(result.outputs, (expected,) * 4)
        self.assertEqual(
            result.reduced_chunks,
            tuple(expected[2 * rank : 2 * rank + 2] for rank in range(4)),
        )
        self.assertEqual(inputs, frozen)

    def test_trace_has_two_phases_and_no_missing_round_or_edge(self):
        result = ring_allreduce([[rank] * 4 for rank in range(4)])
        self.assertEqual(len(result.messages), 24)
        for phase in ("reduce_scatter", "allgather"):
            for step in range(3):
                messages = [
                    m for m in result.messages if m.phase == phase and m.step == step
                ]
                self.assertEqual({m.source for m in messages}, set(range(4)))
                self.assertTrue(
                    all(m.destination == (m.source + 1) % 4 for m in messages)
                )

    def test_each_rank_sends_exactly_one_and_half_payloads(self):
        result = ring_allreduce([[rank] * 8 for rank in range(4)], element_bytes=4)
        self.assertEqual(result.sent_bytes, (48,) * 4)
        self.assertEqual(sum(result.sent_bytes), 2 * (4 - 1) * 32)

    def test_non_power_of_two_ring_is_supported(self):
        result = ring_allreduce([[1, 2, 3, 4, 5, 6]] * 3)
        self.assertEqual(result.outputs, ((3, 6, 9, 12, 15, 18),) * 3)

    def test_single_rank_is_identity_with_no_communication(self):
        result = ring_allreduce([[7, -2]])
        self.assertEqual(result.outputs, ((7, -2),))
        self.assertEqual(result.messages, ())
        self.assertEqual(result.sent_bytes, (0,))

    def test_negative_values_and_cancellation(self):
        result = ring_allreduce([[2, -3], [-2, 3]])
        self.assertEqual(result.outputs, ((0, 0), (0, 0)))

    def test_rejects_empty_or_ragged_inputs(self):
        for inputs in ([], [[]], [[1, 2], [1]]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                ring_allreduce(inputs)

    def test_rejects_unequal_chunks_in_this_teaching_interface(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ring_allreduce([[1, 2, 3], [4, 5, 6]])

    def test_rejects_non_integer_or_boolean_elements(self):
        for value in (1.2, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ring_allreduce([[value]])

    def test_rejects_invalid_element_width(self):
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width), self.assertRaises((ValueError, TypeError)):
                ring_allreduce([[1]], element_bytes=width)


class CostTests(unittest.TestCase):
    def test_flat_ring_matches_alpha_beta_formula(self):
        routes, capacities = flat_routes(4)
        result = ring_cost(256 * 2**20, (0, 1, 2, 3), routes, capacities, 5e-6)
        self.assertEqual(result.rounds, 6)
        expected = 6 * 5e-6 + 1.5 * 256 * 2**20 / 50e9
        self.assertAlmostEqual(result.estimated_seconds, expected)
        self.assertEqual(result.per_round_load_bytes["link-0"], 64 * 2**20)

    def test_shared_nic_counts_both_concurrent_cross_node_edges(self):
        routes, capacities = two_node_routes()
        grouped = ring_cost(256 * 2**20, (0, 1, 2, 3), routes, capacities, 5e-6)
        crossed = ring_cost(256 * 2**20, (0, 2, 1, 3), routes, capacities, 5e-6)
        self.assertEqual(grouped.per_round_load_bytes["A-tx"], 64 * 2**20)
        self.assertEqual(crossed.per_round_load_bytes["A-tx"], 128 * 2**20)
        self.assertAlmostEqual(
            crossed.estimated_seconds - 6 * 5e-6,
            2 * (grouped.estimated_seconds - 6 * 5e-6),
        )

    def test_more_ranks_at_same_payload_adds_rounds_and_traffic(self):
        r4, b4 = flat_routes(4)
        r8, b8 = flat_routes(8)
        small4 = ring_cost(4096, tuple(range(4)), r4, b4, 5e-6)
        small8 = ring_cost(4096, tuple(range(8)), r8, b8, 5e-6)
        self.assertGreater(small8.estimated_seconds, small4.estimated_seconds)

    def test_slow_resource_sets_capacity_term(self):
        routes, capacities = flat_routes(4)
        capacities["link-2"] = 1e9
        result = ring_cost(4000, (0, 1, 2, 3), routes, capacities, 0)
        self.assertEqual(result.bottleneck_seconds_per_round, 1000 / 1e9)

    def test_single_rank_has_zero_cost_without_routes(self):
        result = ring_cost(64, (0,), {}, {}, 0)
        self.assertEqual(result.estimated_seconds, 0)
        self.assertEqual(result.rounds, 0)

    def test_tiny_payload_tree_has_fewer_startups_in_ideal_model(self):
        routes, capacities = flat_routes(8)
        ring = ring_cost(4096, tuple(range(8)), routes, capacities, 5e-6)
        tree = binomial_tree_cost(4096, 8, 50e9, 5e-6)
        self.assertLess(tree, ring.estimated_seconds)

    def test_large_payload_full_block_tree_has_more_critical_path_bytes(self):
        routes, capacities = flat_routes(8)
        ring = ring_cost(256 * 2**20, tuple(range(8)), routes, capacities, 5e-6)
        tree = binomial_tree_cost(256 * 2**20, 8, 50e9, 5e-6)
        self.assertGreater(tree, ring.estimated_seconds)

    def test_rejects_invalid_rank_order(self):
        for order in ((), (0, 0), (1, 2), (False, 1)):
            with self.subTest(order=order), self.assertRaises((TypeError, ValueError)):
                ring_cost(64, order, {}, {}, 0)

    def test_rejects_missing_or_empty_routes(self):
        _routes, capacities = flat_routes(2)
        for bad in ({}, {(0, 1): (), (1, 0): ("link-1",)}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ring_cost(64, (0, 1), bad, capacities, 0)

    def test_rejects_unknown_or_duplicate_resource_in_route(self):
        routes, capacities = flat_routes(2)
        for bad in (("unknown",), ("link-0", "link-0")):
            routes[0, 1] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ring_cost(64, (0, 1), routes, capacities, 0)

    def test_rejects_non_finite_or_non_positive_capacity(self):
        routes, capacities = flat_routes(2)
        for value in (0, -1, math.inf, math.nan, True):
            capacities["link-0"] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                ring_cost(64, (0, 1), routes, capacities, 0)

    def test_rejects_invalid_payload_or_startup(self):
        routes, capacities = flat_routes(2)
        for payload in (0, -2, 3, True):
            with (
                self.subTest(payload=payload),
                self.assertRaises((TypeError, ValueError)),
            ):
                ring_cost(payload, (0, 1), routes, capacities, 0)
        for alpha in (-1, math.nan, math.inf, True):
            with self.subTest(alpha=alpha), self.assertRaises((TypeError, ValueError)):
                ring_cost(64, (0, 1), routes, capacities, alpha)

    def test_tree_restricts_rank_count_to_power_of_two(self):
        with self.assertRaisesRegex(ValueError, "power"):
            binomial_tree_cost(4096, 3, 50e9, 5e-6)

    def test_large_payload_ratio_has_finite_ring_time(self):
        routes, capacities = flat_routes(2, 1e308)
        result = ring_cost(10**309, (0, 1), routes, capacities, 0)
        self.assertAlmostEqual(result.estimated_seconds, 10.0)
        self.assertEqual(result.per_round_load_bytes["link-0"], 5 * 10**308)

    def test_large_payload_ratio_has_finite_tree_time(self):
        self.assertAlmostEqual(binomial_tree_cost(10**309, 2, 1e308, 0), 20.0)

    def test_single_rank_tree_is_zero_even_with_tiny_capacity(self):
        self.assertEqual(binomial_tree_cost(1, 1, 1e-320, 0), 0.0)

    def test_unrepresentable_scalar_inputs_have_explicit_errors(self):
        routes, capacities = flat_routes(2)
        with self.assertRaisesRegex(ValueError, "finite"):
            ring_cost(64, (0, 1), routes, capacities, 10**400)
        with self.assertRaisesRegex(ValueError, "finite"):
            binomial_tree_cost(64, 2, 10**400, 0)

    def test_unrepresentable_final_time_has_explicit_error(self):
        routes, capacities = flat_routes(2, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ring_cost(10**400, (0, 1), routes, capacities, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            binomial_tree_cost(10**400, 2, 1.0, 0)


if __name__ == "__main__":
    unittest.main()
