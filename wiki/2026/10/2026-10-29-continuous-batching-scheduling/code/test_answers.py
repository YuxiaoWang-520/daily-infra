"""Hand-calculated traces and independent accounting invariants."""

import os
import unittest
from collections import Counter
from importlib import import_module
from itertools import product

from models import Prefill, Request

implementation = import_module(os.environ.get("EXERCISE_IMPL", "answers"))
simulate = implementation.simulate
admit_prefills = implementation.admit_prefills


class SchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests = tuple(
            Request(name, 0, count) for name, count in zip("ABCD", (1, 4, 2, 1))
        )

    def test_continuous_trace(self) -> None:
        result = simulate(self.requests, 2, "continuous")
        self.assertEqual(
            [step.request_ids for step in result.steps],
            [("A", "B"), ("B", "C"), ("B", "C"), ("B", "D")],
        )
        self.assertEqual(
            dict(result.output_ticks),
            {
                "A": (1,),
                "B": (1, 2, 3, 4),
                "C": (2, 3),
                "D": (4,),
            },
        )

    def test_static_trace(self) -> None:
        result = simulate(self.requests, 2, "static")
        self.assertEqual(
            [step.request_ids for step in result.steps],
            [("A", "B"), ("B",), ("B",), ("B",), ("C", "D"), ("C",)],
        )
        self.assertEqual(dict(result.output_ticks)["C"], (5, 6))

    def test_future_arrival_waits_for_next_boundary(self) -> None:
        result = simulate((Request("A", 0, 3), Request("B", 1, 1)), 2, "continuous")
        self.assertEqual(
            [s.request_ids for s in result.steps], [("A",), ("A", "B"), ("A",)]
        )
        self.assertEqual(dict(result.output_ticks)["B"], (2,))

    def test_static_late_joiner_waits_for_batch(self) -> None:
        result = simulate((Request("A", 0, 3), Request("B", 1, 1)), 2, "static")
        self.assertEqual(dict(result.output_ticks)["B"], (4,))

    def test_idle_time_is_not_erased(self) -> None:
        result = simulate((Request("A", 3, 1), Request("B", 8, 1)), 2, "continuous")
        self.assertEqual([s.start_tick for s in result.steps], [3, 8])
        self.assertEqual(dict(result.output_ticks), {"A": (4,), "B": (9,)})

    def test_arrival_order_and_stable_ties(self) -> None:
        requests = (Request("late", 4, 1), Request("early", 0, 1), Request("tie", 0, 1))
        result = simulate(requests, 1, "continuous")
        self.assertEqual(
            [s.request_ids for s in result.steps], [("early",), ("tie",), ("late",)]
        )

    def test_capacity_one_has_same_schedule(self) -> None:
        self.assertEqual(
            simulate(self.requests, 1, "static"),
            simulate(self.requests, 1, "continuous"),
        )

    def test_empty_input(self) -> None:
        result = simulate((), 2, "continuous")
        self.assertEqual(result.steps, ())
        self.assertEqual(result.output_ticks, ())

    def test_simulation_preserves_mutable_input(self) -> None:
        requests = list(reversed(self.requests))
        saved = tuple(requests)
        for policy in ("static", "continuous"):
            first = simulate(requests, 2, policy)
            self.assertEqual(tuple(requests), saved)
            self.assertEqual(simulate(requests, 2, policy), first)

    def test_duplicate_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate((Request("A", 0, 1), Request("A", 1, 1)), 2, "continuous")

    def test_invalid_inputs_rejected(self) -> None:
        for capacity in (0, -1, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                simulate(self.requests, capacity, "continuous")
        with self.assertRaises(ValueError):
            simulate(self.requests, 2, "unknown")
        for args in (("", 0, 1), ("A", -1, 1), ("A", 0, 0), ("A", True, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Request(*args)

    def test_generated_cases_conserve_outputs_and_capacity(self) -> None:
        for lengths in product((1, 2, 3), repeat=3):
            requests = tuple(
                Request(name, i % 2, n)
                for i, (name, n) in enumerate(zip("ABC", lengths))
            )
            for policy in ("static", "continuous"):
                for capacity in (1, 2, 3):
                    with self.subTest(
                        lengths=lengths, policy=policy, capacity=capacity
                    ):
                        result = simulate(requests, capacity, policy)
                        counts = Counter(
                            name for step in result.steps for name in step.request_ids
                        )
                        self.assertEqual(
                            counts,
                            Counter({r.request_id: r.output_tokens for r in requests}),
                        )
                        self.assertTrue(
                            all(
                                0 < len(s.request_ids) <= capacity for s in result.steps
                            )
                        )
                        self.assertTrue(
                            all(
                                len(set(s.request_ids)) == len(s.request_ids)
                                for s in result.steps
                            )
                        )
                        for request in requests:
                            ticks = dict(result.output_ticks)[request.request_id]
                            self.assertTrue(
                                all(tick > request.arrival_tick for tick in ticks)
                            )
                            self.assertEqual(tuple(sorted(set(ticks))), ticks)


class AdmissionTests(unittest.TestCase):
    def plan(self, waiting=None, **changes):
        parameters = dict(
            running_decodes=2,
            max_num_seqs=4,
            token_budget=8,
            free_blocks=3,
            block_size=4,
        )
        parameters.update(changes)
        if waiting is None:
            waiting = (Prefill("C", 4), Prefill("D", 3), Prefill("E", 1))
        return admit_prefills(waiting, **parameters)

    def test_token_budget_binds_before_request_capacity(self) -> None:
        result = self.plan()
        self.assertEqual(result.admitted, ("C",))
        self.assertEqual(
            (result.num_scheduled_requests, result.num_scheduled_tokens), (3, 6)
        )
        self.assertEqual((result.remaining_tokens, result.remaining_blocks), (2, 2))
        self.assertEqual(
            (result.stopped_request, result.stop_reason), ("D", "token_budget")
        )

    def test_sequence_limit_binds_with_large_token_budget(self) -> None:
        result = self.plan(token_budget=16)
        self.assertEqual(result.admitted, ("C", "D"))
        self.assertEqual(
            (result.num_scheduled_requests, result.num_scheduled_tokens), (4, 9)
        )
        self.assertEqual(result.stop_reason, "sequence_limit")

    def test_kv_limit_uses_ceil_division(self) -> None:
        result = self.plan((Prefill("C", 5),), free_blocks=1)
        self.assertEqual(result.admitted, ())
        self.assertEqual((result.remaining_tokens, result.remaining_blocks), (6, 1))
        self.assertEqual(result.stop_reason, "kv_capacity")

    def test_exact_limits_fit(self) -> None:
        result = self.plan((Prefill("C", 6),), max_num_seqs=3, free_blocks=2)
        self.assertEqual(result.admitted, ("C",))
        self.assertEqual((result.remaining_tokens, result.remaining_blocks), (0, 0))
        self.assertIsNone(result.stop_reason)

    def test_zero_running_requests(self) -> None:
        result = self.plan(
            (Prefill("C", 4),), running_decodes=0, token_budget=4, free_blocks=1
        )
        self.assertEqual(
            (result.num_scheduled_requests, result.num_scheduled_tokens), (1, 4)
        )

    def test_empty_waiting_preserves_running_work(self) -> None:
        result = self.plan(())
        self.assertEqual(result.admitted, ())
        self.assertEqual(result.num_scheduled_tokens, 2)
        self.assertEqual((result.remaining_tokens, result.remaining_blocks), (6, 3))

    def test_failed_admission_does_not_charge_budget(self) -> None:
        result = self.plan(
            (Prefill("C", 4), Prefill("D", 5)), token_budget=12, free_blocks=2
        )
        self.assertEqual(result.admitted, ("C",))
        self.assertEqual((result.remaining_tokens, result.remaining_blocks), (6, 1))
        self.assertEqual(result.stop_reason, "kv_capacity")

    def test_fcfs_does_not_skip_large_head(self) -> None:
        result = self.plan((Prefill("large", 7), Prefill("small", 1)))
        self.assertEqual(result.admitted, ())
        self.assertEqual(result.stopped_request, "large")

    def test_no_request_slot_or_no_blocks(self) -> None:
        self.assertEqual(self.plan(max_num_seqs=2).stop_reason, "sequence_limit")
        self.assertEqual(self.plan(free_blocks=0).stop_reason, "kv_capacity")

    def test_invalid_counts_and_infeasible_running_work(self) -> None:
        for changes in (
            dict(running_decodes=-1),
            dict(running_decodes=5),
            dict(token_budget=1),
            dict(token_budget=0),
            dict(max_num_seqs=0),
            dict(free_blocks=-1),
            dict(block_size=0),
            dict(block_size=True),
            dict(free_blocks=1.5),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(**changes)
        with self.assertRaises(ValueError):
            self.plan((Prefill("C", 1), Prefill("C", 2)))
        with self.assertRaises(ValueError):
            Prefill("C", 0)

    def test_inputs_unchanged_and_repeatable(self) -> None:
        waiting = [Prefill("C", 4), Prefill("D", 3)]
        saved = tuple(waiting)
        first = self.plan(waiting)
        self.assertEqual(tuple(waiting), saved)
        self.assertEqual(self.plan(waiting), first)


if __name__ == "__main__":
    unittest.main()
