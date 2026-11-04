"""Hand-counted payloads and independent storage-lifetime expectations."""

import importlib
import os
import unittest
from dataclasses import FrozenInstanceError

from models import Event

impl = importlib.import_module(os.environ.get("EXERCISE_IMPL", "answers"))


class StateTests(unittest.TestCase):
    def test_fp32_adam(self):
        b = impl.estimate_state_bytes(16, 4, 4, 4)
        self.assertEqual(
            (
                b.parameters,
                b.gradients,
                b.first_moment,
                b.second_moment,
                b.master,
                b.total,
            ),
            (64, 64, 64, 64, 0, 256),
        )

    def test_explicit_master_is_an_additional_copy(self):
        b = impl.estimate_state_bytes(16, 2, 2, 4, 4)
        self.assertEqual(
            (b.parameters, b.gradients, b.master, b.total), (32, 32, 64, 256)
        )

    def test_low_precision_state_is_an_explicit_assumption(self):
        self.assertEqual(impl.estimate_state_bytes(16, 2, 2, 2).total, 128)

    def test_before_first_backward(self):
        b = impl.estimate_state_bytes(
            16, 4, 4, 4, gradients_live=False, state_initialized=False
        )
        self.assertEqual(b.total, 64)

    def test_after_backward_before_first_step(self):
        b = impl.estimate_state_bytes(16, 4, 4, 4, state_initialized=False)
        self.assertEqual(b.total, 128)

    def test_zero_grad_none_keeps_optimizer_state(self):
        b = impl.estimate_state_bytes(16, 4, 4, 4, gradients_live=False)
        self.assertEqual(
            (b.gradients, b.first_moment, b.second_moment, b.total), (0, 64, 64, 192)
        )

    def test_master_lifetime_is_independent_of_moment_initialization(self):
        b = impl.estimate_state_bytes(
            16, 2, 2, 4, 4, gradients_live=False, state_initialized=False
        )
        self.assertEqual((b.master, b.total), (64, 96))

    def test_amsgrad_adds_one_moment_sized_array(self):
        self.assertEqual(
            impl.estimate_state_bytes(16, 4, 4, 4, amsgrad=True).total, 320
        )
        self.assertEqual(
            impl.estimate_state_bytes(
                16, 4, 4, 4, state_initialized=False, amsgrad=True
            ).amsgrad_max,
            0,
        )

    def test_large_integer_counts_and_zero(self):
        self.assertEqual(
            impl.estimate_state_bytes(10**9, 4, 4, 4).total, 16_000_000_000
        )
        self.assertEqual(impl.estimate_state_bytes(0, 4, 4, 4).total, 0)

    def test_bad_counts_widths_and_flags(self):
        for args, kwargs in [
            ((-1, 4, 4, 4), {}),
            ((True, 4, 4, 4), {}),
            ((1.5, 4, 4, 4), {}),
            ((1, 0, 4, 4), {}),
            ((1, 4, True, 4), {}),
            ((1, 4, 4, 4, -1), {}),
            ((1, 4, 4, 4), {"gradients_live": 1}),
            ((1, 4, 4, 4), {"state_initialized": None}),
            ((1, 4, 4, 4), {"amsgrad": "yes"}),
        ]:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                impl.estimate_state_bytes(*args, **kwargs)

    def test_immutable_result(self):
        b = impl.estimate_state_bytes(1, 4, 4, 4)
        with self.assertRaises(FrozenInstanceError):
            b.parameters = 0


class LifetimeTests(unittest.TestCase):
    def scenario(self):
        return [
            Event("alloc", "weights", 64),
            Event("alloc", "m", 64),
            Event("alloc", "v", 64),
            Event("alloc", "activation", 128),
            Event("alias", "view", source="activation"),
            Event("alloc", "grad", 64),
            Event("drop", "activation"),
            Event("drop", "view"),
            Event("alloc", "temp", 96),
            Event("drop", "temp"),
            Event("drop", "grad"),
        ]

    def test_hand_counted_peak_and_final(self):
        report = impl.replay_lifetimes(self.scenario())
        self.assertEqual(
            [p.live_bytes for p in report.points],
            [64, 128, 192, 320, 320, 384, 384, 256, 352, 256, 192],
        )
        self.assertEqual(
            (report.peak_bytes, report.peak_event_index, report.final_bytes),
            (384, 5, 192),
        )

    def test_alias_does_not_allocate(self):
        report = impl.replay_lifetimes(
            [Event("alloc", "a", 32), Event("alias", "b", source="a")]
        )
        self.assertEqual(
            (
                report.peak_bytes,
                report.final_bytes,
                report.points[-1].live_storage_count,
            ),
            (32, 32, 1),
        )

    def test_last_reference_controls_release(self):
        report = impl.replay_lifetimes(
            [
                Event("alloc", "a", 32),
                Event("alias", "b", source="a"),
                Event("drop", "a"),
                Event("drop", "b"),
            ]
        )
        self.assertEqual([p.live_bytes for p in report.points], [32, 32, 32, 0])

    def test_alias_chain_and_name_reuse(self):
        events = [
            Event("alloc", "a", 8),
            Event("alias", "b", source="a"),
            Event("alias", "c", source="b"),
            Event("drop", "a"),
            Event("alloc", "a", 16),
            Event("drop", "b"),
            Event("drop", "c"),
        ]
        result = impl.replay_lifetimes(events)
        self.assertEqual((result.peak_bytes, result.final_bytes), (24, 16))

    def test_retained_view_raises_later_peak(self):
        events = self.scenario()
        events.pop(7)
        result = impl.replay_lifetimes(events)
        self.assertEqual((result.peak_bytes, result.final_bytes), (480, 320))

    def test_sum_of_component_maxima_overestimates(self):
        report = impl.replay_lifetimes(self.scenario())
        self.assertLess(report.peak_bytes, 64 + 64 + 64 + 128 + 64 + 96)

    def test_empty_trace_and_first_peak_tie(self):
        empty = impl.replay_lifetimes([])
        self.assertEqual(
            (empty.peak_bytes, empty.peak_event_index, empty.final_bytes, empty.points),
            (0, None, 0, ()),
        )
        report = impl.replay_lifetimes(
            [Event("alloc", "a", 8), Event("drop", "a"), Event("alloc", "b", 8)]
        )
        self.assertEqual(report.peak_event_index, 0)

    def test_input_and_output_immutability(self):
        events = self.scenario()
        before = tuple(events)
        report = impl.replay_lifetimes(events)
        self.assertEqual(tuple(events), before)
        with self.assertRaises(FrozenInstanceError):
            report.points[0].live_bytes = 0

    def test_invalid_events(self):
        invalid = [
            [Event("alloc", "", 4)],
            [Event("alloc", "a", True)],
            [Event("alloc", "a", 0)],
            [Event("alloc", "a", 4, "x")],
            [Event("drop", "missing")],
            [Event("alias", "b", source="missing")],
            [Event("alloc", "a", 4), Event("alloc", "a", 8)],
            [Event("alloc", "a", 4), Event("alias", "b", 4, "a")],
            [Event("alloc", "a", 4), Event("drop", "a", source="a")],
            [Event("other", "a")],
            ["alloc"],
        ]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                impl.replay_lifetimes(events)

    def test_trace_container_contract(self):
        for events in (None, "events", iter([])):
            with self.subTest(events=events), self.assertRaises(ValueError):
                impl.replay_lifetimes(events)


if __name__ == "__main__":
    unittest.main()
