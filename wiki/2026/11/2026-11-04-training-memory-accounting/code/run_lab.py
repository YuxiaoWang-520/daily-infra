"""Run two deterministic, hand-checkable payload accounting exercises."""

import json
from dataclasses import asdict

from answers import estimate_state_bytes, replay_lifetimes
from models import Event


def main() -> None:
    layouts = {
        "fp32_adam": estimate_state_bytes(16, 4, 4, 4),
        "explicit_bf16_state": estimate_state_bytes(16, 2, 2, 2),
        "explicit_master_layout": estimate_state_bytes(16, 2, 2, 4, 4),
        "after_zero_grad_none": estimate_state_bytes(16, 4, 4, 4, gradients_live=False),
    }
    events = [
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
    retained = events[:7] + events[8:]
    report = {
        "scope": "synthetic tensor payload, not measured GPU memory",
        "layouts": {
            name: dict(asdict(b), total=b.total) for name, b in layouts.items()
        },
        "billion_fp32_payload_bytes": estimate_state_bytes(10**9, 4, 4, 4).total,
        "normal_lifetime": asdict(replay_lifetimes(events)),
        "retained_view": asdict(replay_lifetimes(retained)),
        "sum_of_component_maxima_bytes": 64 + 64 + 64 + 128 + 64 + 96,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
