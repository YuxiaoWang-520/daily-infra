"""Synthetic timestamps make asynchronous ordering visible on an ordinary CPU."""

import json
from dataclasses import asdict

import torch

from answers import QueueModel


def main() -> None:
    model = QueueModel(0.1)
    first = model.submit("producer", 3.0)
    returned = model.host_ms
    marker = model.event("producer")
    model.wait_event("consumer", marker)
    second = model.submit("consumer", 2.0)
    model.synchronize()
    assert second.start_ms == first.end_ms
    assert abs(model.host_ms - 5.1) < 1e-12
    print(
        json.dumps(
            {
                "scope": "synthetic model, milliseconds; no GPU times measured",
                "cuda_available": torch.cuda.is_available(),
                "host_after_first_submit_ms": returned,
                "producer": asdict(first),
                "consumer": asdict(second),
                "host_after_sync_ms": model.host_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
