"""Run both answers and save explicitly labeled evidence artifacts."""

import argparse
import json
from pathlib import Path

from answers import complete_events, overlap_summary, synthetic_trace
from cpu_profile import run_cpu_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = synthetic_trace()
    (args.output_dir / "synthetic_gpu_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n"
    )
    events = complete_events(trace)
    compute = [(e.start_us, e.end_us) for e in events if e.name.startswith("compute_")]
    comm = [(e.start_us, e.end_us) for e in events if e.name == "communication"]
    result = {
        "synthetic_gpu": overlap_summary(compute, comm, (0, 15000)),
        "real_cpu": run_cpu_profile(args.output_dir),
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
