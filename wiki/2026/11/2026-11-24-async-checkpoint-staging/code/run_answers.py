"""Run the timing model and real CPU checkpoint cases."""

import argparse
import json
from pathlib import Path

from answers import simulate_checkpoints
from dcp_demo import probe_failed_staging, run_checkpoint_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "fast_storage_model": simulate_checkpoints(300, 100, 100, 2000, 6000),
        "slow_storage_model": simulate_checkpoints(300, 100, 100, 2000, 16000),
        "blocking_staging_cpu": run_checkpoint_case(async_staging=False),
        "async_staging_cpu": run_checkpoint_case(),
        "write_failure_cpu": run_checkpoint_case(fail_write=True),
        "already_failed_staging_probe": probe_failed_staging(),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output / "results.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
