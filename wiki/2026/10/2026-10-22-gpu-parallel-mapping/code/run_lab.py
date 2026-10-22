"""Print CPU address-model observations, not device measurements."""

import json
from dataclasses import asdict

from answers import launch_map, sector_ids


def main() -> None:
    result = {
        "scope": "CPU address model; no CUDA execution or measured transactions",
        "ten_elements_block4": [asdict(row) for row in launch_map(10, 4)],
        "sectors_32_lanes_fp32": {
            "aligned": len(sector_ids([4 * lane for lane in range(32)])),
            "offset4": len(sector_ids([4 + 4 * lane for lane in range(32)])),
            "stride128": len(sector_ids([128 * lane for lane in range(32)])),
        },
    }
    assert result["sectors_32_lanes_fp32"] == {
        "aligned": 4,
        "offset4": 5,
        "stride128": 32,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
