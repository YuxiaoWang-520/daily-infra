"""Check a hand example and exact read-count models without GPU timing."""

import json

import torch

from answers import blocked_matmul, tile_traffic


def main() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float64)
    b = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    out = blocked_matmul(a, b, 2, 2, 2)
    torch.testing.assert_close(out, a @ b, rtol=0, atol=0)
    small = tile_traffic(64, 64, 64, 1, 1, 8, 4)
    large = tile_traffic(64, 64, 64, 8, 8, 8, 4)
    assert small["read_elements"] == 8 * large["read_elements"]
    print(
        json.dumps(
            {
                "scope": "CPU block algebra and logical load counts; no GPU performance",
                "hand_result": out.tolist(),
                "tile1": small,
                "tile8": large,
                "tile8_bk16": tile_traffic(64, 64, 64, 8, 8, 16, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
