"""练习二：累计共享物理资源上的同轮流量，并估算容量时间。

完成后把函数实现复制到 costs.py 的对应位置，再运行 test_answers.py。
核对 flat、跨节点相邻和交错 rank 三种情景。
"""

from collections.abc import Mapping, Sequence

from costs import Cost


def ring_cost(
    payload_bytes: int,
    order: Sequence[int],
    routes: Mapping[tuple[int, int], tuple[str, ...]],
    bandwidth_bytes_per_second: Mapping[str, float],
    alpha_seconds: float,
) -> Cost:
    raise NotImplementedError
