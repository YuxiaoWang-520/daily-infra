"""练习一：实现等分块的整数环 SUM，并产生不可变消息 trace。

完成后把函数实现复制到 ring.py 的对应位置，再运行 test_answers.py。
起始文件保留接口；答案文件同时提供 Message 与 RingResult 定义。
"""

from collections.abc import Sequence

from ring import RingResult


def ring_allreduce(
    inputs: Sequence[Sequence[int]], element_bytes: int = 8
) -> RingResult:
    raise NotImplementedError
