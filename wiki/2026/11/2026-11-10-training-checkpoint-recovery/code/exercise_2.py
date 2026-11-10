"""练习二：针对固定 CPU 训练配置完成状态验证与恢复。

将答案复制到 resume_probe.py 的对应函数后，运行 test_resume_probe.py。
保留已经给出的 Trainer、数据集和五种进程启动方式。
"""

from typing import Any

from resume_probe import Trainer


def restore(payload: dict[str, Any], omit: str = "none") -> Trainer:
    """先验证进度与顺序，再恢复各状态；最后恢复 RNG。

    omit 可为 none、optimizer、rng、cursor、scheduler，后四项是错误对照。
    """
    raise NotImplementedError
