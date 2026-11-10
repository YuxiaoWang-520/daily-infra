"""练习一：同目录暂存、完整写入后发布。输入由调用方提前冻结。

完成后将两个函数复制到 checkpoint_io.py，再运行 test_checkpoint_io.py。
"""

from pathlib import Path
from typing import Any


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    """写入失败时保留旧 checkpoint，并清理本次临时文件。"""
    raise NotImplementedError


def load_checkpoint(path: Path) -> dict[str, Any]:
    """在 CPU 上加载 Tensor 与基础类型组成的字典。"""
    raise NotImplementedError
