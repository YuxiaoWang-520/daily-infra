# Tensor Core 前置练习

这两题不需要 GPU：一题数人为补齐的代价，一题观察先存为 FP16／BF16 再用 CPU FP32 计算。已有 Python 3.10+、PyTorch 即可。

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

预期 9 项通过，补齐 `(17,9,17)` 到 `(32,16,32)`，FLOPs 从 5202 变为 32768；1.001 的两种存储结果分别为 1.0009765625 和 1.0。

[完整答案](answers.py) 为推理参考，只接收 CPU FP32／64 输入。拒绝非有限值或存储转换溢出，但 FP32 累加仍可溢出。返回连续克隆输出，不保留补齐输出的大存储视图。它不模拟 Tensor Core 的乘积精度、累加顺序、指令、线程布局或速度，tile 参数也不是通用准入规则。[实际记录](../evidence/runs/)。
