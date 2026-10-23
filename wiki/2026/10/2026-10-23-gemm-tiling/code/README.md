# GEMM 分块练习

先做正文第 8 节。已有 Python 3.10+ 和 PyTorch 即可：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

11 项测试应通过；手算例子输出 `[[4,6],[12,14]]`。FP64 随机比较使用 `1e-12` 相对与绝对容差，FP32 为 `1e-5`，仅用于这些固定小规模测试，不是所有输入的通用误差保证。

[blocked_matmul](answers.py) 为 CPU 推理参考，内部每块调用 CPU `@`，不提供梯度。支持奇数边界、空维度与非连续 strided 输入，不修改输入。没有实现 CUDA 地址、同步、片上布局或矩阵指令。

`tile_traffic` 只统计有效逻辑读取和状态元素：无跨输出块缓存复用，无边缘填充槽位，无 CPU 库内部 packing／临时量。不能把计数当实际 DRAM 流量或分配器峰值。结果和环境见 [evidence/runs/](../evidence/runs/)。
