# 浮点格式练习

先做正文第 8 节两道题，再查看 [完整答案](answers.py)。

- `roundtrip`：观察输入存储舍入。
- `sequential_sum`：指定每一步累加的 dtype，区分输入与累加误差。
- `compare_finite`：为小规模 CPU 验收提供明确、精确的容差判定与误差报告。

在本目录运行：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

需要已有 Python 3.10+、PyTorch。实测 Python 3.10.13、PyTorch 2.9.1 CPU；19 项测试应全部通过。测试与诊断输出存放在 [evidence/runs/](../evidence/runs/)。

诊断应包含：FP16 `1.001` 读回为 `1.0009765625`，BF16 为 `1.0`；FP16 `70000` 溢出，严格 JSON 用字符串 `inf` 表示；1000 个 FP16 `0.1` 的量化输入参考和为 `99.9755859375`，逐步 FP16 累加为 `105.1875`，本例逐步 FP32 累加与参考相同。

`sequential_sum` 是刻意逐步舍入的教学循环，不代表 `torch.sum` 的内部实现。`compare_finite` 使用标准库精确分数和独立 Decimal 上下文，适合小输入参考，速度并非目标；返回误差报告，不提供梯度，不支持 GPU 输入。没有运行 GPU 或真实混合精度训练。
