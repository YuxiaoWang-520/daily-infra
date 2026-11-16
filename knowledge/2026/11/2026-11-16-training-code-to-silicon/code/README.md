# 训练一步的 CPU 追踪

例子固定为 `X(2×3)`、`W(3×4)`、`b(4)`、ReLU、MSE 和 SGD，全部输入为自编合成值。`X` 开启梯度，以展示向更早层传递的输入梯度。

| 文件 | 内容 |
| --- | --- |
| [example_data.py](example_data.py) | 固定合成输入和学习率 |
| [reference.py](reference.py) | 标量 Python 前向、解析梯度和更新 |
| [walkthrough.py](walkthrough.py) | CPU PyTorch、autograd 节点、CPU 操作与 FX 前向输出 |
| [test_walkthrough.py](test_walkthrough.py) | 已知数值、22 值有限差分、FP32／FP64、三次 mm 和 FX 验证 |

在当前已具备 PyTorch 的 Python 环境中，从本篇目录执行。

```bash
PYTHONDONTWRITEBYTECODE=1 python code/walkthrough.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_walkthrough.py -v
```

演示输出为 JSON，包含软件版本、两种 dtype 的结果、CPU 操作名／次数／输入形状和 FX 前向代码。profiler 范围包括一次前向、反向、梯度观测与 SGD，更新后的验证前向在范围之外。输出省略了时间、调用栈和原始 Chrome trace，所见 `aten::mm` 是 CPU 框架操作事件。

标量结果为 `loss_before=7.25`、`loss_after=3.725390625`。浮点实现按测试中声明的容差比较。普通和 `-O` 执行均已通过全部 `8` 项测试，实际结果及命令见附带记录。[真实运行](../evidence/runs/verification.json)、[演示结果](../evidence/runs/03-demo.json)

Ruff 检查需要当前环境已有 Ruff。

```bash
python -m ruff check code
python -m ruff format --check code
```

这里的标量函数服务于固定样例，外部动态输入的 shape／dtype 校验不属于该示例接口。GPU 设备执行的逐层分析与芯片研发路线见 [正文](../README.md)。
