# 训练内存练习与 CPU 验证

两道练习用标准库建立“明确布局下的字节账本”和“唯一 storage 的生命周期”。`payload_probe.py` 另用本机已有 PyTorch 验证实际参数、梯度、Adam 状态和别名；它们不测 GPU 峰值。

## 运行

在本篇目录执行（`python3` 指向已有 Python 3.10+）：

```bash
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/run_lab.py
python3 code/payload_probe.py
```

先完成 [exercises.py](exercises.py)，再通过同一组测试：

```bash
EXERCISE_IMPL=exercises python3 -m unittest discover -s code -p 'test_*.py' -v
```

起始代码故意抛 `NotImplementedError`。参考答案在 [answers.py](answers.py)，不可变数据结构在 [models.py](models.py)。单元测试和 lab 只依赖标准库；CPU probe 需要已有的 PyTorch，验证环境是 2.9.1，不要求安装当日主分支。

## 练习一契约

`estimate_state_bytes(numel, parameter_bytes, gradient_bytes, moment_bytes, master_bytes=0, *, gradients_live=True, state_initialized=True, amsgrad=False)` 返回 `StateBytes`。

- `numel` 是一组同 dtype、无共享 storage 的可训练参数的元素数，为非负 Python `int`；零元素返回全零。`bool` 不视作整数。
- 参数、梯度、moment 的单元素字节数均为正 Python `int`；两个 moment 使用同一宽度。`master_bytes` 为非负 Python `int`，0 表示没有额外 master copy。此参数是调用者明确指定的布局，不由函数推断 AMP 行为。
- 三个开关必须是 `bool`。梯度不存活时梯度项为 0；状态未初始化时两个 moment 及 AMSGrad 项为 0。显式指定的 master copy 不依赖 Adam 的延迟初始化。
- 结果逐项为元素数乘宽度；AMSGrad 在状态初始化后多一份 moment 宽度的数组。`total` 为六项之和。全程使用整数，不做浮点容量换算。
- 非法输入抛 `ValueError`，不修改输入。返回 frozen dataclass。

只计张量有效载荷，不计 step 标量、对齐、激活、临时张量、allocator cache、外部库或通信；不推断冻结、分片、压缩、稀疏格式。异构 dtype 或部分参数才有状态时，应逐组计算后在不重复存储的前提下合并。

## 练习二契约

`replay_lifetimes(events)` 只接收由 `Event` 组成的 `list` 或 `tuple`，返回 `PeakReport`。事件序号从 0 开始，每个点记录该事件执行后的状态。

| 操作 | 输入约束 | 动作 |
| --- | --- | --- |
| `alloc` | 当前未被使用的非空字符串名字，正整数 `nbytes`，`source=None` | 建立一块新的 storage，大小为 `nbytes` |
| `alias` | 新名字、已经存活的 `source` 名字，`nbytes=0` | 增加同一块 storage 的引用，不分配载荷 |
| `drop` | 已经存活的名字，`nbytes=0`、`source=None` | 删除这个引用，最后一个引用消失才释放载荷 |

空白名字、布尔／非整数大小、重复存活名字、无效 source、未知 op、非 Event 项或其他容器类型均抛 `ValueError`。已经删除的名字可以重新用于分配，但它必须获得新的 storage 身份，不能与旧别名混淆。`source` 可以是另一个别名。

每块存活 storage 只计一次。峰值并列时记录第一次出现的位置；空 trace 的峰值与最终载荷为 0、峰值位置为 `None`。输入保持不变，结果及点是 frozen dataclass，点集合是 tuple。模型假设 drop 后最后引用立即释放，不模拟 Python GC、stream 延迟释放、CUDA 分配器和真实 kernel。

## 真实 CPU probe 的边界

- 三种配置均使用 CPU `Linear(4,4,bias=False)`，16 个参数元素，固定权重、固定输入、Adam `foreach=False, fused=False, capturable=False`。
- 分别检查 FP32、FP32 参数配合 BF16 autocast、显式 BF16 参数。保留输出与 loss 的 Python 引用，但账本仅列参数、梯度、moment 和 step；它不是整个程序的内存总和。
- 四个采样点是构造后、backward 后、首次 step 后、`zero_grad(set_to_none=True)` 后。每个配置断言状态键、dtype、step 标量及载荷。FP32 两组分别为 64／128／260／196 bytes；显式 BF16 为 32／64／132／100 bytes。比练习的 256／128 bytes 多出的 4 bytes 是单个参数张量对应的 step 标量。
- 别名探针验证：base 的 16 个 FP32 元素、前 8 个元素的 view、view 的 clone，逻辑字节数之和为 128，唯一 storage 总字节数为 96。只保留 view 时，它仍持有原来 64 bytes 的完整 storage。
- 另一探针验证没有梯度的参数在首次 step 后没有新建 Adam 状态，同时用 FP64 标量核对梯度 2 对应的 `m≈0.2、v≈0.004` 与偏差修正后的参数更新。
- 内存地址只在函数内部用于判定共享 storage，不写入输出。不测进程 RSS、不调用 CUDA allocator，不验证 CUDA kernel、当前 main 二进制、数值收敛或速度。显式 BF16 状态只作存储机制演示，不是稳定训练方案的建议。

[实际运行记录](../evidence/runs/summary.json) · [返回正文](../README.md)
