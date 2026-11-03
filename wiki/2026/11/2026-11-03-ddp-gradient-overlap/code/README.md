# 练习、答案与运行方法

两道基础练习只依赖 Python 3.10+ 标准库。`ddp_probe.py` 是额外的真实双进程验证，需要已经安装且支持 CPU Gloo 的 PyTorch；不要为运行本篇自动修改全局环境。

## 练习一：平均的是 rank，还是有效样本？

在 [exercises.py](exercises.py) 实现两个函数，独立答案在 [answers.py](answers.py)。

`global_mean_from_local_means(means, counts)`：

- `means` 是非空 list／tuple，每行是一个 rank 的一维平均梯度，也是非空 list／tuple；各行等长。
- 每个梯度分量是有限的 Python int／float，拒绝 bool。`counts` 是等长的正 Python int list／tuple，拒绝 bool、0 和负数。
- `counts[r]` 表示该行均值的有效分母；本题不支持空 rank。可以是样本数，也可以是定义目标时的有效 token 数，但各 rank 的 loss 必须采用同一口径。
- 返回不可变 float tuple：逐分量求 `sum(n_r * g_r) / sum(n_r)`。不修改输入。

`ddp_loss_scales(counts)` 使用相同的计数约束，返回 tuple `R*n_r/N`。将每个 rank 的局部 mean loss 乘以其系数后，再交给普通 DDP 做 rank 平均，才得到指定的全局均值。

不满足契约时抛 `ValueError`。普通浮点舍入仍然存在；这不是分布式 collective、自动求导或完整 loss masking 的实现。

验收例子：局部均值 `[[1], [29/3]]`、计数 `[1,3]`，全局结果应为 `(7.5,)`，缩放系数为 `(0.5,1.5)`；直接平均两个 rank 则是 `16/3`。

## 练习二：推演梯度桶的就绪与通信尾部

实现 `schedule_buckets(ready_ms, sizes_bytes, buckets, startup_ms, bytes_per_ms)`，返回 [models.py](models.py) 的不可变 `Schedule`。

- `ready_ms` 为非空 list／tuple，每行对应一个 rank，每列对应一个梯度的就绪时刻。行也是非空 list／tuple，长度都等于梯度数量；时刻是有限非负 Python int／float，拒绝 bool。
- `sizes_bytes` 为非空正 Python int list／tuple，给出每个梯度的通信字节数，拒绝 bool。
- `buckets` 为非空 list／tuple；每个桶是非空 list／tuple 的整数索引。所有梯度索引必须被恰好覆盖一次，不能重复、遗漏或越界；拒绝 bool。
- `startup_ms` 是有限非负数，`bytes_per_ms` 是有限正数，均拒绝 bool。前者是每次通信的固定成本，后者是本教学模型的有效传输率。
- 第 j 个桶的全局就绪时刻取所有 rank、该桶所有梯度就绪时刻的最大值。一个串行通信通道按给定桶顺序工作，等上一个桶完成才能有效开始下一个；耗时为 `startup_ms + bucket_bytes/bytes_per_ms`。
- 反向完成时刻设为所有梯度就绪时刻的最大值；总完成时刻必须等反向和所有通信都结束。`tail_ms` 为反向结束后剩余的通信时间；`serial_ms` 为完全不重叠时的反向时间加全部通信耗时。
- 不修改输入。无效输入或不可表示的派生时间抛 `ValueError`。

这是明确简化的离散时序模型，不是模拟 NCCL 内部算法。所有 rank 共用对齐的时间起点，计算就绪时间不受通信影响，不模拟争用、流水线传输、并行 collective、copy 或最后梯度之后的其他反向工作；`start_ms` 是模型的有效开始，不是某个 rank 的 enqueue 时间。即使单 rank 输入，模型也照所给公式收费。

验收例子：两 rank 的四个梯度都在 `[2,4,6,8] ms` 就绪，每个 4 MiB，固定成本 2 ms，有效传输率 4 MiB/ms。四个小桶／两个中桶／一个大桶的完成时刻分别为 `14/12/14 ms`。这些是给定输入的计算结果，不是实测延迟。

## 自己做题与检查答案

在本篇目录执行参考答案：

```bash
python -m unittest discover -s code -p test_answers.py -v
python code/run_lab.py
```

写好 `exercises.py` 后：

```bash
EXERCISE_IMPL=exercises python -m unittest discover -s code -p test_answers.py -v
```

测试覆盖不同分母、向量梯度、独立样本对照、输入不变性、桶覆盖、慢 rank、固定顺序阻塞、手算时序与依赖不变量。不要求逐 bit 比较所有浮点数。

## 额外验证：真实 CPU Gloo DDP

已有合适 PyTorch 时运行：

```bash
python code/ddp_probe.py
```

本机 macOS 复验显式指定了回环接口：`GLOO_SOCKET_IFNAME=lo0 python code/ddp_probe.py`。其他主机应使用自身有效接口；`lo0` 不是跨系统通用名称。

程序用 spawn 创建两个进程、以临时 FileStore 会合，不需要手设 MASTER_PORT；会清理自己创建的临时目录和进程组。默认通信行为，没有注册自定义通信 hook。可用 `--output evidence/runs/ddp.json` 显式保存机器可读结果。

模型是 CPU float64 的无 bias 单参数线性层，初始 `w=1`；loss 为 `0.5*mean((w*x)^2)`，SGD 学习率为 0.1。它验证等量样本、不等量样本的错误平均／正确加权，以及 `no_sync()` 累积时有／没有除以两个 micro-batch 的差异；每条检查梯度和更新后的参数，并核对两 rank 一致。

`no_sync()` 包住第一组 forward 和 backward，第二组正常同步；第一组保留的局部梯度不同，最后梯度相同。本实验不计通信次数，不测试 GPU 重叠、速度或真实训练收敛。实际环境、网络选项、结果或失败限制以本篇审查记录和 `evidence/runs/` 为准。
