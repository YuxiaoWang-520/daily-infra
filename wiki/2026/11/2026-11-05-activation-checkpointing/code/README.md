# 练习与运行

两道题分别回答“粗略预算怎么做”和“重算是否改变训练结果”。先读题目，再按需查看答案。

## 环境与命令

容量题只用 Python 标准库；实验题使用现有 PyTorch。本次环境为 Mac mini、Python 3.12.16、PyTorch 2.15.0、CPU FP64。没有安装或升级依赖，没有 GPU 性能结果。

在本篇目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py
```

其他机器可把绝对路径替换为已有环境的 `python`；不要执行旧文章的历史 MacBook 环境路径。

## 练习一：用分段模型做预算

在 [exercises.py](exercises.py) 实现 `estimate_uniform` 和 `choose_plan`。`n=layers` 层构成均匀链，`s=segment_size` 层一段，单份激活大小为 `A=activation_bytes`。保存的边界份数近似为 `ceil(n/s)`，重算一段最多再留 `s` 份；按 `(ceil(n/s)+s)A` 做规划。它故意不扣共享边界，也不模拟逐事件释放，端点与其他成本另计，不能当成真实显存峰值。

`forward_us=F`、`backward_us=B` 均为整条链的假设耗时，`overhead_us=h` 是每段的额外开销。本题假定每层完整重算一次，原耗时 `F+B`，新耗时 `2F+B+ceil(n/s)h`。不模拟 PyTorch early stop。

输入：正整数 `n`、`s`、`A`，且 `s<=n`；`F` 有限且为正，`B`、`h` 有限且非负。禁止把布尔值当整数。合成时间超出浮点数范围时报 `ValueError`。预算是非负整数字节数，只覆盖上述激活模型。

输出：`SegmentPlan` 的段数、激活份数、字节数、原耗时和新耗时，以及 `slowdown` 比值。选择函数枚举 `s=1..n`，在预算内先最小化耗时，平局时先少用字节、再选较小的 `s`；无可行方案返回 `None`。枚举为 `O(n)` 时间、`O(1)` 额外空间，仅用于小规模教学。

验收例：`n=16`、`A=1 MiB`、`F=16 μs`、`B=32 μs`、`h=0.2 μs`，取 `s=4` 得到 4 段、8 份、8 MiB、48 μs 与 64.8 μs，时间比为 1.35。预算 10 MiB 时选 `s=8`，估算 64.4 μs；预算 7 MiB 时不可行。`n=17,s=4` 必须向上取整为 5 段。

独立测试：[test_answers.py](test_answers.py)。答案：[answers.py](answers.py)。练习者可只运行容量测试：

```bash
PYTHONDONTWRITEBYTECODE=1 EXERCISE_IMPL=exercises /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
```

未完成的模板会按预期失败。`s=1` 或 `s=n` 时，此粗略模型甚至给出 `n+1` 份，说明切段方式并不保证节省。

## 练习二：重算后，梯度真的相同吗？

在 [exercise_checkpoint.py](exercise_checkpoint.py) 实现 `run_trial`。创建 4 个顺序 block，每个为 `Linear(8,8) → sin → Dropout(p=0.4)`，使用 FP64 CPU。输入是 `linspace(-0.9,0.9,16).reshape(2,8)`，并要求输入梯度。用 CPU seed 2026 初始化每次的新模型；loss 为输出平方的平均，执行一次 `SGD(lr=0.01)`。

三次运行必须有相同初始参数与随机状态：普通执行、逐 block checkpoint 并保存 RNG、逐 block checkpoint 且关闭 RNG 保存。checkpoint 显式指定 `use_reentrant=False`；使用默认 early stop，不编译模型。

返回对象须有 `output`、`loss`、`input_gradient`、`parameter_gradients`、`updated_parameters`、`next_random`、`saved_activation_bytes`、`forward_entry_calls` 字段。梯度与更新后的参数都返回独立的 detached clone，参数顺序与 `model.parameters()` 一致。实验结束恢复调用者的 CPU RNG；不要通过全局 `manual_seed` 改动其他设备的 RNG。

在 **模型初次 forward** 外围设置 `saved_tensors_hooks`。pack 时只记录 storage 的设备、地址和字节数，按 storage 去重；排除参数、原始输入以及零字节 storage。pack 返回 detached alias，不额外复制载荷，也不要在统计列表里持有 Tensor。loss 与 backward 不在统计上下文中。嵌套时只应用最内层 hooks，所以这里观察的是外层可见的保存，无法看见 checkpoint 的全部内部存储。

验收：

1. 保存 RNG 时，输出、输入梯度、全部 8 个参数张量的梯度、一次 SGD 后的参数与基线匹配；梯度和更新值采用 `rtol=atol=1e-12`，输出要求完全相同，之后的 CPU 随机样本也应相同。
2. 关闭 RNG 保存时，本次初始 forward 输出仍相同，但至少一份参数梯度不同，之后的 CPU 随机样本不同。这个预期反例用来发现“只比较 loss”的错误验收。
3. 普通执行的 block 入口序列为 `0,1,2,3`，checkpoint 的序列为 `0,1,2,3,3,2,1,0`。入口计数不能证明每次都执行完 block。
4. checkpoint 的外层 forward 激活 storage 统计更小；只有一个 block 时，排除原始输入后该统计为 0。这个 0 绝不表示实际训练没有内存占用。
5. `dropout_p=0` 时，即使关闭 RNG 保存，参数梯度也匹配。处理非法维度、概率、seed 和布尔配置。

独立测试：[test_torch_probe.py](test_torch_probe.py)。完整答案：[torch_probe.py](torch_probe.py)。组合运行与 JSON 输出：[run_lab.py](run_lab.py)。试题模板可通过 `CHECKPOINT_IMPL=exercise_checkpoint` 接受同一套测试。

## 解释结果时的边界

外层 hooks 的 storage 统计不包括参数、输入、loss 保存值、RNG 快照、图元数据、局部重算 workspace、分配器缓存和其他设备内存；也没有跟踪释放时刻。它只能证明这个固定 CPU eager 实验中外层保存载荷的变化，不能代替 CUDA `max_memory_allocated`，不能推广为显存下降比例。

没有测量训练吞吐或 GPU 显存。本题的微秒数是输入假设，不是 Mac mini 上测得的性能。真实性能测试需要固定 batch、序列长度、模型、dtype、编译与分布式配置，预热、同步后重复测量完整训练步，并同时核对梯度和全步峰值。
