# 练习与复现

现有解释器为 `/Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python`，Python 3.12.16、PyTorch 2.15.0。两道练习分别验证整数容量模型与真实 CPU／Gloo 的分片更新。全部使用合成数据和现有依赖。

## 练习一　所有权与通信预算

模板为 [exercises.py](exercises.py)，结果类型为 [models.py](models.py)。实现 `state_budget` 和 `ring_budget`，所有数量均用整数计算。

输入参数量 P、分片组人数 N、状态字节宽度均为正整数，拒绝 bool 和浮点数。stage 为 0—3 的整数，拒绝 bool。所有非法配置抛 ValueError。令 `S=ceil(P/N)`，`Q=S×N`。此教学模型将 flat 向量统一补齐，所有已分片的向量状态占 S 个槽位，仍复制的状态占真实 P 个元素。

| stage | 参数元素 | 梯度元素 | 优化器状态所对应的参数元素 |
| --- | --- | --- | --- |
| 0 | P | P | P |
| 1 | P | P | S |
| 2 | P | S | S |
| 3 | S | S | S |

默认参数 2 B、梯度 2 B、优化器状态合计 12 B／参数。最后一项包括 FP32 主副本与两个 FP32 Adam moments。这是论文式容量假设，可以传入其他宽度。输出为各项字节数、总和、S 和 Q，不包含激活、临时物化、通信缓冲、step 标量及 allocator 开销。

手算验收 P=10^9、N=8，四阶段分别为 16、5.5、3.75、2 GB，GB 按 10^9 B。P=5、N=2 时 S=3、Q=6，stage3 每 rank 48 B，stage1 为 10+10+36=56 B。即使 N>P，也保留统一填充槽位。真实 FSDP2 按每参数维度分片，应分别核算逻辑 shape、padding 和实际 storage。

通信题以同一个 padded 长度 Q 为基准，采用理想 ring 的每 rank 发送量。每次 AllGather 或 ReduceScatter 对应 `S×(N−1)` 个元素的发送。参数宽度为 bp，梯度宽度为 bg，AllGather 次数 k 只能为 1 或 2。输出 AG 发送、RS 发送、总发送与对照 DDP 的发送字节。

```text
AG_sent  = k × S × (N−1) × bp
RS_sent  = S × (N−1) × bg
total    = AG_sent + RS_sent
DDP_sent = 2 × S × (N−1) × bg
```

参数在前向后保留到反向时选 k=1，前向后重分片再收集时选 k=2。P=8、N=4、bp=bg=2 时，k=2 总发送 36 B，DDP 为 24 B。改成 bg=4，二者均为 48 B。N=1 时发送量为 0。该估算省略协议、拓扑和启动延迟，也没有计接收量。

完整答案见 [answers.py](answers.py)，手算与边界测试见 [test_answers.py](test_answers.py)。模板测试通过环境变量选择实现。

```bash
PYTHONDONTWRITEBYTECODE=1 EXERCISE_IMPL=exercises /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
```

## 练习二　真实两进程的分片 Adam

模板为 [probe_exercises.py](probe_exercises.py)，答案为 [sharded_probe.py](sharded_probe.py)。完成 `run_probe(steps=3)`，steps 必须为 1—10 的整数，拒绝 bool。验收用固定两个 spawn 进程和 CPU／Gloo，TemporaryDirectory 独占生成 rendezvous 文件与结果文件，结果文件使用 x 模式，进程组在 finally 关闭。

合成模型只有 5 个 FP64 权重，两个 rank 各保存 3 个补齐后的参数槽位。每步先用 AllGather 收集完整参数，在本地样本上计算半平方误差，再用 SUM ReduceScatter 把归约后的梯度分给各自所有者，最后调用本地 torch.optim.Adam。完整临时参数保留经过反向，训练每步只有一次 AG。moments 和权重各自只更新本 rank 的 3 个槽位。

rank0 有第一条样本，rank1 有其余三条。正确 loss 为本地误差和除以全局样本数 4。负对照用本地均值除以 world size 2，故改变了目标。两条路径都与全集样本的 FP64 Adam 参考比较 3 步后的参数、最后梯度、m 和 v。

初始权重为 [0.1,0.2,0.3,0.4,0.5]，特征与标签固定在答案中。手算第一维的首步正确梯度为 `(0.65+0.05)/4=0.175`，错误对照为 `(0.65+0.05/3)/2=1/3`。正确首步完整梯度应为 [0.175,0.4125,0.525,0.7625,0.9375]。

验收覆盖以下结果。

1. 两个真实 rank 正确连接 Gloo，分母为 4，本地样本数分别为 1 和 3。
2. 正确路径参数、梯度、m、v 与参考的最大绝对误差均不超过 1e-12。
3. 错误对照的梯度与 m 差异大于 1e-4，避免只比较 Adam 首步权重而漏掉归一化问题。
4. 每 rank 的参数与两个 moment 向量合计 9 个元素，局部梯度为 3 个元素。padding 在权重、梯度和 moments 中始终为零。
5. 每步训练通信顺序一致，AG 与 RS 各一次。结果检查额外用五次 AG，单独记录，样本计数另有一次 AllReduce。
6. 当前验证使用完整参考副本、结果汇集和同步通信，因此这些向量元素计数描述所有权。真实显存峰值与并发通信需要其他测量。

测试见 [test_sharded_probe.py](test_sharded_probe.py)，组合执行见 [run_lab.py](run_lab.py)。在本篇目录执行以下命令。组合脚本要求一个尚不存在的输出文件，防止覆盖已有结果。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/fsdp-zero-lab-new.json
```

本机正式输出保存在 [lab.json](../evidence/runs/lab.json)，命令与退出码见 [summary.json](../evidence/runs/summary.json)。基础测试验证显式集合通信与局部 Adam 的数学语义，FSDP2 的 hooks、DTensor 与 GPU stream 路径以固定源码分析作为证据。
