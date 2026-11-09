---
title: "FSDP／ZeRO 怎样分片参数、梯度与优化器状态？"
date: "2026-11-09"
timezone: "Asia/Shanghai"
status: published
topics: [FSDP2, ZeRO, 状态分片, AllGather, ReduceScatter]
prerequisites: ["17 梯度通信", "18 训练内存", "19 激活重算"]
source_checked_at: "2026-11-09T11:17:35.354881+08:00"
---

# FSDP／ZeRO 怎样分片参数、梯度与优化器状态？

训练时，每个数据并行进程都会使用模型，但其中一些状态可以由不同进程分别负责。ZeRO 逐步分片优化器状态、梯度与参数，FSDP2 则通过参数分片、按需收集和梯度归约，把这类所有权安排落到执行流程中。节省的容量需要结合完整参数的临时物化、通信和激活来评估。

本篇回答一个工程问题。当训练状态装不进单卡时，怎样让多个 rank 分担，又保持一次全局训练更新的数学含义？rank 是通信组中的进程编号，常见 GPU 训练采用一进程一卡。本文 N 专指参与这一份状态分片的组人数。

前置可阅读 [第 17 篇梯度通信](../2026-11-03-ddp-gradient-overlap/README.md)、[第 18 篇训练内存](../2026-11-04-training-memory-accounting/README.md) 与 [第 19 篇激活重算](../2026-11-05-activation-checkpointing/README.md)。

## 1. 先让 4 个进程分担 8 个参数

假设模型有 8 个参数 p0—p7，4 个 rank 使用不同训练样本。普通数据并行中，每个 rank 都保存完整参数、梯度和优化器状态。前向计算使用本地样本，随后同步梯度，各自完成相同更新。

Adam 优化器为每个参数维护两个随训练推进的向量状态。m 平滑梯度，v 平滑梯度的平方。若每个 rank 只负责其中 2 个参数的更新，它只需保存这 2 个参数对应的 m、v，以及所需的主参数副本。

| rank | 负责更新的参数及状态 |
| --- | --- |
| 0 | p0、p1 |
| 1 | p2、p3 |
| 2 | p4、p5 |
| 3 | p6、p7 |

各 rank 仍要让自己的样本经过模型，因此需要在相应计算阶段获得所需的参数。分片安排同时决定了长期保存什么、计算前收集什么，以及梯度产生后送给谁。ZeRO 原始论文的第 5 节按这三个层次展开。[ZeRO 原始论文 v3](https://arxiv.org/abs/1910.02054v3)

### 为什么要先归约梯度，再由所有者更新？

Adam 的简化更新为以下形式。g 是同步后、对应全局目标的梯度，t 是更新步数，β1、β2 是平滑系数。

```text
m_t = β1 × m_(t−1) + (1−β1) × g
v_t = β2 × v_(t−1) + (1−β2) × g²
m_hat = m_t / (1−β1^t)
v_hat = v_t / (1−β2^t)
p_t = p_(t−1) − lr × m_hat / (sqrt(v_hat) + eps)
```

平方与开方会影响等价关系。例如两份梯度是 1 和 3，全局平均梯度为 2，它的平方是 4，而两份局部平方的平均是 5。各自用局部梯度更新后再平均状态，会改变 v 的演化。正确的分片更新要先形成全局梯度，再让状态所有者执行原优化器规则。练习二用真实 Adam 同时比较参数、m 和 v。[Adam 官方算法说明](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)

## 2. 三个阶段究竟省下多少？

沿用论文中的一种混合精度假设，每个参数有 2 B 计算参数、2 B 梯度，以及合计 12 B 的优化器相关状态。12 B 来自一个 FP32 主参数副本和两个 FP32 Adam moments。于是完整模型状态为 16 B／参数。这是特定存储假设，实际 dtype 和优化器实现需要逐项核验。

当参数总数 P 能均匀分给 N 个 rank 时，理想每 rank 模型状态容量为以下数量。

| 安排 | 参数 | 梯度 | 优化器相关状态 | 合计 |
| --- | --- | --- | --- | --- |
| 普通数据并行，记作 stage0 | 2P | 2P | 12P | 16P |
| ZeRO stage1 | 2P | 2P | 12P/N | 4P+12P/N |
| ZeRO stage2 | 2P | 2P/N | 12P/N | 2P+14P/N |
| ZeRO stage3 | 2P/N | 2P/N | 12P/N | 16P/N |

单位均为字节。P=10^9、N=8 时，对应 16、5.5、3.75、2 GB，GB 使用十进制 10^9 B。stage1 的参数和梯度仍是完整副本，stage2 继续保留完整参数，stage3 才把三类向量状态都分摊。论文第 5 节与图 1 的同一假设可按这些式子复算。[ZeRO 论文第 5 节](https://arxiv.org/pdf/1910.02054v3#page=10)梯度等状态的存活区间会随 zero_grad 等操作变化，这份容量账需要结合实际生命周期使用。

### P 除不尽 N 时怎么办？

练习一采用一个明确的教学布局。把所有参数看成 flat 向量，每份占 `S=ceil(P/N)` 个槽位，总长补齐到 `Q=N×S`。已分片状态都按 S 核算，仍复制的状态按真实 P 核算。P=5、N=2 时，每 rank 有 3 个槽位，stage3 为 `3×16=48 B`。

FSDP2 的实际布局按参数 Tensor 分片，默认沿 dim0 表达为 DTensor。DTensor 表示跨设备的逻辑 Tensor 及其分布方式。比如 shape 为 (3,8) 的参数分给两份，逻辑形状会是 (2,8) 与 (1,8)，内部还会处理通信 padding。因此实际布局需要按每个参数的 shape、局部分片和 storage 计算。[固定源码的参数分片与补齐](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L279-L408)

## 3. 参数分片以后，一次前向和反向怎么执行？

AllGather 把各 rank 的参数分片收集起来，使每个参与者获得这一组完整参数。ReduceScatter 则先把各 rank 的同位置梯度相加或按约定归约，再把结果的不同分片交给各自的所有者。

下面画出一个参数组在“前向后重新分片”条件下的生命周期。图中的释放表示相应完整参数副本可以结束生命周期，实际显存回收还受引用和 stream 使用约束。

```mermaid
flowchart LR
  A["长期状态<br/>参数分片 + 本地优化器状态"] --> B["前向前 AllGather<br/>物化本组完整参数"]
  B --> C["本地样本前向<br/>保存所需激活"]
  C --> D["前向后 reshard<br/>释放完整参数副本"]
  D --> E["反向前 AllGather<br/>再次物化完整参数"]
  E --> F["反向产生本地梯度"]
  F --> G["ReduceScatter<br/>交付归约后的梯度分片"]
  G --> H["各 rank 本地 Adam<br/>只更新自己负责的状态"]
  H --> A
```

完整图源见 [state-lifecycle.mmd](assets/state-lifecycle.mmd)。当前 PyTorch FSDP2 的 pre／post hooks 和参数组动作对应这条路径，是否在前向后释放由 `reshard_after_forward` 控制。[参数组执行源码](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L627-L724)

### 如果前向后保留完整参数呢？

设 `reshard_after_forward=False`，完整参数保留经过反向，通常可省去反向前的那次收集，同时延长完整副本的存活时间。这是通信量和容量之间的一项可配置取舍。

本次核验的 main 以及稳定 v2.15.0 使用 `None` 作为自动选择。初始化先按会重新分片准备，第一次执行时的延迟初始化（lazy init）识别真正的根模块后，为自动策略下的根模块保留前向后的完整参数。子模块默认重分片，显式传入 True 或 False 则遵循指定策略。分析默认行为时要核验模块层级和实际版本。[入口与自动策略](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L260-L308)、[根模块 lazy init](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L179-L232)

## 4. 一步一步追踪当前 FSDP2 源码

本篇实现分析固定官方 main 提交 `cbee6335ae15a556794e2552c553f5befda84516`，提交时间为 2026-11-09。稳定发布 v2.15.0 的提交为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`，本机安装包为 PyTorch 2.15.0。版本、差异与来源校验见 [versions.json](evidence/versions.json) 和 [source-manifest.json](evidence/source-manifest.json)。

### 4.1 谁把分片参数交给优化器？

`fully_shard` 解析 DeviceMesh，也就是进程和设备的分组，初始化参数组并在模块上安装 FSDP 行为。`FSDPParam._init_sharded_param` 创建参数的分片 DTensor，然后将模块的参数属性指向它。优化器需要在这些参数完成分片配置后，使用对应参数构造。底层长期分片参数与临时完整参数由实现管理切换。[FSDP2 入口](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L260-L325)、[参数注册](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L375-L408)

### 4.2 谁触发收集和释放？

`FSDPState` 注册 forward pre-hook 与 forward hook。`FSDPParamGroup.pre_forward` 调用 `unshard`、`wait_for_unshard`，把收集好的完整参数供前向使用。`post_forward` 调用 `reshard`，其实际动作依赖前向后策略。`pre_backward` 再请求参数，提前发起后续参数的收集称为预取，已保留或预取好的状态会复用。[hooks 注册](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_state.py#L118-L145)、[参数状态切换](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L580-L688)

`FSDPParam.to_sharded` 将模块指回分片参数，并在相应条件下释放完整参数存储。`to_unsharded` 将模块指向物化后的完整参数。一次 gather 完成还涉及数据 copy-out、事件和 stream 依赖，源码按这些生命周期组织临时缓冲。[参数切换方法](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L1029-L1035)、[完整参数切换](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L1122-L1142)

### 4.3 梯度归约怎么保持平均语义？

`post_backward` 先取出 autograd 梯度，再执行重分片和 `foreach_reduce`。该函数根据 reduction dtype、分组和 divide factor 选择归约操作与前后缩放。默认 collective 实现最终调用 `all_gather_single` 和 `reduce_scatter_single`。高层的“平均梯度”可能通过 AVG，也可能通过 SUM 配合除法实现。[反向路径](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L688-L835)、[归约与缩放](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L648-L746)

数据分片也必须与目标函数一致。各 rank 样本数相等时，局部均值再按 rank 平均等价于全体样本平均。样本数不等时，要核对有效样本或 token 总分母。练习二明确使用 SUM ReduceScatter，让每个 rank 的本地 loss sum 先除以全局样本数，结果得到全局样本平均梯度。[归约因子实现](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L1000-L1064)

## 5. 额外通信究竟有多少？

用一个独立小例子计算。P=8、N=4，参数和梯度都为 2 B／元素。理想 ring 中，每次 AllGather 或 ReduceScatter 每 rank 发送 `(N−1)×P/N=6` 个元素，也就是 12 B。

- DDP 的梯度 AllReduce 可以按 RS 加 AG 分解，此模型下发送 24 B。
- 前向后重分片的参数组需要两次参数 AG 和一次梯度 RS，发送 36 B。
- 保留完整参数经过反向时，一次参数 AG 加一次梯度 RS，发送 24 B。

若统一按参数宽度 bp、梯度宽度 bg、`f=(N−1)/N` 表示，理想发送字节为以下形式。

```text
DDP                         = 2 × f × P × bg
前向后重分片，再收集          = f × P × (2bp + bg)
前向后保留                  = f × P × (bp + bg)
```

补齐布局时把 P 替换为 Q。这里计的是每 rank 发送量，接收量需要另计。论文第 7 节在统一元素宽度和带宽主导条件下比较 2Ψ 与 3Ψ。上面保留 N 和 dtype 的推导把这个条件展开，实际协议、拓扑、延迟、预取和调度仍会影响时间。[ZeRO 论文通信分析](https://arxiv.org/pdf/1910.02054v3#page=13)

例如把梯度改为 4 B，而参数仍为 2 B，两次参数 AG 加一次梯度 RS 的估算与 DDP 都是 `8fP` B。通信量的比值需要明确 dtype 和具体路径。梯度累积、重算、多次调用参数组与跨节点复制也可能改变每步 collective 的次数。

## 6. 为什么常驻状态除以 N 后仍可能 OOM？

显存峰值需要按同一时刻的存活对象核算。第 2 节只描述模型状态的分摊，实际还需同时看完整参数物化、激活、归约缓冲、工作区、padding 和 allocator 情况。

如果只把整个模型作为一个大参数组，收集时会物化整个模型。按合适的 block 分组可以缩小一次完整物化的范围，也为通信和计算重叠创造调度窗口。但更多小组会增加 collective 启动次数，过细粒度可能使延迟和调度成本主导。预取更多后续组可以提前准备数据，同时会延长或重叠临时缓冲的存活期。[参数组 AllGather 分配](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L342-L400)

| 观察与约束 | 适合评估的改变 | 需要承担的代价和验证 |
| --- | --- | --- |
| 优化器状态占大头，完整参数仍可容纳 | 先评估状态分片方案 | 核对 optimizer、梯度布局和每步更新等价性 |
| 参数副本本身成为容量约束 | 完整状态分片与按组收集 | 参数通信、临时完整副本及组大小 |
| 通信暴露在关键路径，容量有余量 | 保留某些组的完整参数，调整预取 | 临时参数活得更久，峰值会上升 |
| 激活仍是主要占用 | 结合第 19 篇的激活重算 | 额外计算，以及与参数收集调度的共同影响 |
| 多节点链路慢于节点内互联 | 评估分片组与复制组的拓扑安排 | 组间状态复制、额外归约与更复杂的配置 |

排障时先检查每组参数数量、dtype、生命周期和实际显存曲线，再看 collective 的时间线及其与计算的重叠。涉及混合分片的配置时，N 应取相应分片组人数。进程数、所有状态的分摊比例和真实带宽需要分别核对。

## 7. 两道练习与详解

### 练习一　整数容量与理想 ring 发送量

在 [exercises.py](code/exercises.py) 实现 `state_budget` 和 `ring_budget`。输入 P、N、stage 和状态字节宽度，输出各类状态字节、统一分片长度和总补齐长度。通信函数另接收 1 或 2 次参数收集，报告 AG、RS、总发送和 DDP 对照。

输入均按整数契约验证，拒绝 bool、非正数量与不支持的 stage／收集次数。默认宽度为 2／2／12，P=10^9、N=8 的状态总量应是 16、5.5、3.75、2 GB。通信题用 P=8、N=4 复算第 5 节。完整契约与独立题目见 [练习说明](code/README.md)。

<details>
<summary>详解与参考答案</summary>

先用整数 `(P+N−1)//N` 算出 S，避免大整数经浮点除法失真。根据 stage 为每类状态选择 P 或 S，再乘以字节宽度。通信题每次发送元素数为 `S×(N−1)`，分别乘参数和梯度的宽度。

边界要覆盖 P 不能均分、N 大于 P、N=1 和很大的整数。不同 dtype 的通信题检验了固定倍率结论的前提。完整答案见 [answers.py](code/answers.py)，手算断言见 [test_answers.py](code/test_answers.py)。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
```

</details>

### 练习二　只更新本地状态，能否得到同样的 Adam？

在 [probe_exercises.py](code/probe_exercises.py) 完成两个真实 CPU／Gloo 进程的分片训练。模型有 5 个 FP64 权重，补齐后两个 rank 各持 3 个槽位。每步 AG 参数、计算局部 loss、SUM RS 梯度，再对本地分片做 Adam，运行 3 步。

rank0 使用 1 个样本，rank1 使用 3 个样本。正确路径将局部 loss sum 除以全局样本数 4。再故意使用局部均值除以 rank 数 2，作为错误对照。对照全集数据参考，比较参数、梯度、m、v，检查 padding 和训练通信顺序。

完整输入、数据、输出字段与验收标准见 [练习说明](code/README.md)。正确路径误差应不超过 1e-12，错误对照的梯度与 m 应出现可观测差异。

<details>
<summary>详解与参考答案</summary>

权重收集得到完整临时向量，取前 5 个有效元素参与 autograd。每 rank 生成完整本地梯度，补零到 6 个元素后做 SUM ReduceScatter，每份结果只交给相应本地 Adam。

首步第一维的两份未归一化梯度为 0.65 和 0.05，因此正确结果为 `(0.65+0.05)/4=0.175`。错误路径为 `(0.65+0.05/3)/2=1/3`。Adam 的首步权重可能主要反映梯度符号，所以验收同时比较 moments，并连续运行 3 步。

当前教学调度保留完整临时参数经过反向，每步执行一次参数 AG 和一次梯度 RS。FSDP2 的 hooks、DTensor 以及前向后释放再收集的路径由固定源码分析核验。实验额外保存参考模型并汇集结果，用于正确性对照。

完整实现见 [sharded_probe.py](code/sharded_probe.py)，验收见 [test_sharded_probe.py](code/test_sharded_probe.py)，组合脚本见 [run_lab.py](code/run_lab.py)。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/fsdp-zero-lab-new.json
```

</details>

## 8. 本机实际验证结果

本机为 Mac mini，macOS 27 arm64，Python 3.12.16、PyTorch 2.15.0。实验使用两个 spawn 的 CPU／Gloo 进程，FP64，5 个参数、4 个合成样本、3 次 Adam 更新，lr=0.01、β=(0.9,0.999)、eps=1e-8。

| 3 步后与全局参考的最大绝对差 | 正确样本分母 | 错误 rank 均值对照 |
| --- | --- | --- |
| 参数 | 0 | 4.8641×10^-5 |
| 梯度 | 1.1102×10^-16 | 0.24417 |
| m | 2.7756×10^-17 | 0.067430 |
| v | 2.1684×10^-19 | 9.3056×10^-4 |

两个 rank 重建的结果一致。各 rank 的参数与两个 moment 向量合计 9 个元素，梯度分片为 3 个元素，padding 始终为零。step 计数标量、完整参考和检查输出另占空间，实际内存峰值需要按存活对象测量。

本篇 24 项测试、Ruff 检查和格式检查通过。实验结果支撑分片更新的 CPU 数学等价性与负对照。GPU 显存、NCCL 通信性能和 FSDP2 整体训练路径需要对应设备上的运行证据。原始报告见 [lab.json](evidence/runs/lab.json)，实际命令与退出码见 [summary.json](evidence/runs/summary.json)。

## 9. 面试官继续追问

| 层次 | 追问 | 参考回答要点 |
| --- | --- | --- |
| 基础 | stage1、2、3 分别把谁分给不同 rank？ | 按顺序累加分片优化器状态、梯度和参数，并逐项核算字节 |
| 基础 | 梯度也分片后，为什么仍能正确更新？ | 每份状态只需自己负责参数的全局梯度分片 |
| 推导 | 为什么不能先用局部梯度做 Adam 再平均状态？ | 平方与归一化带来非线性，局部平方平均与平均梯度平方不同 |
| 实现 | 参数平时只留分片，前向怎样计算？ | 对当前参数组 AG，完整副本提供给模块计算，按策略释放 |
| 实现 | 当前默认会在前向后释放所有完整参数吗？ | 自动策略区分根模块和子模块，需核验版本与显式参数 |
| 选择 | 常驻状态缩小 N 倍，峰值还取决于什么？ | 临时完整参数、预取组、激活、归约缓冲、padding 和 allocator |
| 选择 | 为什么小参数组也可能慢？ | collective 启动与调度开销增加，需平衡容量和通信效率 |
| 排障 | 1.5 倍通信比值依赖哪些假设？ | 相同元素宽度、两次参数 AG 加一次梯度 RS，且采用指定流量口径 |
| 排障 | 各 rank 有效 token 数不同如何保证同一目标？ | 使用全局有效分母并与实际归约缩放一致，加入不等样本负对照 |
| 排障 | 检查权重一致是否足够？ | 同时比较梯度、m、v、多步更新、padding 和样本分母 |

## 10. 适用边界与延伸

本文以 ZeRO 原始论文说明分片思路，以当日 PyTorch FSDP2 源码追踪当前执行。论文式 FP16 预算、flat 补齐练习、FP64 CPU 程序与当前 DTensor 布局分别记录假设。具体 DeepSpeed 配置、CPU offload、复杂混合并行与 GPU 性能可以继续专题核验。

下一篇候选为训练 checkpoint 应保存哪些状态，怎样在中断后恢复一致的训练进度，可衔接参数所有权、优化器状态与随机性。

资料与源码笔记见 [参考资料](references.md)，实际审查见 [审查记录](review.md)，返回 [Wiki 索引](../../../index.md)。
