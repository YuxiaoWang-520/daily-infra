# 原始论文阅读笔记

实际访问日期为 `2026-11-17`，时区 `Asia/Shanghai`。阅读使用 arXiv 原文 HTML，归档以下自己的推导和对应关系。两篇原文标注 arXiv perpetual non-exclusive license，本文保存阅读笔记及固定定位，原文通过来源页获取。

## 1. Megatron-LM 的线性层配对

题名为 *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*。作者为 Mohammad Shoeybi、Mostofa Patwary、Raul Puri、Patrick LeGresley、Jared Casper、Bryan Catanzaro。首次提交为 `2019-09-17`，阅读版本为 `v4`，修订于 `2020-03-13`，arXiv ID 为 `1909.08053`。[原文与版本](https://arxiv.org/abs/1909.08053)、[实际阅读的 §3](https://arxiv.org/html/1909.08053v4#S3)

阅读 §3 的公式（1）—（3）、图 3、通信函数示例及附录 B.1—B.2。核心连接是第一层输出特征切分、本地逐元素非线性、第二层输入特征切分，以及输入／输出边界的归约。图中的输入边界前向保持值，反向合并贡献，输出边界采用相应的前向归约和反向传递。

本文自行构造 `H=2、F=4` 的 MLP，用 ReLU 简化手算，并加入 bias、MSE 和梯度验收。`XW₁` 按列分块、`ZW₂` 按行分块的恒等式及实际数值在正文展开。附录的混合并行与 RNG 提醒读者，TP 组、DP 组和随机性需要分别核对，本文固定例子没有 dropout。

当前实现对应 `ColumnParallelLinear`、`RowParallelLinear` 和 `mappings.py` 的通信 autograd。现代普通线性层可以将 dgrad 通信放入自定义 backward，具体分支还受 SP、禁用归约、expert、融合和重算等配置影响。原论文抽象与本篇两 rank CPU 结果的关系见正文第 3、5、6 节。

## 2. Sequence Parallel 与输入重新收集

题名为 *Reducing Activation Recomputation in Large Transformer Models*。作者为 Vijay Korthikanti、Jared Casper、Sangkug Lym、Lawrence McAfee、Michael Andersch、Mohammad Shoeybi、Bryan Catanzaro。首次提交为 `2022-05-10`，阅读版本为 `v1`，arXiv ID 为 `2205.05198`。[原文与版本](https://arxiv.org/abs/2205.05198)、[实际阅读的 §4.2.2](https://arxiv.org/html/2205.05198v1#S4.SS2.SSS2)

阅读 §4.2.1—4.2.2、图 4—6 和公式（3）—（4）。论文在 token 独立区域按 sequence 维切分，用 AllGather／ReduceScatter 在该布局与特征切分区域之间转换，并分析基本通信的 ring 分解关系。

同节随后讨论仅保存输入分片的策略。权重梯度需要完整输入，因此反向还要额外 AllGather，并尝试与计算重叠。本文把这个条件作为预算的可选项，分别计算四个基本转换和加上重新收集后的五次通信。

当前 Megatron 源码在 forward 保存原始 `input`，SP 分支收集完整输入供 GEMM 使用。backward 的 `sequence_parallel` 且 `wgrad_compute` 分支再次收集输入，然后继续计算权重梯度。这与论文的保存策略相互对应。[前向保存及收集片段](upstream/Megatron-LM/linear-forward-saved-input.txt)、[反向重新收集片段](upstream/Megatron-LM/linear-backward-regather-and-dgrad.txt)

本文只取与线性层边界和保存策略有关的机制。论文的性能与整层容量公式依赖其模型和硬件条件，本文采用自己的容量／ring 估算，实际运行范围为 CPU 代数及普通 TP 的两 rank 验证。

返回 [正文](../README.md) 或 [参考资料](../references.md)。
