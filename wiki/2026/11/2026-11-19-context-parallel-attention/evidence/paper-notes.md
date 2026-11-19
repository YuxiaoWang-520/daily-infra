# Ring Attention 论文阅读笔记

访问日期：`2026-11-19`（Asia/Shanghai）。本文为自己的阅读、重算与实现对应笔记，不是论文全文转录。

## 版本与定位

- 题名：*Ring Attention with Blockwise Transformers for Near-Infinite Context*。
- 作者：Hao Liu、Matei Zaharia、Pieter Abbeel。
- arXiv：`2310.01889`。首次提交 `2023-10-03`；本篇固定 v4，提交于 `2023-11-27T06:38:47Z`。
- [固定版本摘要](https://arxiv.org/abs/2310.01889v4)、[固定版本 HTML](https://arxiv.org/html/2310.01889v4)。先搜索定位了 v3，核对版本历史后将正文与主要笔记统一到 v4。
- 实际阅读重点：第 3 节（环与 blockwise compute、通信隐藏条件）、Algorithm 1、附录 A 的 Figure 4 前向与反向代码。
- 页面许可为 arXiv 永久非独占分发许可；没有向此仓库取得完整再分发许可，因此不保存论文原文和插图。下列内容为独立解释与推导。

## 从论文中提炼出的机制

1. 把完整序列切成由不同 host 管理的 query 块。在 Attention 的块循环中，Q 的 owner 固定，K／V 沿环经过不同 host。
2. 块的访问次序不影响精确实数语义，前提是摘要正确合并。自己的表述是保存最大值 `m`、归一化分母 `ℓ`、加权分子 `u`；改变最大值时，已有 `ℓ` 和 `u` 必须一起缩放。
3. 每个 host 还负责本地 token 对应的逐 token 计算。对这些操作的分布，与 Attention 的跨 token 依赖应分开理解。
4. 隐藏通信依赖计算块足够大。对论文的 BF16 MHA 简化模型，局部两次乘法约 `4c²D` FLOP，K＋V 通信负载约 `4cD` 字节；用 `F FLOP/s` 与 `B byte/s` 表达，有 `c≳F/B`。真实启动延迟、共享链路和 kernel 效率会收紧条件。

本篇没有摘录或复用论文硬件表的性能数字，也没有将论文在 A100／TPU 的结果当作本机实验。其“近乎无额外开销”的讨论须连同块大小、重叠与实验配置一起阅读。

## Algorithm 1 与附录 A 怎样互相核对？

Algorithm 1 是概念级调度。不能只按简化循环表面的计数实现一个漏掉 owner 的程序。附录 A 的 `_ring_attention_fwd` 使用 `lax.scan(..., arange(0, axis_size))`，将所有 host 的 K／V 块纳入本地 query 的累计状态；`_blockwise_attention_fwd` 返回分子、分母和最大值。`lax.ppermute` 完成沿环的数据传递。

附录中的实现会在扫描中执行旋转。本篇前向演示计算 `p` 个 owner、只执行 `p-1` 次必要旋转，最后不再发送一个不会使用的块；这是一种教学调度选择，未声称与论文代码逐行相同。

`_ring_attention_bwd` 保留前向输出与归一化信息，计算分块 dQ、dK、dV，并让 K、V、dK、dV 一起经过 host。它说明 dK／dV 必须积累不同 query 分片的贡献；把每个 host 的局部 Attention 独立求导后停止，会缺失远端 query 的贡献。

## 与本篇的当前源码对照

| 论文机制 | 本次 TE／Megatron 证据 | 差异 |
| --- | --- | --- |
| Q 固定、K／V 旋转 | TE `AttnFuncWithCPAndKVP2P` 的通信缓冲与收发 | TE 有 CUDA stream、buffer 切换、backend 选择及流水收尾。 |
| 累计 Softmax 归一化状态 | TE `flash_attn_fwd_softmax_lse_correction` 与输出校正 | TE 接收 kernel 的已归一化局部输出和 LSE；练习显式维护 `(m,ℓ,u)`。 |
| 因果依赖 | Megatron zigzag 索引，TE 因果 tile 选择 | 本篇全局位置显式 mask 便于验算，没有复刻 kernel 的半块选择优化。 |
| 跨 query 分片累加 K／V 梯度 | TE 反向 P2P 缓冲和 `dk.add_`／`dv.add_` 分支 | FP8、causal、格式等分支不同；本机没有执行这些 GPU 路线。 |

所有源码版本和片段定位见 [source-manifest.json](source-manifest.json)。TE 是可信官方工程实现对照，不是论文作者 JAX 仓库的同一份代码。

## 自编重算与实测边界

四 token 数值例子验证“局部输出不能平均”；中心有限差分和独立 CPU SDPA 验证公式及梯度。单进程 CP 模型验证布局与归一化组合。双进程 Gloo 实验额外验证可微 AllGather 的 K／V 梯度归约语义，以及真实 P2P 环形前向。

本次没有运行论文 JAX 程序、分布式环形反向、GPU memory-efficient Attention、计算通信重叠或长上下文性能。全局 dense Attention 的总计算依然对序列长度呈平方增长，增加设备和容量不等于解决全部扩展瓶颈。
