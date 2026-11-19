# 官方文档与代码阅读笔记

日期：`2026-11-19`。本文件记录自己的归纳；直接的固定代码证据另存于 [source-manifest.json](source-manifest.json)。

## Megatron Core Context Parallel Overview

[官方页面](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)把 CP 定义为沿序列切分网络输入和激活。Attention 的本地 Q 依赖全局 K／V，因此需交换 K／V；反向需汇总 K／V 激活梯度。文档的 AllGather／ReduceScatter 描述的是收集及归约分片的语义，并说明其可由环形 P2P 实现，不应由此猜测 trace 中必定出现一次名为 AllGather 的 collective。

文档将这一策略与 Megatron 的 SP 相比：SP 处理某些 token 独立区域的激活，CP 进一步将 Attention 工作沿上下文分配。本文沿用这个限定语境，不推广到所有文献中叫 sequence parallel 的方法。

页面还讨论 GQA／MQA 减少 K／V 通信负载、因果负载均衡，以及与 TP／PP／DP 组合。它的性能图来自指定配置；本文不引用成新硬件或本机测量。

通过 Parallel 提取时，页面快照获取日期为 `2026-11-18`，检索发生于 `2026-11-19`。与其依赖浮动文档的精确实现细节，本文同日 fetch 了实际源码。文档中的历史最低版本声明没有用作当前全部功能的支持矩阵。

## 当前调用链：先核对组，再核对 backend 与 kernel

1. Megatron `TEDotProductAttention` 根据 CP 配置接入 CP 进程组、全局 rank 列表、stream 和通信类型。实际 forward 将 Q、K、V 委托给 TE。
2. TE `DotProductAttention` 选择 backend。归档了 FlashAttention 入口和 fused 入口，不能因为二者都支持 CP 就说所有格式都走相同 kernel。
3. backend 在 CP 条件成立时调用 `attn_forward_func_with_cp`，根据 `cp_comm_type` 分派。
4. P2P autograd Function 安排通信缓冲和每步 Attention，逐步合并 LSE 与输出；`cp_size+1` 的外层循环还处理合并收尾，不表示 `p+1` 个不同 K／V owner。
5. 因果布局有 diagonal／lower／upper 的特定选择；本篇逐位置 mask 仅实现其数学语义。反向同样需要按格式与因果情况修正和累加。

当前配置与稳定 tag 都有 P2P／AllGather／A2A／层次式选择，稳定 TE tag 也保留了本文引用的 LSE 和输出合并结构。没有做两个版本完整实现相等的判断。当前 TE 快照检查滑动窗口与格式等限制，正文只将其表述为这个 SHA 的行为。

## PyTorch 正确性对照为何分成两类？

单进程测试使用独立的 CPU `scaled_dot_product_attention`，Boolean mask 的 True 表示允许。注意非方形 query/key 时，默认的 `is_causal=True` 不会自动知道分片的全局 query 偏移；本地分片对照显式传入全局位置生成的 mask。完整方形 dense 对照则使用 `is_causal=True`。

双进程第一部分用当前安装包仍提供的可微 `torch.distributed.nn.functional.all_gather`。其 `_AllGather.backward` 对 NCCL／XCCL 使用归约分片，对 Gloo 等路径通过 AllToAll 后求和模拟；其中 Gloo 的 AllToAll 又可由 scatter 实现。归档了当前 main 及安装包实际 `_AllGather`，没有把 Gloo 路径说成实测 NCCL ReduceScatter。

该接口已发出弃用警告，公开记录未压制。选择它是为了在既有 CPU 环境建立可核对的训练语义基线，不是推荐新工程继续绑定旧接口。当前主分支和安装包提示的新接口路径也存在命名差异，应按目标版本核验 functional collectives 的实际 API、自动求导及 backend 支持。

## 教学模型没有覆盖什么？

普通 autograd 会保留每块的中间图，真实实验还为独立参考保留完整合成输入，因此不能测得“CP 训练显存缩小 p 倍”。Gloo P2P 只验证块经过环时输出正确，未实现环形反向与计算通信重叠。固定完整源码 SHA 解决可定位性，并不替代缺失的 GPU 执行。
