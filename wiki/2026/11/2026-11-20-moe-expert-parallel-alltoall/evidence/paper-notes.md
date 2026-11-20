# 稀疏 MoE 与 Switch 原始论文阅读笔记

访问日期：`2026-11-20`（Asia/Shanghai）。下列是自己的理解、推导与实现对应，不是论文原文转录。

## 1. Shazeer 等，2017

- 题名：*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*。
- 作者：Noam Shazeer、Azalia Mirhoseini、Krzysztof Maziarz、Andy Davis、Quoc Le、Geoffrey Hinton、Jeff Dean。
- 固定版本：arXiv `1701.06538v1`，提交于 `2017-01-23T18:10:00Z`。[HTML](https://arxiv.org/html/1701.06538v1)、[版本记录](https://arxiv.org/abs/1701.06538)。
- 实际阅读范围：第 2 节与式（1）、第 2.1 节式（2）—（5）、第 4 节，附录 A 的 load 估计与 importance 区别。

式（1）的重要结构是：不同专家有独立参数，gate 为零的专家可省去计算，非零专家输出进行线性加权。专家可以是网络，本文标量函数只是手算工具，实际代码采用两层 tanh 专家。

原文 Noisy Top-k 在 logits 中加入可学习尺度的高斯噪声，再只保留最高的 k 项并对它们归一化。本文没有运行该噪声过程，也没有复现原文训练；确定性教学门控只保留 Top-k 与归一化结构。选择索引是离散的，选定集合内的 gate 才通过通常的反向路径传播梯度。

第 4 节区分专家收到的 gate 总量与收到的样本数量：少量大权重和大量小权重可能得到相近 importance，却产生不同的设备负载。附录 A 为 load 引入平滑估计，与后来 Switch 风格的单个点积辅助项不同。本文没有把二者称作同一条 loss。

## 2. Fedus、Zoph、Shazeer，Switch

- 题名：*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*。
- 作者：William Fedus、Barret Zoph、Noam Shazeer。
- arXiv `2101.03961`：v1 `2021-01-11`；本文固定 v3，`2022-06-16T20:36:07Z`。[HTML](https://arxiv.org/html/2101.03961v3)、[版本记录](https://arxiv.org/abs/2101.03961)。
- 实际阅读：第 2.1 节式（1）—（2）、第 2.2 节容量及式（4）—（6）、附录 F 的路由与 AllToAll dispatch／返回伪代码。

Switch 先定义对所有专家的概率，再使用 Top-1；选中的 gate 并不因此恒为 1。本文默认在选中集合内重归一化，所以 Top-1 主任务 gate 梯度为零。这个差别由独立测试覆盖，不能只用“Top-1”三个字概括。

容量给专家 batch 大小设定上限，超过时 token 的对应专家计算会被跳过；增大容量可以减少溢出，但 padding 的空槽会增加开销。本文把 Top-1 的平均分配 `T/E` 扩展为 Top-k 的 `Tk/E`，并明确 T 是一次路由调用的范围，不代表自动跨所有 EP rank 的全局配额。

辅助 loss 把离散分配频率 f 与可导平均概率 P 相乘，乘以专家数与系数。原文式（4）—（6）是 Top-1。本篇的 Top-k 计数除以 Tk，与所读 Megatron helper 的归一化一致；没有复现分布式 global-batch／sequence-batch 等所有策略。

## 3. 自编例子与边界推演

- 四 token 的 Top-2 分配共有 8 行；每专家计数 `[2,2,3,1]`，原 owner 到专家 owner 的计数矩阵 `[[3,1],[1,3]]`。
- 同一个 `t1` 的 pre gate `[.6,.2]` 输出 `4.4`，post gate `[.75,.25]` 输出 `5.5`；选择相同专家不足以证明两种配置等价。
- 容量 2、按分数保留时删除 `t2→e2`，且不重新归一化；对应输出由 `10.875` 变成 `7.5`。
- 本例完整概率均值 `[.2625,.1875,.35,.2]`、分配频率 `[.25,.25,.375,.125]` 给出 `L_aux=.01075`（系数 `.01`）。
- 辅助项达到某个均匀参考数值并不是负载均匀的充分证明：本教学 tie 规则在全部 logits 相等时总选专家 `[0,1]`，f 不均匀而 P 均匀，loss 仍为系数值。正文没有宣称该值是所有输入的严格全局下界。

## 4. 与当前官方实现和本机验证的关系

Megatron 当前普通非融合 Softmax helper 明确提供 pre／post 两种归一化，路由后按容量 mask 保留概率；AlltoAll dispatcher 保存 splits 和逆排列，传递 token 与概率，再调用本地专家、返回原 token owner。aux 输入会另行构造。

本机教学 gate 保留在源端，专家计算后才乘 gate；上游所读路径传递概率到专家。前向线性加权语义可对齐，但实际通信消息和 kernel 执行不同。EP＋TP 的实际路径还有 TP 集体通信；本文固定 TP=1。

真实验证为 CPU float64：23 项单进程测试及 2 个 Gloo rank 的混合／空 owner 场景，包含第一阶梯度和一步 SGD 算术对照。没有执行论文训练、GPU fused dispatcher、分布式容量或 aux 训练、长期训练稳定性或性能比较。

## 再分发范围

两篇 arXiv 页面均标示永久非独占分发许可；未取得全文或图像再分发许可。本仓库仅保存上述原创笔记与定位，不打包原论文 HTML／PDF、不复用图像。源码证据另按各自许可证归档。
