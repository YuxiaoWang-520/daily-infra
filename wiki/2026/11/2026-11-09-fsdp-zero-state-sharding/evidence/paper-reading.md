# ZeRO 原始论文阅读笔记

阅读日期为 2026-11-09，时区 Asia/Shanghai。

论文题名为 ZeRO: Memory Optimizations Toward Training Trillion Parameter Models，作者为 Samyam Rajbhandari、Jeff Rasley、Olatunji Ruwase、Yuxiong He。arXiv 首发为 2019-10-04，本次版本 v3 发布于 2020-05-13。DOI 为 10.48550/arXiv.1910.02054。[原始版本入口](https://arxiv.org/abs/1910.02054v3)

## 实际阅读范围

阅读第 3.1 节模型状态，第 4.1 节所有权与使用时机，第 5.1—5.3 节三阶段分片，第 7.1—7.2 节通信分析，并核对图 1 的状态比较与说明。PDF 中相关页为第 7、9—11、13—14 页，章节号用于跨工具定位。[原始 PDF](https://arxiv.org/pdf/1910.02054v3)

## 符号与推导对应

| 论文符号 | 本文对应 | 含义 |
| --- | --- | --- |
| Ψ | P | 参数元素数量 |
| Nd | N | 数据并行分片组人数 |
| K | 默认 12 B／参数 | FP32 主副本与两个 FP32 Adam moments 的合计宽度 |

在论文选定的 FP16／FP32 布局下，每参数还需 2 B 参数和 2 B 梯度。按三类状态是否复制，逐项相加即可得到 16P、4P+12P/N、2P+14P/N、16P/N。正文与练习用 P=10^9、N=8 另行复算。

论文通信分析按统一元素量比较普通数据并行与参数按需收集。本文保留有限 N 因子，分别乘参数与梯度字节宽度，并明确只计每 rank 发送，得到两次参数 AG 加一次梯度 RS 的预算。N=4、P=8 和混合 dtype 的结果另由测试核验。

## 与当前实现的关系

参数按需收集、梯度交给状态所有者，与当前 FSDP2 的参数组生命周期对应。当前实现通过 DTensor、hooks、collective 对象和 stream 依赖完成这些动作，默认根模块还会保留前向后的完整参数。

论文分析为分片机制提供原始依据，当前实现配置由固定 PyTorch commit 核验。CPU 练习使用一份 flat 向量、显式同步 collective、本地 FP64 Adam 和完整参考，描述所选教学布局与保留参数的调度。

## 原文归档范围

arXiv 页面列出非独占分发许可。发布目录保存本文自行撰写的笔记、符号推导、章节定位与版本链接，论文全文留在原始发布入口。本文没有复制论文插图，机制图为自绘 Mermaid。[许可入口](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
