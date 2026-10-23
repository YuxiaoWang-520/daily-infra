# Attention 原论文、当前源码与对应笔记

核验日期：2026-10-23，Asia/Shanghai。论文原 PDF 已实际下载、提取文本并阅读下列位置；只保存自写笔记与获取元数据，不重新分发整篇 PDF。

## 1. Online normalizer

Maxim Milakov、Natalia Gimelshein，Online normalizer calculation for softmax。[arXiv:1805.02867v2](https://arxiv.org/abs/1805.02867v2)，首发 2018-05-08，所读 v2 为 2018-07-28，PDF 标题与版本标识一致。

阅读第 2 节、Algorithm 1／2 的朴素与稳定 softmax，第 3 节 Algorithm 3、Theorem 1 及归纳证明。关键是运行最大值 m 和归一化量 l，更新 l 时先乘 `exp(m_old-m_new)`，再加入新元素的指数贡献。

自己的推导：当一个分数块到来时，以块最大值与旧 m 的较大者统一刻度；原单元素递推可按结合关系推广为块。论文的 online normalizer 一遍计算最大值与分母，之后还有一遍写概率。本文不把它缩写成“一遍访问就返回完整向量”。

全屏蔽与空集合零输出是本篇显式扩展约定，不是原算法有限有效向量的数学结论。原论文历史 GPU 性能没有搬用为本机或当前实现结果。

## 2. FlashAttention

Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré，FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness。[arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135v2)，首发 2022-05-27，v2 提交 2022-06-23；PDF 标题页另排 June 24, 2022，按 arXiv 版本号和摘要页定位，不混淆这两个日期。

阅读第 2.2 节／Algorithm 0 的标准 S、P、O，第 3.1 节／Algorithm 1 的块合并和重算动机，第 3.2 节的 IO 分析及其前提，以及 Appendix B.3／Algorithm 2 的缩放、mask 和 dropout 定义。未声称复现反向实现或全部性能实验。

对应笔记：

- 朴素实现物化 `N×N` 的分数与概率，中间容量二次增长。
- 合并块时最大值和分母要换刻度；加权分子也需要同样换刻度。
- 论文 Algorithm 1 的 K／V 外层及阶段归一化 O，与本文 Q 外层、未归一化 u 的循环不同；统一到同一不变量后可解释代数对应，不据此宣称相同的 IO 调度或最优性。
- 保存统计量后重算局部 S／P 可服务反向，本文没有实现反向或 dropout。标准稠密运算仍有成对计算成本，不把“内存节省”说成“计算线性”。

两篇 PDF 的实际获取命令、UTC 时间、大小、SHA256、文本提取退出码见 [paper-acquisition.json](evidence/paper-acquisition.json)。

## 3. 当前代码：从行块入口到重缩放，再看 CuTeDSL

官方仓库 `Dao-AILab/flash-attention`，默认 `main` 本次刷新后固定 `767386d814051d1d853c45ac952d8259b5bc7235`。获取方式为固定 GitHub API 文件下载，没有完整 clone、构建或 GPU 执行。

| 主张 | 固定源码 | 阅读结论与限制 |
| --- | --- | --- |
| Query 行块入口 | [flash_fwd_kernel.h 1082—1098 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/csrc/flash_attn/src/flash_fwd_kernel.h#L1082-L1098) | 一条 C++／CUDA 路径的块、batch、head 映射，不是所有架构统一 dispatch。 |
| QK、mask、更新、PV | [326—374 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/csrc/flash_attn/src/flash_fwd_kernel.h#L326-L374) | 局部 P 会转 FP16／BF16，不是 CPU FP64 参考的 bitwise 模拟。 |
| 最后归一化与输出转换 | [438—443 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/csrc/flash_attn/src/flash_fwd_kernel.h#L438-L443) | 部分和累加与最终输出类型分开。 |
| 同时重缩放旧 sum 与输出状态 | [softmax.h 136—165 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/csrc/flash_attn/src/softmax.h#L136-L165) | `scores_scale` 同乘 row_sum 和 acc_o；scale 与 exp2 刻度需统一理解。 |
| 无效最大值与 exp2 | [65—89 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/csrc/flash_attn/src/softmax.h#L65-L89) | 防止 -inf 减 -inf；其他保护受 Check_inf 等模板条件，不泛化为任意用户 mask 支持。 |
| 当前有 FA4 CuTeDSL 入口 | [README 80—99 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/README.md#L80-L99) | 只用作实现家族与入口证据，不引用 README 的速度图片或安装任何依赖。 |
| CuTeDSL 的在线状态与缩放 | [cute/softmax.py 187—250 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/flash_attn/cute/softmax.py#L187-L250) | row_max／row_sum／row_scale，包含 check_inf 分支。 |
| 最终统计与缩放函数 | [252—301 行](https://github.com/Dao-AILab/flash-attention/blob/767386d814051d1d853c45ac952d8259b5bc7235/flash_attn/cute/softmax.py#L252-L301) | finalize 与 rescale_O 的方法语义；没有追完所有架构调用点、变体或流水。 |

实际快照：[C++ 前向](evidence/upstream/Dao-AILab/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h)、[C++ softmax](evidence/upstream/Dao-AILab/flash-attention/csrc/flash_attn/src/softmax.h)、[CuTeDSL softmax](evidence/upstream/Dao-AILab/flash-attention/flash_attn/cute/softmax.py)、[README](evidence/upstream/Dao-AILab/flash-attention/README.md)、[许可证](evidence/upstream/Dao-AILab/flash-attention/LICENSE)。[source-manifest.json](evidence/source-manifest.json) 记录完整命令、时间与摘要。

## 4. CPU 验收边界

独立对照使用本机 PyTorch 2.9.1 CPU 的 softmax 和 SDPA；说明接口时另核对 [PyTorch 2.14 SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。不能把 CPU 对照叫 FlashAttention 后端运行。

内部统一 FP64、推理前向、单 head，无 dropout／训练／缓存偏移。causal 为 top-left `j<=i`，不是对所有非方形与增量解码 API 的通用约定。任意稠密 mask 是参考实现输入，不等于上述 C++ 模板路径支持表。

tile 可能覆盖完整问题；其分数元素上界为 `min(Bq,Nq)*min(Bk,Nk)`。没有另外持久保存全量 S／P，但可选 mask 自身可能二次大，输入提升为 FP64 也需线性容量；常数份临时量不等于实测峰值。

[15 项测试](evidence/runs/tests.txt)、[逐块诊断与容量整数模型](evidence/runs/lab.json)、[环境与命令](evidence/runs/summary.json)、[校验和](evidence/SHA256SUMS)、[归档检查](evidence/runs/archive-check.txt)。
