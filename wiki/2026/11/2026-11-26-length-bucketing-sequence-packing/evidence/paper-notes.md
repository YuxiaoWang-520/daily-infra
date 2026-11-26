# 原始论文阅读笔记

论文为 Krell、Kosec、Perez、Fitzgibbon 的 *Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance*，arXiv:2107.02027v2。初次提交 2021-06-29，v2 修订于 2022-10-05。2026-11-26 实际读取 [原始 HTML v2](https://arxiv.org/html/2107.02027v2) 的下列章节；页面缓存 acquired_at 为 2026-10-07。

以下为自行归纳的机制与对应关系，不转载原图、整页正文或性能表。原文标注 CC BY-SA 4.0；本篇只保存必要笔记与精确定位，不保存全文。

## §3.1.1、§3.1.2：装箱算法的对象不同

SPFHP 处理长度直方图，从长到短遍历，优先选择剩余空间最多的 pack，并限制最大 packing depth。NNLSHP 把长度计数写为非负最小二乘问题：长度直方图为 b，预先枚举的长度组合组成 A，非负重复数为 x，拟合加权的 Ax 与 b，再处理整数重复数。

本篇 FFD 则逐个处理原样本 ID：长度降序后，选择第一个仍能放下该样本的 pack。它与 SPFHP 的选择规则不同，也没有 NNLSHP 的直方图求解。教程不复现上述两种论文算法，没有借用它们的运行时间或接近最优率；保留一个 FFD 用 3 行而存在 2 行可行解的反例。

## §3.2.1：绝对位置与样本身份

论文为 packed BERT 重置每条原序列的绝对位置，并显式选择对应位置嵌入。长度 `[2,3]` 对应位置 `[0,1,0,1,2]`，不是整行递增。

对应本篇 `pack_batch` 的 positions 和 `ToyAttentionLM.position_embedding`。CPU 模型也使用学习型绝对位置，正反对照只支持这一条件；没有把结论推广到 RoPE 等未验证编码。

## §3.2.2：跨样本可见范围

为保持原样本的独立性，论文在 Attention softmax 前使用块对角 mask。隔离的依据是原样本归属，分隔符本身不代替这些可见范围约束。

对应本篇 `segments` 与 bool `allow`。论文的 BERT 是双向注意力背景；本篇因果 LM 在同段条件之外再加 `j≤i`，得到块内下三角。当前 PyTorch bool True 转成加性 0，False 转成负无穷；math 路径先做矩阵乘，再施加 mask。这个代码事实支持语义隔离，并不证明 GPU 稀疏执行或吞吐。

## §3.2.3：目标权重不能跟着物理 pack 改变

论文讨论 per-sequence 任务被错误改成 per-pack 权重时的目标变化。其处理是保留原序列身份，按任务原本的统计单位计算损失与准确率。

本篇选择另一种明示目标：所有有效监督 token 等权，合计 loss sum 与有效目标数后再取平均。它不是宣称论文要求所有任务用 token mean。next-token 跨界和已移位标签属于本篇因果任务的额外处理；不能从 BERT 的 masked-LM／NSP 标签约定直接照搬。

## §3.3 与 §4.1：有效 batch 和硬件边界

物理行数不变时，packing 可能提高一个更新内的真实样本与 token 数，进而改变有效 batch。论文讨论调整累计／batch 等设置，并指出超参数启发式不能保证完全消除变化。

本篇等价测试严格固定底层两条样本和参数，只比较两种表示。它不证明改变每步样本集合后训练曲线仍完全相同。论文 §4.1 的装箱与加速比较具有 BERT、IPU 和具体长度分布背景；本篇不引用这些性能数值作为 GPU 预期。

## 对照结论

论文支撑的是保留样本边界、模型位置和原目标权重的思路。本篇数学补充因果 mask、局部标签移位、token 目标与普通 DDP 平均下的归一化；以固定 PyTorch 源码和 CPU 输出／梯度对照核验。论文算法、教学 FFD、CPU dense math 与未运行的 GPU varlen 四者保持分开。
