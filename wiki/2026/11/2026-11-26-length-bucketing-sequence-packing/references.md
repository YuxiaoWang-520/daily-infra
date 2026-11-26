# 参考资料与证据

主题：长度分桶、sequence packing、独立样本 Attention 与损失目标。实际源码核验时间为 2026-11-26T11:06:56+08:00；网页检索与原始论文阅读在当天进行，服务返回的页面缓存日期另列。

## 关键论断与来源

| 编号 | 论断与实际定位 | 一手来源 | 本地归档及限制 |
| --- | --- | --- | --- |
| P1 | BERT packing 的位置、Attention、per-sequence 任务和有效 batch 需调整 | Krell 等，[原论文 v2](https://arxiv.org/html/2107.02027v2)，§3.2.1—3.2.3、§3.3 | [paper-notes.md](evidence/paper-notes.md)；正文把块内改为因果规则，没有移植 BERT／IPU 加速数据 |
| P2 | SPFHP／NNLSHP 是直方图方法，与简单 FFD 不同 | 同上，§3.1.1、§3.1.2；§4.1 对 FFD 的比较 | [论文笔记](evidence/paper-notes.md)；本篇只实现教学 FFD，不声称复现两种论文算法 |
| S1 | SDPA bool True 为参与、dropout 按参数执行 | [functional.py，L6493—6537](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/torch/nn/functional.py#L6493-L6537) 与 [官方文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html) | [片段](evidence/upstream/pytorch/main-functional.py-L6493-L6537.txt)、[网页笔记](evidence/reading-notes.md)；文档 acquired_at 为 2026-11-23 |
| S2 | bool mask 转为 0／负无穷加性偏置 | [attention.cpp，L573—586](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/aten/src/ATen/native/transformers/attention.cpp#L573-L586) | [原样片段](evidence/upstream/pytorch/main-attention.cpp-L573-L586.txt)；说明语义，不等于稀疏执行 |
| S3 | 指定 math 路径后，矩阵乘先于 mask 和 softmax | [math dispatch，L893—919](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/aten/src/ATen/native/transformers/attention.cpp#L893-L919)、[math 计算，L979—1015](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/aten/src/ATen/native/transformers/attention.cpp#L979-L1015) | [dispatch](evidence/upstream/pytorch/main-attention.cpp-L893-L919.txt)、[计算](evidence/upstream/pytorch/main-attention.cpp-L979-L1015.txt)；CPU 运行包身份另列，不声称构建过 main |
| S4 | CE 按现有 target 对齐，不负责 LM 的标签移位 | [Python CE 接口](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/torch/nn/functional.py#L3525-L3558)、[native CE](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/aten/src/ATen/native/LossNLL.cpp#L632-L666) | [接口](evidence/upstream/pytorch/main-functional.py-L3525-L3558.txt)、[调用](evidence/upstream/pytorch/main-functional.py-L3604-L3616.txt)、[native](evidence/upstream/pytorch/main-LossNLL.cpp-L632-L666.txt)；其他模型封装可能自行 shift，需遵守其约定 |
| S5 | 无类别权重时，mean 按非忽略目标数归一化 | [忽略目标分支](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/aten/src/ATen/native/LossNLL.cpp#L246-L271)、[分母](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/aten/src/ATen/native/LossNLL.cpp#L288-L304) | [忽略处理](evidence/upstream/pytorch/main-LossNLL.cpp-L246-L271.txt)、[归一化](evidence/upstream/pytorch/main-LossNLL.cpp-L288-L304.txt)；本篇限整数类别、无类别权重、无 label smoothing |
| S6 | varlen 使用累计边界、展平布局与最大长度，因果由 window_size 表达 | [varlen 接口](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/torch/nn/attention/varlen.py#L361-L406)、[CUDA int32 示例](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/torch/nn/attention/varlen.py#L476-L484) | [接口片段](evidence/upstream/pytorch/main-varlen.py-L361-L406.txt)、[示例片段](evidence/upstream/pytorch/main-varlen.py-L476-L484.txt)；GPU 路径未运行 |
| S7 | varlen 根据条件选择后端，再向私有 op 传入累计长度 | [选择](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/torch/nn/attention/varlen.py#L172-L230)、[调用](https://github.com/pytorch/pytorch/blob/205e1e1fca6c6634bff5b06691d6648a4f72fe9a/torch/nn/attention/varlen.py#L487-L546) | [选择片段](evidence/upstream/pytorch/main-varlen.py-L172-L230.txt)、[调用片段](evidence/upstream/pytorch/main-varlen.py-L487-L546.txt)；不宣称所有输入可用，也不对外部后端给出性能结论 |

## 版本与源码获取

- 官方仓库 `https://github.com/pytorch/pytorch`，当天 GitHub API 核验默认分支 `main`。
- 真实 fetch 后 main 固定为 `205e1e1fca6c6634bff5b06691d6648a4f72fe9a`。
- 最新 release API 返回 `v2.15.0`、发布时间 `2026-10-28T17:09:49Z`，稳定 tag commit 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。
- [稳定版 varlen 参数与说明](evidence/upstream/pytorch/v2.15.0-varlen.py-L348-L370.txt) 与 main 分别归档；公共参数列表匹配，整个实现文件不相同。本文不将 main 中的后端逻辑视为所有 2.15.0 二进制的保证。
- 实际运行包为 PyTorch 2.15.0，构建 SHA `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。本机只有 CPU 验证，没有编译或运行缓存中的 main。
- [source-update.json](evidence/source-update.json) 保存命令、结果摘要、默认分支及缓存 checkout 保持情况；[source-manifest.json](evidence/source-manifest.json) 保存 14 个原样片段的路径、行号、完整文件／片段摘要及固定链接。
- 上游源码归属与许可见 [LICENSE](evidence/upstream/pytorch/LICENSE)。只归档相关片段，没有把整个源代码仓库加入文章。

## 原始论文与读取范围

Mario Michael Krell、Matej Kosec、Sergio P. Perez、Andrew Fitzgibbon，*Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance*。原始提交日期 2021-06-29，本篇读取 2022-10-05 修订的 **arXiv:2107.02027v2**。原始链接：[摘要与版本](https://arxiv.org/abs/2107.02027)、[HTML v2](https://arxiv.org/html/2107.02027v2)，DOI 为 `10.48550/arXiv.2107.02027`。

当天实际读取 §3.1.1、§3.1.2、§3.2.1—3.2.3、§3.3 及相关 §4.1 比较段落。[本地笔记](evidence/paper-notes.md) 记录假设、公式与代码对应。服务返回 acquired_at 为 2026-10-07；不写成当天论文更新。

HTML 标注 CC BY-SA 4.0。为精简归档，本篇没有保存整篇论文或整页提取文本，保留自行撰写的相关章节笔记、原始版本定位与检索元数据；没有转载论文图片或性能表。算法、位置、因果目标与硬件边界按笔记逐项区分。

## 实际验证与交付

[verification.json](evidence/runs/verification.json) 固定 5 个 Python 源码摘要、环境、5 条命令与真实退出码；[results.json](evidence/runs/results.json) 与 [stdout](evidence/runs/answers-run.txt) 保存成本算术和实际 CPU Attention 结果。[技术核验](evidence/technical-audit.json)、[独立代码复核](evidence/python-review.json)、[两轮审查](review.md) 与 [交付检查](evidence/delivery-check.json) 的范围分别记录。

没有真实数据集、GPU、varlen 执行、分布式通信、吞吐或收敛测量。DDP 分母用数学推导及单进程 autograd 对照，不冒充实际多 rank 训练。
