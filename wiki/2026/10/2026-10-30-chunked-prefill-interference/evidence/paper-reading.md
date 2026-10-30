# 原论文阅读笔记

访问日期：2026-10-30（Asia/Shanghai）。以下为自己的机制解释与定位笔记；不复制论文全文、原图或性能表。本文图解由教学例子重新绘制，不是论文原图。

## P1：SARATHI（2023）

- 题名：*SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*。
- 作者：Amey Agrawal、Ashish Panwar、Jayashree Mohan、Nipun Kwatra、Bhargav S. Gulavani、Ramachandran Ramjee。
- 版本：arXiv `2308.16369v1`，首次提交 2023-08-31。
- DOI：[10.48550/arXiv.2308.16369](https://doi.org/10.48550/arXiv.2308.16369)。
- 实际读取：[官方完整 HTML v1](https://arxiv.org/html/2308.16369v1) 的 §4.2、Figure 6、§4.3、§5.4；[版本元数据](https://arxiv.org/abs/2308.16369v1)。

### 对本篇最有用的内容

§4.2 和 Figure 6 给出了跨块仍保留因果依赖的机制：后块的新 Query 不能忘记历史 Key／Value，也不能访问未来。分块改变执行组织，因果可见范围仍由原序列位置决定。

为自己复算这一点，本篇使用 6 个位置、Q/K 全 0、V 逐位置递增的例子；第二块开头是全局位置 3，允许 Value 0 到 3，结果为 1.5。此例是本篇合成的，不是论文测量。

§4.3 将 Decode 与 Prefill chunk 混合，目标在于实际批内计算的复用与利用率。单纯把一个大工作分多次，不会自动证明吞吐增加；我们用线性耗时模型只说明调度机会与首字的取舍，没有模拟该资源复用收益。

§5.4 讨论分块成本。这里保留“切分存在开销”的适用提醒，不照搬原实验的加速比、模型、硬件或最佳 chunk 值。

## P2：Sarathi-Serve（OSDI 2024）

- 题名：*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*。
- 作者：Amey Agrawal、Nitin Kedia、Ashish Panwar、Jayashree Mohan、Nipun Kwatra、Bhargav Gulavani、Alexey Tumanov、Ramachandran Ramjee。
- 使用版本：USENIX OSDI 2024 出版方 PDF，会议为 2024-07-10 至 07-12，正文页 117—134；带封面的文件共 19 页。
- [出版方页面](https://www.usenix.org/conference/osdi24/presentation/agrawal)；[出版方全文 PDF](https://www.usenix.org/system/files/osdi24-agrawal.pdf)。
- 相关预印本 ID：arXiv `2403.02310`，v1 为 2024-03-04，观测到的 v3 为 2024-06-17。本篇阅读依据是出版方版本，不把预印本与出版方文件视为逐字相同。
- 实际阅读：§4.1、§4.2（Algorithm 3）、§4.3、§5.4.1。

### 对照 Algorithm 3，理解“先给谁份额”

§4.2 的预算调度先考虑已有 Decode，再推进部分 Prefill，最后尝试新 Prefill，并检查内存与剩余预算。它给出的是减少整段 Prefill 抢占调度机会的机制，不是“旧请求以后再也不增加一毫秒延迟”的保证。

“stall-free”在这里应结合论文假设理解：不让一次完整 Prefill 独占成为长停顿来源；混合批次自身仍有执行时长。它不是 GPU kernel 中途抢占机制，也不是严格的所有请求延迟上限。

§4.3、§5.4.1 说明预算选择与性能／延迟的取舍。形状和重复访问的成本使很小的 chunk 未必好；选择依赖目标负载与延迟约束。文中历史 vLLM 分类只适用于当时对比环境，本篇单独查看今天的默认分支，不沿用历史实现判断。

## 论文、代码与本篇模型的边界

| 层次 | 本篇实际核对的内容 | 不能混同的对象 |
| --- | --- | --- |
| 2023 原始工作 | 分块因果关系、Prefill／Decode 混合思路 | 2026 年默认分支的全部实现 |
| 2024 出版论文 | 预算调度 Algorithm 3、机制与取舍 | 今天每种引擎模式的优先级与默认参数 |
| 作者公开代码 | `96f9911…` 的三遍选择、余量裁剪与约束 | 原论文实验所用二进制、依赖和硬件 |
| 当前 vLLM | `e49ea84…` 的统一进度模型和特定 runner 接口 | 原样复现 Algo 3 或稳定发行实现 |
| 本篇 Attention | 固定 QKV、一层单头、CPU 运算 | 完整模型、GPU 内核或生产精度验收 |
| 本篇调度 | 明确给定的线性成本、固定请求集合 | 权重 piggyback 收益、真实 TTFT／ITL 或吞吐 |

## 原文获取与保存情况

本地下载出版方 PDF 初次与续传均超时，arXiv PDF 连接也超时；部分文件没有作为证据。随后通过网页工具成功读取出版方完整 PDF 和 arXiv 官方完整 HTML 的上述章节。命令、失败退出码、来源和成功阅读方式均在 [paper-acquisition.json](paper-acquisition.json)。没有声称本地完整 PDF 下载成功，也没有用不完整文件提取结论。

出于只保存必要材料和遵循来源再分发边界的原则，仓库保存自己的阅读笔记、准确版本与章节、获取记录及原创图源，不再分发论文全文。未复制论文的原图；本篇的因果表和时间线均为原创合成例子。
