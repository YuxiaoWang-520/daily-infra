# Roofline 资料与源码笔记

核验日：2026-10-22。结论依据原论文、当日实际获取的官方源码文件和本地算术验收；没有构建完整 Triton 仓库，也没有 GPU 测量。

## 核心论断对应资料

| 论断 | 一手定位 | 本篇处理 |
| --- | --- | --- |
| 强度的内存层级、两条屋顶与拐点 | [Roofline 技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)，第 3 节、Figure 1 | 区分真实 DRAM 流量与教学最低输入输出账本。 |
| 可持续带宽与额外优化上限 | 同报告第 3—4 节 | 不用规格峰值保证实际实现，不引用历史 CPU 数字冒充现代 GPU。 |
| 分步 softmax 的中间量读写 | [固定源码 42—59 行](https://github.com/triton-lang/triton/blob/c1641abbb17a80b8c9cad1fa05b4c70d6ac260ab/python/tutorials/02-fused-softmax.py#L42-L59) | 按算子张量账本数 `8MN+4M` 个元素，不宣称是 profiler 流量。 |
| 融合后的加载、计算、存出 | [固定源码 84—109 行](https://github.com/triton-lang/triton/blob/c1641abbb17a80b8c9cad1fa05b4c70d6ac260ab/python/tutorials/02-fused-softmax.py#L84-L109) | `tl.load → max/exp/sum/div → tl.store`；片上容量和资源适用条件保留。 |

官方教程的可读入口为 [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，实现事实以固定文件为准。相邻注释第 67 行写 `MN bytes`，但比值使用 `2MN`；本文用 `2MN × itemsize` 修正单位，不能直接照抄。

## 论文阅读记录

题名：Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures。作者：Samuel Williams、Andrew Waterman、David Patterson。版本：UCB/EECS-2008-134，2008-10-17。阅读了 PDF 标题页、版权页、第 3 节模型、第 4 节 ceilings 与 Figure 1 的解释。

第 3 节将流量明确放在缓存与主存之间，强调缓存过滤后的字节数；带宽使用微基准可持续值。推导为 `min(P,B*I)`，并从交点得到 `I=P/B`。第 4 节说明主屋顶下还可能有指令级并行、指令组合、内存亲和性等上限。因此，单凭算法最低读写量只能先做假设，不能完成对真实 kernel 的诊断。

本篇为初学者增加 `T>=max(F/P,Q/B)` 的时间解释，用虚构速率推算例子；这不是原文硬件结果复现。方阵 GEMM 的输入输出账本只说明理想复用，后续分块篇会补充重读模型。

PDF 已下载、恢复完整后提取文本并阅读。初次超时、续传成功、退出码与 SHA256 均保留在 [获取记录](evidence/paper-acquisition.json)。原报告对服务器再发布有限制，本仓库仅保存自己的阅读笔记与获取元数据，不重新分发整份论文。

## 源码归档

官方仓库：`https://github.com/triton-lang/triton`；默认分支 `main`，本批获取时固定 `c1641abbb17a80b8c9cad1fa05b4c70d6ac260ab`。使用 GitHub API 实际下载固定版本文件，路径、命令、时间、字节数和摘要见 [source-manifest.json](evidence/source-manifest.json)。

- [完整教程快照](evidence/upstream/triton-lang/triton/python/tutorials/02-fused-softmax.py)。
- [上游许可证](evidence/upstream/triton-lang/triton/LICENSE)。
- [核心文件校验和](evidence/SHA256SUMS)。

本地练习为独立标准库成本模型，不调用 Triton，不承诺实际设备性能。[运行命令与环境](evidence/runs/summary.json)、[诊断输出](evidence/runs/lab.json) 和 [链接、源码行号及哈希核验](evidence/runs/archive-check.txt) 共同给出验证范围。
