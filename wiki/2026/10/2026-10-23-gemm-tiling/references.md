# GEMM 分块资料与论文笔记

2026-10-23，Asia/Shanghai。读取原论文相关段落、当前官方教程固定源码和 CPU 教学参考，不复用教程宣传数字作为本机结果。

## 论文

Kazushige Goto、Robert A. van de Geijn，Anatomy of High-Performance Matrix Multiplication。正式发表：ACM TOMS，34（3），Article 12，2008 年 5 月，[DOI 10.1145/1356052.1356053](https://doi.org/10.1145/1356052.1356053)。

实际下载阅读的是作者学校站点 [GotoTOMS_final.pdf](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf)。该文件排版仍有 `20YY` 占位，不能把它伪称为正式出版 PDF；以 URL、获取时间、SHA256 定位版本，正式元数据由 DOI 与作者站点出版记录核对。[获取记录](evidence/paper-acquisition.json) 保存下载和文本提取结果。网页工具后续打开该 PDF 超时，不影响此前成功下载、提取并实际阅读的本地副本。

阅读范围：第 1—2 节的目标与分块符号，第 3 节和 Figure 5 的块级三重循环，第 4.1 节基础层级模型，第 4.2.1—4.2.3 节的缓存层、TLB 与 packing。没有把整篇历史硬件性能或其特定 TLB 描述直接推广到现代 GPU。

自己的对应笔记：内核之上需要按层级组织矩阵片段，复用收益与容量／带宽同时有关；非连续原始子矩阵可通过 packing 改变消费布局，但搬运本身也要摊销。本文 CPU 代码省略 packing 和微内核，只验证 `C_tile += A_panel @ B_panel` 的代数。本篇 GPU 源码采用独立 output tile 与 K 循环，不声称与论文整套 CPU 循环层次逐一等同。

全文再发布权限未获确认，仓库只保留自写笔记和获取元数据。

## 当前源码对应

官方 `triton-lang/triton` 默认 `main` 刷新后固定 `1ec1eee840b8a35b15101e5ab9a4388e905291f3`，逐文件实际下载，无全仓构建。

| 前文问题 | 固定实现 | 差异／边界 |
| --- | --- | --- |
| 谁算哪一块 | [252—264 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/python/tutorials/03-matrix-multiplication.py#L252-L264) | grouped program id 改善局部性机会，不保证硬件开始顺序。 |
| 坐标如何变地址 | [279—290 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/python/tutorials/03-matrix-multiplication.py#L279-L290) | 显式 stride，M／N 取模可能产生冗余；有效元素账本未数这些槽位。 |
| 部分和如何保留 | [292—312 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/python/tutorials/03-matrix-multiplication.py#L292-L312) | FP32 accumulator，K mask=0，最后转 FP16；CPU 参考是 FP32／64。 |
| 输出与入口约束 | [314—351 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/python/tutorials/03-matrix-multiplication.py#L314-L351) | 输出 mask、A 连续、FP16 输出；不能无条件支持所有参考输入。 |

[完整教程](evidence/upstream/triton-lang/triton/python/tutorials/03-matrix-multiplication.py)、[许可证](evidence/upstream/triton-lang/triton/LICENSE)、[获取命令与哈希](evidence/source-manifest.json)。可读入口：[Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)，行为以固定文件为准。

`tile_traffic` 的推导是本篇明确假设下的有效加载账本，不是论文测量数据，也不是 Triton 边缘重读或 CPU 库内部流量。实际 allocator 与编译器资源应另外检查。

验证：[summary.json](evidence/runs/summary.json)、[测试](evidence/runs/tests.txt)、[输出](evidence/runs/lab.json)、[校验和](evidence/SHA256SUMS)、[链接与源码检查](evidence/runs/archive-check.txt)。
