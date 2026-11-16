# 参考资料与证据

主题为 CUDA caching allocator 的复用、碎片与 OOM。检索与读取日期为 2026-11-16，时区 Asia/Shanghai（UTC+08:00）。本篇执行范围为 CPU 教学模型，CUDA 行为由固定源码和官方语义文档支撑。

## 关键论断与来源

| 编号 | 核验的论断 | 一手来源与定位 | 本地证据与范围 |
| --- | --- | --- | --- |
| S1 | 普通 native 分配先找可复用块，失败后按条件取得新段、回收重试 | `DeviceCachingAllocator::malloc`、`get_free_block` | [main 分配路径](evidence/upstream/main/native-core.lines.txt)，含 stream 与池条件 |
| S2 | 同段邻块按状态合并，普通整段回收受活跃块限制 | `free_block`、`try_merge_blocks`、`release_blocks` | [main 释放路径](evidence/upstream/main/native-core.lines.txt)，限定普通 segment |
| S3 | allocated、active、inactive split 与 snapshot 三态的含义 | `torch/cuda/memory.py` 与 native 计数更新 | [API 定义](evidence/upstream/main/memory-api.lines.txt)，expandable 被排除出 inactive split |
| S4 | 默认小请求缓冲与大段大小、舍入及分裂条件 | `AllocatorConfig.h`、`round_size`、`get_allocation_size`、`should_split` | [常量](evidence/upstream/main/allocator-config.lines.txt)与 [native 实现](evidence/upstream/main/native-core.lines.txt)，固定默认条件 |
| S5 | 释放请求与事件完成可以发生在不同阶段 | `free_locked`、`recordStream`、`insert_events`、`process_events` | [事件与计数代码](evidence/upstream/main/native-core.lines.txt)，本机未执行 CUDA event |
| S6 | 默认池回收可能先等待事件，Python API 有明确绑定路径 | `empty_cache` → `_cuda_emptyCache` → `emptyCache` → `release_cached_blocks` | [Python API](evidence/upstream/main/memory-api.lines.txt)、[绑定](evidence/upstream/main/python-binding.lines.txt)、[native](evidence/upstream/main/native-core.lines.txt) |
| S7 | expandable 分离虚拟地址与物理映射，布局和映射均有约束 | native 设计注释、官方 DevLog 的活跃块场景 | [设计注释](evidence/upstream/main/native-core.lines.txt)、[官方阅读笔记](evidence/document-notes.md) |
| S8 | 两版指定大小策略函数一致，捕获相关释放入口存在差异 | 固定 main 与 v2.15.0 对照 | [逐函数比较与哈希](evidence/source-comparison.json)、[稳定版](evidence/upstream/v2.15.0/native-core.lines.txt) |
| S9 | 教学预算下可复现 12 MiB 空闲仍无法满足 7 MiB 请求，释放邻块后可恢复 | 本篇原创 CPU 模型 | [模型输出](evidence/runs/05-model-lab.json)与[命令记录](evidence/runs/06-verification.json)，未作 GPU 容量实测 |

## 源码版本与获取

官方仓库为 [pytorch/pytorch](https://github.com/pytorch/pytorch)，默认分支为 main。实际获取采用已有缓存中的 fetch 和固定 Git 对象读取，缓存路径为 `.cache/upstream/pytorch`。[实际 fetch 输出](evidence/runs/00-upstream-fetch.txt)

- main 固定为 `2a2d844941bf776d1a49a8ff8730e90f5242f836`，提交时间为 2026-11-16 02:48:52 UTC。
- 最新稳定 release 为 `v2.15.0`，发布于 2026-10-28，对应 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。[官方 release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)
- 执行环境的 torch 为 2.15.0，构建 SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`，Python 为 3.12.16，macOS 27 arm64，CUDA 不可用。[环境与版本](evidence/versions.json)

固定链接、原文件 blob SHA-256、摘录 SHA-256 与行号区间统一列在清单中。8 组摘录合计 1,247 行，归档原文件许可。`.gitattributes` 对 `evidence/upstream/` 和原始测试先行日志 `evidence/runs/01-red-tests.txt` 设置空白检查例外，相关字节均有哈希核验。[源码清单](evidence/upstream/manifest.json) · [许可证](evidence/upstream/LICENSE)

## 官方文档与阅读笔记

实际阅读 PyTorch 2.15 CUDA semantics 的内存管理、stream 语义和 allocator 配置，Understanding CUDA Memory Usage 的 snapshot 状态与外部分配说明，以及 2026-06-01 官方 DevLog 的 segment、分裂、expandable 与活跃块场景。[文档阅读笔记](evidence/document-notes.md)

原始网页正文保留在发布方，本篇归档自己的阅读笔记、精确定位与相关许可允许的源码摘录。动态官方网页的本次访问日期与搜索记录另列。[研究记录](evidence/research-log.json)

## 论文范围

本篇研究 PyTorch 分配器的工程机制和统计口径，best-fit 与邻块合并按当前代码和原创合成状态验证。正文未介绍某篇论文提出的专门算法，所需一手证据由官方实现、API 文档和工程分析提供，因此没有引入无关论文或论文原文。

## 可复现交付

两道练习的起始代码、完整答案、不可变模型类型、验收测试与实验入口位于 `code/`。实际命令、退出码和代码哈希保存在 [运行清单](evidence/runs/06-verification.json)，代码独立审查和回归修正见 [代码审查](evidence/python-review.md)。

文章的图解由 [Mermaid 源文件](figures/allocator-layout.mmd) 重建。[完整性清单](SHA256SUMS)、[逐文件核验输出](evidence/runs/07-checksums.txt) 与 [交付检查](evidence/delivery-check.json) 一同归档。所有数值实验属于 CPU 合成模型，GPU 专属验证范围见正文与审查记录。[源码阅读笔记](evidence/source-notes.md)
