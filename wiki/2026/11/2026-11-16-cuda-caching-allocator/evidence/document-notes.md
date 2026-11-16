# 官方文档阅读笔记

访问日期为 2026-11-16，时区 Asia/Shanghai。以下内容为本次阅读后的中文笔记。

## CUDA semantics，PyTorch 2.15

实际阅读 Memory management、Optimizing memory usage with PYTORCH_ALLOC_CONF，以及 CUDA streams 的跨 stream 使用示例。文档区分 Tensor 分配与 allocator 保留，说明 empty_cache 释放未占用缓存的用途。当前配置名称为 PYTORCH_ALLOC_CONF，旧 CUDA 名称作为兼容别名。native 是文档所述默认 backend，cudaMallocAsync 使用另一种底层实现。[官方章节](https://docs.pytorch.org/docs/2.15/notes/cuda.html#memory-management)

文档把 max_split_size_mb 的调节条件关联到发生 OOM 且存在大量 inactive split 的工作负载，并说明成本依赖分配模式。跨 stream 示例要求建立数据依赖，同时记录后续使用，分配器生命周期与执行依赖分别处理。具体回收是否等待事件由本篇固定 native 代码确认。

## Understanding CUDA Memory Usage，PyTorch 2.15

实际阅读 snapshot API 的 Segment／Block 字段，以及 Identifying Non-PyTorch allocations。普通段的 total_size、块大小与三个状态可构成分层核算，active_awaiting_free 对应等待其他 stream 使用完成。外部分配的可见性需要额外证据，NCCL 等场景可使设备占用高于 allocator 记录。[官方文档](https://docs.pytorch.org/docs/2.15/torch_cuda_memory.html)

本篇练习采用自己的不可变、单池简化类型。真实 snapshot 中的设备、pool、地址、requested_size、frames 和 trace 还提供更多定位信息，分析真实输出时需要保留其范围与采样条件。

## When does fragmentation occur in the CUDA caching allocator?

作者为 Edward Yang，官方 DevLog 发布于 2026-06-01。实际阅读 Segments, blocks, and splitting、Without expandable segments，以及 Expandable segments don’t eliminate all fragmentation。它用分配顺序说明普通段之间的布局限制，并展示可扩展段内仍被存活分配分隔的空闲范围。[原文](https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/)

本篇自行构造 16 MiB 预算与 8 MiB 最小段的模型，明确标为教学配置，实验代码与数据由本机生成。论文范围记录在 references.md，源码版本与原文发布日期分别标注。
