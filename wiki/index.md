# Wiki 索引

按推荐学习顺序阅读；每篇目录包含正文、参考资料、代码练习、答案与审查记录。尚未通过审查的草稿不列入下表。

| 序号 | 发布日期（Asia/Shanghai） | 核心问题与文章 | 领域 | 前置阅读 |
| --- | --- | --- | --- | --- |
| 01 | 2026-10-22 | [Tensor 的 shape 一样，为什么 view、复制和访问方式仍然不同？](2026/10/2026-10-22-tensor-shape-stride-contiguity/README.md) | Tensor、内存布局、PyTorch、Attention 布局 | 无，从零开始 |
| 02 | 2026-10-22 | [广播看似没有复制数据，为什么仍可能增加计算或访存？](2026/10/2026-10-22-broadcasting-cost/README.md) | 广播、零 stride、物化、梯度与分块 | 第 01 篇 |
| 03 | 2026-10-22 | [同一个计算，为什么 FP32、BF16、FP16 的结果不同？](2026/10/2026-10-22-floating-point-formats/README.md) | 浮点格式、范围与精度、累加、混合精度与数值验收 | 第 01—02 篇 |
| 04 | 2026-10-22 | [一个算子究竟受算力还是内存带宽限制？](2026/10/2026-10-22-roofline-basics/README.md) | FLOPs、带宽、算术强度与 Roofline | 第 01—03 篇 |
| 05 | 2026-10-22 | [GPU 如何把一次算子调用分配给线程、warp 和 block？](2026/10/2026-10-22-gpu-parallel-mapping/README.md) | 线程映射、warp、mask、访存与资源 | 第 01、04 篇 |
| 06 | 2026-10-23 | [GPU 计时为什么需要理解异步执行和同步？](2026/10/2026-10-23-gpu-asynchronous-timing/README.md) | stream、event、依赖、同步与计时 | 第 04—05 篇 |
| 07 | 2026-10-23 | [为什么矩阵乘法需要分块与数据复用？](2026/10/2026-10-23-gemm-tiling/README.md) | 块乘代数、数据复用、流量模型与资源 | 第 01、04—06 篇 |
| 08 | 2026-10-23 | [Tensor Core 怎样使用，什么条件决定它是否有效？](2026/10/2026-10-23-tensor-core-conditions/README.md) | 存储与计算类型、MMA、padding、编译与指令 | 第 03、05、07 篇 |
| 09 | 2026-10-23 | [算子融合什么时候划算，什么时候会适得其反？](2026/10/2026-10-23-kernel-fusion/README.md) | 中间张量、启动、资源压力与编译决策 | 第 02、04、06—08 篇 |
| 10 | 2026-10-23 | [Attention 的计算与中间张量从哪里来，为什么需要分块？](2026/10/2026-10-23-attention-online-softmax/README.md) | Attention、在线归一化、分块与当前 FlashAttention | 第 03、07—09 篇 |
| 11 | 2026-10-26 | [生成一个 token 时，哪些数据可以复用？](2026/10/2026-10-26-prefill-decode-kv-cache/README.md) | 自回归生成、Prefill／Decode、逐层 KV Cache | 第 01、03、10 篇 |
| 12 | 2026-10-27 | [KV Cache 怎么算大小，MHA、GQA、MQA 各改变了什么？](2026/10/2026-10-27-kv-cache-capacity-gqa/README.md) | KV 容量、分组 Attention、缓存 dtype 与 TP 复制 | 第 03、10—11 篇 |
| 13 | 2026-10-28 | [KV Cache 为什么需要分页，逻辑 token 怎样找到物理块？](2026/10/2026-10-28-paged-kv-block-table/README.md) | 分页 KV、块表、槽映射、页尾余量与回收复用 | 第 10—12 篇 |
| 14 | 2026-10-29 | [请求为何需要逐步调度，batch 大小由什么决定？](2026/10/2026-10-29-continuous-batching-scheduling/README.md) | 连续批处理、请求准入、token 预算、KV 容量与延迟口径 | 第 11—13 篇 |
| 15 | 2026-10-30 | [长 Prompt 为什么会干扰其他请求，如何拆分 Prefill？](2026/10/2026-10-30-chunked-prefill-interference/README.md) | 分块 Prefill、跨块因果关系、token 预算与 TTFT／ITL 取舍 | 第 10—11、13—14 篇 |
| 16 | 2026-11-02 | [多个请求什么时候能安全复用同一段前缀？](2026/11/2026-11-02-prefix-cache-safe-reuse/README.md) | 前缀缓存、哈希身份、命中边界、共享块生命周期与隔离 | 第 11、13—15 篇 |
| 17 | 2026-11-03 | [训练的梯度为什么需要通信，通信能否与计算重叠？](2026/11/2026-11-03-ddp-gradient-overlap/README.md) | 数据并行、有效分母、梯度桶、AllReduce、通信重叠与累积 | 第 03、06、09 篇 |
| 18 | 2026-11-04 | [参数之外，训练显存究竟花在哪里？](2026/11/2026-11-04-training-memory-accounting/README.md) | 训练容量、Adam 状态、混合精度、激活、storage 生命周期与 allocator 口径 | 第 01、03、17 篇 |

下一篇候选：用重算换显存的代价如何计算？

2026-10-22 起按用户要求集中完成起步前 10 篇，使用实际归档日期；日常定时任务仍按工作日每天一篇执行。范围见 [本次任务说明](../docs/BOOTSTRAP10.md)。

完整问题链见 [选题路线](../ROADMAP.md)。
