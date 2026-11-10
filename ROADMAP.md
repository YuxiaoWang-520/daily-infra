# 选题路线

每天回答一个能在工程实践中用上的问题，按照前置知识推进。下面是初始路线，不是内容边界；每次研究都可以补充重要分支，新增问题须写明它服务的场景和所需基础。

## 起步阶段：先建立能读代码和判断瓶颈的基础

| 顺序 | 主问题 | 绑定的工程场景 |
| --- | --- | --- |
| 01 | [Tensor 的 shape 一样，为什么 view、复制和访问方式仍然不同？](wiki/2026/10/2026-10-22-tensor-shape-stride-contiguity/README.md) | 转置、view、隐式拷贝与算子输入布局 |
| 02 | [广播看似没有复制数据，为什么仍可能增加计算或访存？](wiki/2026/10/2026-10-22-broadcasting-cost/README.md) | 批处理、广播运算、扩展视图与物化 |
| 03 | [同一个计算，为什么 FP32、BF16、FP16 的结果不同？](wiki/2026/10/2026-10-22-floating-point-formats/README.md) | 累加误差、数值范围、训练稳定性与验证 |
| 04 | [一个算子究竟受算力还是内存带宽限制？](wiki/2026/10/2026-10-22-roofline-basics/README.md) | FLOPs、字节数、算术强度与 Roofline |
| 05 | [GPU 如何把一次算子调用分配给线程、warp 和 block？](wiki/2026/10/2026-10-22-gpu-parallel-mapping/README.md) | 并行映射、同步、资源限制与占用率 |
| 06 | [GPU 计时为什么需要理解异步执行和同步？](wiki/2026/10/2026-10-23-gpu-asynchronous-timing/README.md) | warmup、stream、事件计时与端到端延迟 |
| 07 | [为什么矩阵乘法需要分块与数据复用？](wiki/2026/10/2026-10-23-gemm-tiling/README.md) | GEMM、合并访存、片上存储和 tile 选择 |
| 08 | [Tensor Core 怎样使用，什么条件决定它是否有效？](wiki/2026/10/2026-10-23-tensor-core-conditions/README.md) | 数据类型、形状、对齐、累加与实际指令路径 |
| 09 | [算子融合什么时候划算，什么时候会适得其反？](wiki/2026/10/2026-10-23-kernel-fusion/README.md) | 中间张量、启动开销、编译器与资源压力 |
| 10 | [Attention 的计算与中间张量从哪里来，为什么需要分块？](wiki/2026/10/2026-10-23-attention-online-softmax/README.md) | 基础 Attention、online softmax 与 FlashAttention |
| 11 | [生成一个 token 时，哪些数据可以复用？](wiki/2026/10/2026-10-26-prefill-decode-kv-cache/README.md) | 自回归生成、Prefill／Decode 与 KV Cache 生命周期 |
| 12 | [KV Cache 怎么算大小，MHA、GQA、MQA 各改变了什么？](wiki/2026/10/2026-10-27-kv-cache-capacity-gqa/README.md) | 长上下文、并发容量与模型结构选择 |
| 13 | [KV Cache 为什么需要分页，逻辑 token 怎样找到物理块？](wiki/2026/10/2026-10-28-paged-kv-block-table/README.md) | 推理引擎的块管理、碎片与地址映射 |
| 14 | [请求为何需要逐步调度，batch 大小由什么决定？](wiki/2026/10/2026-10-29-continuous-batching-scheduling/README.md) | continuous batching、TTFT、ITL 与吞吐权衡 |
| 15 | [长 Prompt 为什么会干扰其他请求，如何拆分 Prefill？](wiki/2026/10/2026-10-30-chunked-prefill-interference/README.md) | chunked prefill、调度预算与尾延迟 |
| 16 | [多个请求什么时候能安全复用同一段前缀？](wiki/2026/11/2026-11-02-prefix-cache-safe-reuse/README.md) | prefix caching、命中条件、隔离与失效 |
| 17 | [训练的梯度为什么需要通信，通信能否与计算重叠？](wiki/2026/11/2026-11-03-ddp-gradient-overlap/README.md) | 数据并行、梯度桶、AllReduce 与慢节点 |
| 18 | [参数之外，训练显存究竟花在哪里？](wiki/2026/11/2026-11-04-training-memory-accounting/README.md) | 梯度、优化器状态、激活与容量估算 |
| 19 | [用重算换显存的代价如何计算？](wiki/2026/11/2026-11-05-activation-checkpointing/README.md) | activation checkpointing、随机性与训练吞吐 |
| 20 | [训练在等数据，怎样判断瓶颈在磁盘、CPU 还是传输？](wiki/2026/11/2026-11-06-dataloader-input-pipeline/README.md) | DataLoader、预取、pinned memory 与输入流水线 |
| 21 | [FSDP／ZeRO 怎样分片参数、梯度与优化器状态？](wiki/2026/11/2026-11-09-fsdp-zero-state-sharding/README.md) | 训练状态分片、通信与容量，前置为第 17—19 篇 |
| 22 | [训练中断后，checkpoint 怎样恢复一致的进度？](wiki/2026/11/2026-11-10-training-checkpoint-recovery/README.md) | 模型、优化器、随机性和数据进度，前置为第 18—21 篇 |
| 23 | AllReduce 的数据怎样沿链路流动，为什么增加 GPU 后通信反而更慢？ | 通信量、环／树直觉与链路瓶颈，前置为第 04、17、21 篇 |

每篇既要独立可读，也要链接已有前置文章。较新的实现可以作为当天的进阶部分，但不能让读者先理解尚未解释的术语。

## 持续扩展的知识地图

| 领域 | 值得持续追问的问题 | 常见落点 |
| --- | --- | --- |
| 算子与数值 | 如何从访存、并行和数值要求选择实现？怎样证明优化后仍正确？ | GEMM、归约、norm、softmax、稀疏与通信算子 |
| 编译与执行 | eager、图捕获、融合、动态 shape 如何影响执行？ | PyTorch／ATen、Triton、编译缓存、CUDA Graph |
| 推理引擎 | 请求如何从 API 到调度器、执行器和 token 输出？ | vLLM、SGLang、TensorRT-LLM 等官方当前实现 |
| Cache 与内存 | 谁拥有缓存、如何分配回收、何时复用与搬迁？ | KV、prefix cache、allocator、碎片、offload 与分层缓存 |
| 训练并行 | DP、TP、PP、CP、SP、EP 怎样改变计算、通信与显存？ | Megatron、FSDP、ZeRO、MoE 路由与负载均衡 |
| 模型与算法 | 哪些结构或算法改变了系统的性能边界？ | Attention 变体、MoE、量化、稀疏、投机解码与多模态 |
| 存储与数据 | 数据、参数和 checkpoint 如何高效、可靠地流动？ | 数据格式、分片、对象存储、本地缓存、checkpoint 与恢复 |
| 网络与拓扑 | 一次 collective 经过哪些链路，为什么会成为瓶颈？ | PCIe、NVLink、RDMA、通信算法与拓扑感知放置 |
| 服务与资源 | 如何把单机能力变成满足 SLO 的线上服务？ | P／D 分离、并行部署、容量规划、自动扩缩容与多租户 |
| 可观测与可靠性 | 性能退化或训练不稳定时，先测什么、如何缩小范围？ | profiler、trace、排队、OOM、hang、重试、故障恢复与成本 |
| 前沿工程 | 新方法究竟解决了哪个瓶颈，改变了什么约束？ | 依据当天可核验的官方代码、论文与真实采用情况补充 |

## 后续选题方法

1. 优先补齐下一篇依赖的基础，避免用术语跳过推导。
2. 每轮回顾各领域覆盖情况，逐步扩展到完整训练与服务链路。
3. 用实际场景筛选新题：它会影响容量、吞吐、延迟、正确性、成本或可靠性中的哪一项？
4. 新论文和新版本只有在能说明核心问题时才进入主线；先核验代码与论文，再判断是否值得专题展开。
5. 日常推进不设最终篇数。已发布文章发现错误时追加勘误；实现变化时说明版本差异并链接新的专题。
