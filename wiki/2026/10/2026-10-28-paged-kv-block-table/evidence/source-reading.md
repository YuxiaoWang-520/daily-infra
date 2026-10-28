# 当前源码阅读路径与边界

版本：vLLM `main`，`84e455367336b62547d1f1273e9b50bc7f8edb47`，2026-10-28 实际更新。这里只读公开源码，不代表运行了引擎。逐文件原文与哈希见 [manifest](source-manifest.json)。

## 从请求需要容量，到取得块号

1. `vllm/v1/core/sched/scheduler.py` 的 `Scheduler` 在 1424—1443 行调用 `KVCacheManager.allocate_slots`；本轮新增、已命中、lookahead、外部 KV、预留等分别传递。本文只抽取普通增长主线，没有把这些量统一叫作有效历史。
2. `vllm/v1/core/kv_cache_manager.py` 的 `allocate_slots` 在 578—607 行比较需求、watermark 与可用块，容量不足返回 `None`，随后经 coordinator 分配；735—737 行提供请求块号接口。`free` 在 628—637 行继续委托 coordinator。
3. `vllm/v1/core/kv_cache_coordinator.py` 的 `allocate_new_blocks`（315—345 行）向各缓存类型 manager 分派。多个缓存组、编码器和其他状态不是同一份统一公式。
4. `vllm/v1/core/single_type_kv_cache_manager.py` 的 `allocate_new_blocks`（370—410 行）先有共享尾页 CoW 路径，然后普通增长使用 `cdiv(num_tokens, block_size)-len(req_blocks)`，新增块后扩展请求表。本篇摘写明确省略 CoW。`free`（591—600 行）归还请求块，其普通路径逆序传递，以配合池的复用策略。
5. `vllm/v1/core/block_pool.py`：初始化在 159—192 行建立块与空闲队列，并移走编号 `0` 的 null block。706—740 行取新块并增加引用；816—849 行递减引用，零引用后根据缓存哈希和 caching 状态前插或后插。代码还有跨池、pinning 等路径，本文未实现它们。教学 FIFO 与从 `0` 起可用的编号只为使实验可复算，不能当作当前实际 ID 序列。

## 从块号到写入槽和元素地址

1. `vllm/v1/worker/gpu_model_runner.py`：2137—2152 行构造本轮绝对位置并调用 `compute_slot_mapping`。3822—3886 行将映射按 group／layer 提供；3869 行附近把无效 token 尾部填为 `-1`。
2. `vllm/v1/worker/block_table.py`：80—87 行分别创建 int32 块表与 int64 槽映射缓冲。`ComputeSlotMappingKernel.kernel` 的 351—376 行处理虚拟块／块内位置；上下文并行大小为 `1` 时可化简为正文公式。内核的 Triton tile 大小与 KV 页长是不同参数。
3. `vllm/v1/worker/gpu/block_table.py` 的 V2 `_compute_slot_mappings_kernel`：382—414 行显式区分 CP=1 与多 rank 分布，查物理块后生成槽，dummy／无效项产生 PAD。没有执行 V2，也未断言它在所有平台默认启用。
4. `vllm/v1/attention/backends/flash_attn.py`：1432—1466 行 `do_kv_cache_update` 从缓存视图取得 K／V 目标，调用 `reshape_and_cache_flash`。平台分派在 `fa_utils.py`（22—28 行），CUDA 选择 `vllm._custom_ops`。
5. `vllm/_custom_ops.py`：2638—2658 行的包装调用 `torch.ops._C_cache_ops.reshape_and_cache_flash`。`csrc/cache_kernels.cu`：335—366 行的相应 kernel 从槽号拆回块与偏移，跳过负槽，再结合 `block_stride` 和 `page_stride` 得到目标行；后续还根据 head 等布局处理元素。本篇的 slot 不是字节地址。

## 从池到 Attention 读取

`flash_attn.py` 中普通 dense 调用（1404—1424 行）传入缓存、块表、`seqused_k`、Query 位置范围与因果等参数；同文件 114—116 行的包装调用实际 FlashAttention 接口。这些信息足以确认读取方获得了分页缓存与有效性信息，不能推断所有设备实现的流量和指令，也不能要求先执行教学 gather。

`worker/utils.py` 的 `AttentionGroup` 在 291—329 行选择 kernel 页长，在 368—395 行按需转换块表。管理器页与 kernel 页不同的配置需要追踪实际 ID、页长与布局；本篇手算取两者一致。这里不是“所有后端总是只有一个 P”的证据。

`GPUModelRunner` 的缓存初始化在 6918—6945 行附近调用 `allocate_kv_cache`；`worker/utils.py` 的 564—624 行从配置大小分配底层缓冲并建立视图。它支持按布局组织多个层／组，不是每新增一个 token 就分配一块新 CUDA Tensor。请求在池内归还块也不等于销毁整个缓冲。

## 原始设计、API 与教学代码

归档的 `docs/design/paged_attention.md` 开头明确是历史说明。因此没有把其中的旧 kernel 路径、参数解释或当年默认值当作当前引擎证据。

PyTorch 2.9 的 SDPA 官方文档本次实际读取：`dropout_p=0.0` 关闭 dropout，非方形的 `is_causal=True` 使用左上角对齐。本篇末尾 Query 的所有给定 KV 都可见，普通 SDPA 不启用该因果开关。地址正确性先通过原始连续 K／V 的逐元素比较确认，再对照 Attention 输出；不是将两个相同的块表 gather 互相比。

教学答案只有一个 CPU FP64、单层单头、独占页池。它的 `read` 会 gather，分配采用 FIFO，没有 null block、引用共享、hash 缓存、CP、设备同步或多租户安全边界。生产路径的条件没有被这些 CPU 测试覆盖。
