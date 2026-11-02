# 当前源码阅读笔记

日期：2026-11-02，Asia/Shanghai。源码与版本资料采集起始时间为 `11:15:49+08:00`。官方仓库 `https://github.com/vllm-project/vllm.git`，默认分支 main，完整 SHA `e8ec12c5edc3e14598bb122237d2798a3d4f3fa0`，提交时间 `2026-11-02T03:00:36Z`。

## 1. 来源与证据层次

缓存工作区检查为干净后，实际执行 `git fetch --depth=1 origin main`、`git switch --detach origin/main`，均退出 0。[versions.json](versions.json) 保存命令和输出。浅 fetch 的跟踪引用更新不涉及对本 Wiki 远端强制推送。

GitHub 版本元数据第一次查询遇到 TLS handshake timeout，随后原样重试成功；没有关闭 TLS 校验。当天最新稳定发行元数据为 `v0.32.0`，发行时间 `2026-10-18T06:50:48Z`，tag commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。没有单独分析稳定版实现。

13 份相关公开源码／文档原文共 669,945 字节，由 `git show <SHA>:<path>` 原样提取，逐文件命令、路径、大小、SHA-256 和永久链接见 [source-manifest.json](source-manifest.json)。包含原始 [Apache-2.0 LICENSE](upstream/vllm-project/vllm/LICENSE) 与版权头。原字节中的上游尾空格不改写，`.gitattributes` 只对 `evidence/upstream/**` 关闭 whitespace 检查，原创文件仍检查。

实际执行的是本篇 Python 标准库 hash／元数据模型。引擎、真实 KV、GPU 内核、整模型输出、访问控制和性能测试均未执行。

## 2. 从请求进入查找

读取路径依次为：

1. [EngineCore.step 685—710](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/engine/core.py#L685-L710)：先调度，再把计划交给 executor、处理输出。
2. [Scheduler 1106—1127](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/sched/scheduler.py#L1106-L1127)：对尚无计算进度的请求查找 prefix hit；[663—675](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/sched/scheduler.py#L663-L675) 区分本地常规查找和 connector 特殊路径。本文沿本地常规路径，不展开外部加载。
3. [KVCacheManager.get_computed_blocks 266—307](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_manager.py#L266-L307)：先检查允许读取，再取得请求上限，交给 coordinator；返回匹配块与长度等元数据。
4. [UnitaryKVCacheCoordinator 567—599](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_coordinator.py#L567-L599)：单组要求 hash 与块大小相同，用物理块大小作为 alignment。
5. [FullAttentionManager 789—800](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/single_type_kv_cache_manager.py#L789-L800)：基础阶段从头查连续完整块，遇 miss 停止。
6. [BlockPool.get_cached_block 202—228](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/block_pool.py#L202-L228)：每个 KV group 都应有可用块，组合 group ID 与 hash 查询。

查找循环中的“首个 miss 停止”是所选路径的策略，不是关于后缀索引不存在的数学证明。前块已驱逐、后块仍驻留时，哈希表允许存在后面的键；本篇测试故意构造这个情形，结果仍只返回连续前缀。

## 3. 指纹到底覆盖了什么？

[kv_cache_utils.py 707—741](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_utils.py#L707-L741) 的 `hash_block_tokens` 将 parent hash、本块 token tuple、extra keys 交给所选 hash 函数。首块使用 `NONE_HASH` 根，而不是把模型权重逐字节 hash 一遍。

`get_request_block_hasher` 953—1035 增量处理完整 hash 单元，更新 parent；这里的 hash 单元在进阶配置中未必等于物理块。`num_hash_lookahead_tokens` 非零时还把后续 token 放入身份，供需要额外依赖的 drafter 使用。本文严格因果推导只用于普通目标模型，不拿它覆盖所有投机状态。

extra keys 的实际组成：

- 592—609：LoRA name 与 path；路径字段用来区分同名指向不同适配器。它不是适配器权重内容的完整摘要，路径和文件内容的生命周期仍须正确管理。
- 612—635：直接传入 prompt embedding 时，按块原始 embedding 数据生成稳定指纹。
- 637—675：组合 LoRA、多模态标识、第一块的 cache salt、prompt embedding 键。后续块通过 parent 链继承首块 salt。

多模态标识本身应由正确的输入处理机制提供；如果调用方把不同有效输入错误标成同一身份，强 hash 不能补救。模型权重与布局还受引擎实例、缓存组和失效管理约束；不声称当前 hash tuple 已逐项包含全部模型计算配置。

`cache_salt` 只控制键空间，不能替代可信租户绑定、网络授权、加密或全部侧信道防护。源码字段与官方设计说明支持其隔离机制，正文不沿用文档中无条件“防止攻击”的强承诺。

`vllm/utils/hashing.py` 26—60：`sha256` 先用 Pickle，`sha256_cbor` 用 canonical CBOR；相同 SHA 算法不代表输入编码相同或跨环境键兼容。教师代码用固定 JSON 数组、自定义协议标签和显式 namespace，方便标准库练习，并不冒充上游格式。

## 4. 为什么要保留最后位置？

[Request.get_max_cache_hit_length 319—344](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/request.py#L319-L344) 从 `num_tokens-1`、请求自己的读取上限及调用方 bound 取最小。普通生成需要重算最后输入位置以取得 logits；完整块对齐又可使更多位置一起重算。

额外的 `skip_reading_prefix_cache`、prompt logprobs／指定 prompt scoring 起点会影响允许跳过的前缀，不能一概认为所有请求都走同一个 L-1 公式。正文公式明确限定普通初始生成、无更紧限制、单组完整块路径。

## 5. 生命周期与写入规则

- `KVCacheManager.allocate_slots` 569—608 先作容量检查、接入已有计算块，再分配新块，以保护准备复用的块。
- 610—624 可在计划阶段登记即将计算的可提交 token 块。登记元数据不独立证明 GPU 完成，实际安全性还取决于引擎执行顺序与依赖；本文没有重放这条异步链。
- `BlockPool.touch` 792—810 为命中块增加引用，零引用命中先离开空闲候选队列。
- `SingleTypeKVCacheManager.free` 591—599 按请求块逆序释放。
- `BlockPool.free_blocks` 816—847 对引用降为 0 的块区分是否有缓存键：无键块优先放队首，有键温缓存放队尾。索引此时不必删除。
- `get_new_blocks` 706—740 从空闲候选中取块，并在重新用途分配时清掉旧 hash 索引；`_maybe_evict_cached_block` 769—790 负责相应元数据清理。
- `is_block_writable` 812—814 不只要求 `ref_cnt==1`，还要求没有登记 hash；“只剩一个使用者”不足以允许随意修改已缓存内容。

当前引擎允许同 hash 的多个物理副本。本文 toy pool 也支持，但自己选择优先活跃副本，再按 ID，以避免误报空间不足；上游 `get_cached_block` 描述返回可用副本，本文没有把教学选择优先级宣称为其具体策略。

## 6. 当前细粒度路径，不能被基础公式抹掉

当前 `CacheConfig.prefix_match_unit`（203—214）说明 hash／匹配边界可以比物理块细。`partial_hash_hits_enabled`（kv_cache_utils 764—803）检查 Mamba align 状态和各缓存组管理器支持；coordinator 690—716 决定是否用 hash 粒度对齐。`FullAttentionManager.find_longest_cache_hit` 的 802—823 另有整块之后的内部边界探测，`BlockPool.cache_partial_block` 在 486 起登记相应元数据。

本篇基础练习将物理块、hash 单元和 hit alignment 全设为 B；没有实现部分块、混合模型或 checkpoint 存储。官方设计概述中的“只缓存 full blocks”不足以覆盖所有当前分支，应以条件代码为准。相应原文完整保存在 [本地源目录](upstream/vllm-project/vllm/)。

## 7. 本篇验证的准确边界

实际运行 29 个标准库测试与 lab，验证哈希身份变化、查找策略、引用计数、温缓存、驱逐、同 key 副本和失败原子性。测试从活跃请求重算引用而非复用答案内部计数，并有独立进程 hash seed 对照。

没有运行引擎、GPU、真实 KV 内容、整模型缓存开关前后数值比较、跨实例传输、性能或认证隔离测试。正文的容量数字为指定模型几何下的手算估计，不是显存测量。原始输出见 [tests.txt](runs/tests.txt)、[lab.json](runs/lab.json) 与 [summary.json](runs/summary.json)。
