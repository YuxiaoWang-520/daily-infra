# 一手资料阅读笔记

当天实际检索与 fetch：2026-11-25，Asia/Shanghai。以下为自行归纳的笔记，不复制整页文档。URL、提取时间与缓存时间见 [web-retrieval.json](web-retrieval.json)。

## 从采样到 worker

1. `DistributedSampler.__init__` 的默认副本数和 rank 来自默认进程组；本篇显式传值，混合并行时由调用方选择 DP 组内值。固定数据集大小和稳定索引映射是其使用前提。
2. 本次 main 和 v2.15.0 的 `distributed.py` 整文件相同。长度先向上取整补齐或向下取整截断，`__iter__` 才用 `seed+epoch` 的独立 Generator 生成排列。补齐可循环多次，`N<R` 不能只追加一小段。
3. rank 切片取的是位置，补齐位置可以指向相同 ID。因此“各 rank 子集 exclusive”的概述不能被解释为补齐模式下全局 ID 从不重复。正文用具体反例说明。
4. `BatchSampler` 的尾部判断是第二个决策：map-style 本地 sampler 长度除以 batch 大小。两个 drop_last 不能被当成相同开关。
5. DataLoader 构造器区分 IterableDataset 与 map-style。普通 IterableDataset 拒绝自定义 sampler 和 batch_sampler；IterDataPipe 另有图级处理分支，不能套用为全部流式 Dataset 的自动分片保证。
6. `_MapDatasetFetcher` 使用收到的索引列表；有 `__getitems__` 时可批量读取，否则逐 ID 读取。`_IterableDatasetFetcher` 在自己的 iterator 上取数并判断不完整批次。多进程时这个判断处于 worker 副本内。

## 从预取到恢复

1. `_reset` 初始化 send／receive 任务编号并投递 `prefetch_factor×num_workers` 次。`_try_put_index` 先调用 `_next_index`，之后才将索引排入 worker 队列；所以 sampler 游标已前进不代表读取已完成。
2. `_process_data` 在返回结果附近继续补投任务；训练循环刚拿到第二批时，sampler 可能已经提供第六批的索引。真实计数器观察到 4→6，并未读取框架私有计数作为保存 API。
3. `committed_batches=2` 是本文对这份一致状态的应用合同。探针只重建输入，未执行真实 optimizer.step；更新、累积、失败或 AMP 跳步的生产语义应由训练系统定义。
4. 计划身份加入具体 ID 是防止“配置相同而 RNG 实现或顺序变化”静默接续。它不自动验证数据内容真实性，也不认证 token；生产 dataset_revision 必须关联不可变 manifest。
5. 普通随机增强不能只靠重新设置初始 seed 恢复中途状态。worker 初始化当前会种 Torch、Python random 和可用的 NumPy，但自定义 Generator 等仍由数据代码负责。普通持久 worker 的 `_ResumeIteration` 分支重新建 fetcher，图式 DataPipe 有共享种子处理，不等同于重新启动全部初始化逻辑。
6. 本文用稳定键生成合成值，跨 worker 数重放成立的前提是值不依赖调度。实际数据增强需固定变换版本、参数、ID 身份及是否区分补齐出现位置。

## 文档缓存与证据边界

Parallel 返回 data 页标题为 main 文档，acquired_at 为 2026-08-03；randomness 页为 2026-11-21。搜索结果还出现若干旧稳定版本，不把搜索标题当成本日最新 release。本文的当前源代码版本和 release 信息来自实际 Git fetch 与 GitHub API。

官方 data 文档支持 map-style 索引由主进程生成、IterableDataset 副本需分配、每 worker 丢尾、prefetch_factor 的批次数含义，以及 `in_order=False` 的复现与分布风险。正文对应结论再与固定源码和 CPU 探针互证。

Reproducibility 页明确区分同一环境的随机源控制和跨版本／平台完全复现。本文只使用这个范围提醒及 DataLoader 随机性说明，没有借该页其他算子细节扩展结论。

物理 shard 请求成本是自行列出的串行模型：1,024 次固定开销加 1 GiB 搬运，和一次固定开销加同样搬运量比较。没有现实硬件测量，也没有从有限局部 shuffle 推断收敛等价。
