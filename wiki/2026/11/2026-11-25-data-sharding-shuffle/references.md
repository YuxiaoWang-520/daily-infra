# 参考资料与证据

主题：训练数据的逻辑采样、物理 shard 归属、随机性与已提交进度。实际源码核验时间为 2026-11-25T11:08:04+08:00；检索与运行均在当天进行，在线提取的缓存日期另列。

## 关键论断与一手来源

| 编号 | 论断与精确定位 | 一手来源 | 本地归档与限制 |
| --- | --- | --- | --- |
| S1 | 固定大小数据集；seed+epoch、补齐／丢尾、按 rank 间隔取样；set_epoch | [PyTorch DistributedSampler，main 固定 SHA，L66—157](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/distributed.py#L66-L157) | [main 片段](evidence/upstream/pytorch/main-distributed.py-L66-L157.txt)、[stable 片段](evidence/upstream/pytorch/v2.15.0-distributed.py-L66-L157.txt)；不保证数据内容被自动锁定 |
| S2 | 本地 Sampler 再按 batch_size 合批，独立处理尾部 | [BatchSampler，L333—354](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/sampler.py#L333-L354) | [原样片段](evidence/upstream/pytorch/main-sampler.py-L333-L354.txt)；正文批次数公式限 map-style 与固定合批 |
| S3 | 普通 IterableDataset 不接收自定义 sampler／batch_sampler；IterDataPipe 有单独分支 | [DataLoader 构造分支，L320—374](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/dataloader.py#L320-L374) | [原样片段](evidence/upstream/pytorch/main-dataloader.py-L320-L374.txt)；不外推到其他数据加载库 |
| S4 | 先预取任务、返回数据时补投；派发不是训练完成 | [启动预取，L1234—1278](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/dataloader.py#L1234-L1278)、[派发与处理，L1532—1569](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/dataloader.py#L1532-L1569) | [启动片段](evidence/upstream/pytorch/main-dataloader.py-L1234-L1278.txt)、[补投片段](evidence/upstream/pytorch/main-dataloader.py-L1532-L1569.txt)；实际 CPU 安装包探针在 runs 下 |
| S5 | worker 初始随机性与普通持久 worker 重置路径不同 | [base_seed，L686—709](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/dataloader.py#L686-L709)、[初始化，L274—282](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/_utils/worker.py#L274-L282)、[重置，L338—355](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/_utils/worker.py#L338-L355) | [base_seed](evidence/upstream/pytorch/main-dataloader.py-L686-L709.txt)、[初始化](evidence/upstream/pytorch/main-_utils-worker.py-L274-L282.txt)、[重置](evidence/upstream/pytorch/main-_utils-worker.py-L338-L355.txt)；没有声称完整保存 worker RNG |
| S6 | map fetcher 按主进程索引取数据；iterable fetcher 在本副本内判断尾批 | [两种 fetcher，L20—57](https://github.com/pytorch/pytorch/blob/19f0c2704b8086f392172c18d37f3557ecff3aef/torch/utils/data/_utils/fetch.py#L20-L57) | [原样片段](evidence/upstream/pytorch/main-_utils-fetch.py-L20-L57.txt)；普通 CPU 功能验证，不是吞吐测试 |
| D1 | 数据类型、主进程派发、worker 副本、prefetch_factor、in_order | [torch.utils.data，main 在线文档](https://docs.pytorch.org/docs/main/data.html) | [阅读笔记](evidence/reading-notes.md)、[提取元数据](evidence/web-retrieval.json)；服务返回 acquired_at 为 2026-08-03，当前实现另经 Git 核验 |
| D2 | 跨 PyTorch release、commit、平台与 CPU／GPU 不保证完全复现 | [Reproducibility，main 在线文档](https://docs.pytorch.org/docs/main/notes/randomness.html) | [阅读笔记](evidence/reading-notes.md)、[提取元数据](evidence/web-retrieval.json)；acquired_at 为 2026-11-21 |

## 源码获取、版本与许可证

- 官方远端为 `https://github.com/pytorch/pytorch`，GitHub API 当天返回默认分支 `main`。
- 当天实际 fetch 后固定 main 为 `19f0c2704b8086f392172c18d37f3557ecff3aef`。只更新缓存引用，没有切换或清理其已有 checkout。
- 最新 release API 返回 `v2.15.0`，发布于 `2026-10-28T17:09:49Z`，tag commit 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。
- 已比较四个完整文件：`distributed.py`、`sampler.py`、`_utils/fetch.py`、`_utils/worker.py`，main／tag 字节相同。DataLoader 的相关 main 分支按表逐段核验，没有推断其他路径无差异。
- 实际运行包为 PyTorch 2.15.0，构建 commit 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。运行包、稳定 tag 和默认分支的身份分别保留。
- [source-update.json](evidence/source-update.json) 保存获取命令及实际结果摘要；[source-manifest.json](evidence/source-manifest.json) 固定每段路径、原始行号、完整文件与片段 SHA-256、固定 commit URL。
- 11 段源码按原字节存档，带原始文件位置。上游归属与完整许可见 [PyTorch LICENSE](evidence/upstream/pytorch/LICENSE)。这些片段用于解释实现，不是从该 checkout 编译出的运行包。

## 论文与推导范围

本篇没有讲述一个特定论文提出的训练算法。均分、取余、间隔索引、请求开销相加和有限状态重放属于本文直接说明的基础模型；PyTorch 具体采样语义由官方实现核对。没有把 `randperm` 的内部实现、流式 shuffle 质量或收敛结论写成已经研究的论文算法，因此不列无关论文，也不以论文替代接口证据。

`rank_indices` 和 `DataPlan` 是教学实现，输入范围比上游接口更窄；`assigned_shards` 是静态等 worker 数的整数分配例子；SHA-256 合成值是便于核对的无状态增强替身。请求开销模型完全使用明示假设，没有外部性能数据。

## 验证与交付

[测试与命令清单](evidence/runs/verification.json) 记录本机解释器、依赖版本、命令、退出码及 5 个 Python 文件摘要；[results.json](evidence/runs/results.json) 记录真实 CPU worker 行为。[技术核验](evidence/technical-audit.json)、[独立代码复核](evidence/python-review.json)、[两轮审查](review.md) 和 [交付检查](evidence/delivery-check.json) 分别界定其范围。

运行记录公开的是合成 ID、整数值和状态，不包含真实训练数据、内部日志、访问凭据或设备性能指标。没有 GPU、多 rank、远程存储、吞吐、跨版本兼容或崩溃注入验收。
