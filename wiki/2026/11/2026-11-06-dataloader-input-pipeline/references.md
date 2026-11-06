# 参考资料与证据

主题为训练输入流水线的瓶颈定位。2026-11-06 以 Asia/Shanghai 时间实际检索官方资料，并更新 PyTorch 官方默认分支。源码永久链接与归档 SHA 对应，网页检索的缓存获取日期单独保留。

## 核心论断与来源

| 编号 | 需要核验的论断 | 一手定位 | 本地证据 | 适用范围 |
| --- | --- | --- | --- | --- |
| S1 | 单进程与多进程取数路径不同 | `dataloader.py` 的两类迭代器与 `_next_data` | [main DataLoader](evidence/upstream/pytorch/main/torch/utils/data/dataloader.py) | 固定 main 与稳定版本对照 |
| S2 | map-style worker 读取后在 worker 内 collate | `_worker_loop`、`_MapDatasetFetcher.fetch` | [worker](evidence/upstream/pytorch/main/torch/utils/data/_utils/worker.py)、[fetcher](evidence/upstream/pytorch/main/torch/utils/data/_utils/fetch.py)、[collate](evidence/upstream/pytorch/main/torch/utils/data/_utils/collate.py) | 自动合批的普通索引式 dataset |
| S3 | 预取乘积约束任务，结果可能等待有序交付 | `_reset`、`_try_put_index`、`_next_data` | [源码笔记](evidence/source-reading.md) | 区分任务槽位与可消费就绪批数 |
| S4 | 有效 pinning 有设备条件，多进程可用主进程线程 | `_BaseDataLoaderIter` 与 `_pin_memory_loop` | [pinning](evidence/upstream/pytorch/main/torch/utils/data/_utils/pin_memory.py)、[Memory.cpp](evidence/upstream/pytorch/main/aten/src/ATen/native/Memory.cpp) | CUDA 路径解释与 CPU 实验分开 |
| S5 | stream 顺序、非阻塞调用与设备重叠需要分别核验 | 官方 CUDA 语义与 pinned-memory 教程 | [文档阅读笔记](evidence/documentation-reading.md) | 设备分析依据，实际 CPU 实验不开启 pinning |
| S6 | worker 读取期间仍被占用，队列只能覆盖有限暂停 | worker 串行路径与本文资源预算推导 | [答案](code/answers.py)、[回归测试](code/test_answers.py) | 教学下界与孤立暂停模型 |
| S7 | CPU 后台取数能够利用合成消费的等待时间 | 本机真实 DataLoader 三轮对照 | [原始观测](evidence/runs/lab.json) | 小文件、人工等待、缓存未控制的特定配置 |

固定 main 的 [DataLoader](https://github.com/pytorch/pytorch/blob/b79f21bab5e9e221b05bc903f7d8171a8c04a8a0/torch/utils/data/dataloader.py)、[worker](https://github.com/pytorch/pytorch/blob/b79f21bab5e9e221b05bc903f7d8171a8c04a8a0/torch/utils/data/_utils/worker.py)、[fetcher](https://github.com/pytorch/pytorch/blob/b79f21bab5e9e221b05bc903f7d8171a8c04a8a0/torch/utils/data/_utils/fetch.py)、[pinning](https://github.com/pytorch/pytorch/blob/b79f21bab5e9e221b05bc903f7d8171a8c04a8a0/torch/utils/data/_utils/pin_memory.py) 与 [ATen Memory](https://github.com/pytorch/pytorch/blob/b79f21bab5e9e221b05bc903f7d8171a8c04a8a0/aten/src/ATen/native/Memory.cpp) 为实现依据。

## 源码与版本

官方远端为 `https://github.com/pytorch/pytorch.git`。通过 GitHub 仓库 API 核验默认分支为 main，在干净的既有缓存执行 `git fetch --depth 1 origin main`，退出码为 0。固定提交为 `b79f21bab5e9e221b05bc903f7d8171a8c04a8a0`，提交时间为 `2026-11-06T02:08:49Z`。从这个 SHA 用 `git show SHA:path` 读取，缓存工作区的旧 checkout 与今日读取版本分别记录。

GitHub latest release API 返回稳定版本 v2.15.0，发布于 `2026-10-28T17:09:49Z`，tag 提交为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。本机安装包报告 2.15.0，构建 Git SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。安装包与稳定 tag 的所选 DataLoader Python 文件逐项比对结果写在清单中，版本号与完整仓库提交分别记录。[稳定发布](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)

归档保留 9 个上游文件，包括 BSD 风格 [LICENSE](evidence/upstream/pytorch/LICENSE)。镜像内容逐文件与固定 Git blob 核对，记录 SHA-256。main 与 v2.15.0 的相关文件比较见 [版本差异](evidence/stable-main.diff)，具体命令与结果见 [版本记录](evidence/versions.json) 和 [来源清单](evidence/source-manifest.json)。

## 文档实际阅读

| 官方资料 | 本次读取与 provider 获取日期 | 阅读范围 |
| --- | --- | --- |
| [torch.utils.data 2.15](https://docs.pytorch.org/docs/2.15/data.html) | 2026-11-06 请求，provider 2026-11-02 获取 | 单／多进程、参数、平台 spawn、内存与有序交付 |
| [CUDA semantics 2.15](https://docs.pytorch.org/docs/2.15/notes/cuda.html) | 2026-11-06 请求，provider 2026-11-02 获取 | stream 内顺序、跨 stream 依赖、计时与异步 |
| [Pinned memory 与 non_blocking 教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) | 2026-11-06 请求，provider 2026-11-05 获取 | 页锁定准备成本、主机同步、并发传输条件与生命周期 |

检索通过 Parallel Search，再读取指定官方页面，实际 fetch 的 errors 为空。检索入口曾返回旧版资料，正文采用指定的 2.15 文档及固定源码。网页全文未复制，归档自己的阅读笔记与精确定位，PyTorch data 文档的源码版本另完整归档。获取信息见 [research-log.json](evidence/research-log.json)。

## 推导与论文范围

本篇解释运行时数据加载机制和基础资源容量，没有介绍某一篇论文提出的特定算法。官方 API、实现及实际程序足以核验这些语义。容量取最大值和队列覆盖公式是本文明确假设下的教学推导，依靠手算与代码测试检验。

因此本篇没有另行引用或归档无关论文。H2D 并发条件来自官方设备语义，CPU 合成实验验证范围单独记录。

## 复现与交付

题目、完整实现与运行命令见 [code/README.md](code/README.md)。真实命令、退出码、测试与实验范围见 [runs/summary.json](evidence/runs/summary.json)。独立 Python 审查见 [python-review.md](evidence/python-review.md)，两轮技术与教学自审见 [review.md](review.md)。

图源为本文自绘 Mermaid，服务于队列和进程边界说明。文件完整性见 [SHA256SUMS](evidence/SHA256SUMS)，最终链接与交付检查见 [delivery-check.json](evidence/delivery-check.json)。
