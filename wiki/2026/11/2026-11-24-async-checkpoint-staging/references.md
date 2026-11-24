# 参考资料与证据

主题：异步 checkpoint 的暂存隔离、计算／保存重叠和完成边界。访问日为 2026-11-24，Asia/Shanghai；源码实际获取时间为 `2026-11-24T11:01:44.438331+08:00`。

## 关键论断与来源

| 编号 | 需要证实的论断 | 一手来源与定位 | 本地归档 | 证据边界 |
| --- | --- | --- | --- | --- |
| S1 | 入口选择 writer stager 或显式配置的 DefaultStager | `state_dict_saver.py::async_save` | [选择分支](evidence/upstream/pytorch/save-default-stager-and-state.txt)、[选项](evidence/upstream/pytorch/staging-options-defaults.txt) | 不把类构造默认值当作入口默认行为 |
| S2 | CPU 快照也需要独立存储，缓存会复用 buffer | `BlockingAsyncStager.stage`、`_create_cpu_state_dict` | [暂存与缓存](evidence/upstream/pytorch/blocking-stage-copy-and-cache.txt)、[新 storage](evidence/upstream/pytorch/new-cpu-tensor-storage.txt) | 本篇只实测 CPU 普通 Tensor、不缓存 |
| S3 | 保存线程先取得暂存结果，入口有单／双 Future 两种返回 | `async_save`、`save_wrapper` | [提交](evidence/upstream/pytorch/upload-dispatch-and-cleanup.txt)、[线程入口](evidence/upstream/pytorch/worker-waits-stage.txt)、[两个 Future](evidence/upstream/pytorch/two-futures-and-completion.txt) | 线程路径；未验证进程执行器 |
| S4 | 写数据完成后还需 finish／metadata 处理 | `save.write_data/finish_checkpoint`、`FileSystemWriter.finish` | [等待与完成](evidence/upstream/pytorch/write-wait-global-finish.txt)、[数据同步](evidence/upstream/pytorch/data-sync.txt)、[metadata](evidence/upstream/pytorch/metadata-publication.txt) | 本地功能验证，不证明所有存储故障持久性 |
| S5 | 异步保存有主机内存与未完成请求管理成本 | PyTorch 官方 async checkpoint recipe | [官方笔记](evidence/official-reading-notes.md)、[检索记录](evidence/web-retrieval.json) | 本文时间与 GiB 例子均为模型假设 |
| S6 | 已失败 staging Future 的报告与 upload 结果可能不同 | 固定 main／stable 的 already-done 分支、本机版本探针 | [main](evidence/upstream/pytorch/two-futures-and-completion.txt)、[stable](evidence/upstream/pytorch/stable-completion.txt)、[实测](evidence/runs/results.json) | 版本特定，未声称所有未来版本仍有此行为 |
| S7 | done 与成功不同，线程池需正确关闭 | Python Future／executor 语义及直接实验 | [官方笔记](evidence/official-reading-notes.md)、[语言文档记录](evidence/futures-retrieval.json)、[代码](code/dcp_demo.py) | 超时不能解释为强制取消任意 I/O |
| S8 | 一次一个策略的背压与 drain 可逐步复算 | 基础事件模型、逐步 oracle | [答案](code/answers.py)、[测试](code/test_answers.py)、[运行](evidence/runs/answers-run.txt) | 固定成本、无竞争；不是性能预测 |

## 源码与版本

- 官方远端：`https://github.com/pytorch/pytorch.git`，默认分支 main。
- 当日 fetched commit：`0d8b36742ee52e2926f5c349cdae9e57c82ed2f0`。
- 最新正式 release：[`v2.15.0`](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，发布时间 `2026-10-28T17:09:49Z`；tag commit 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。
- 已有安装包：`2.15.0`，构建 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`；运行结论归属于它，未把它等同于 tag 或 main。
- 获取命令、输出、时间与退出码：[source-update.json](evidence/source-update.json)。没有 checkout、reset 或清理缓存工作树。
- 17 个原样片段，共 476 行；每项的完整 SHA、源文件路径／行号、永久链接、源文件摘要与片段摘要：[source-manifest.json](evidence/source-manifest.json)。完整 [LICENSE](evidence/upstream/pytorch/LICENSE) 和 [文件头](evidence/upstream/pytorch/source-headers.json) 保留。
- 实际调用链、默认条件、异常分支与版本比较：[源码阅读笔记](evidence/source-reading-notes.md)。

## 文档与论文说明

实际读取 [PyTorch 2.15 DCP API](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)、[官方异步保存教程](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) 和 [Python 3.12 concurrent.futures](https://docs.python.org/3.12/library/concurrent.futures.html) 的相关部分。自己归纳的结论和缓存获取日期边界见 [官方阅读笔记](evidence/official-reading-notes.md)。没有归档完整版权网页，没有凭浮动 `/stable/` 结果证明当前发布版本。

本篇不介绍论文算法、最优 checkpoint 间隔或分布式快照协议；时间模型与容量关系是基础推导，因此没有强行引入无关论文。GPU 复制、Pinned／shared memory、进程保存器、多 rank 和掉电持久性不属于已执行验证。

## 复现与审查

[练习说明](code/README.md)、[起始代码](code/exercises.py)、[完整时间答案](code/answers.py)、[完整 DCP 答案](code/dcp_demo.py) 和 [运行入口](code/run_answers.py) 保留。真实执行环境、命令、退出码、日志和 5 个 Python 文件摘要在 [verification.json](evidence/runs/verification.json)。18 项测试在普通及优化模式通过，Ruff 检查通过；第一次未实现的失败测试与初次 lint 记录也保留。

最终公开状态、恢复值与临时文件名在 [results.json](evidence/runs/results.json)，stdout 在 [answers-run.txt](evidence/runs/answers-run.txt)。临时 checkpoint 由实验自行创建、加载、清理，未归档二进制文件，也没有从外部加载不可信 checkpoint。二进制未保存不影响公开小张量结果和代码的可复算性。

两轮自审与独立通用代码复核如实记录在 [review.md](review.md)、[Python 报告](evidence/python-review.json) 和 [self-review.json](evidence/self-review.json)。[技术核验](evidence/technical-audit.json)、[交付检查](evidence/delivery-check.json) 和 [SHA256SUMS](evidence/SHA256SUMS) 固定发布范围；摘要清单排除自身与交付报告，避免循环引用。
