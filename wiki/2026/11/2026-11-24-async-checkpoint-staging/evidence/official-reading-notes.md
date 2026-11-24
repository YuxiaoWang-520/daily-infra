# 官方资料阅读笔记

访问日：2026-11-24，Asia/Shanghai。检索用于定位当前一手文档，具体实现另由当日实际 fetch 的 Git 对象核验；没有复制整份网页或教程。

## DCP API：数据安全与返回类型

读了 [PyTorch 2.15 DCP API](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html) 的 `async_save`、`AsyncSaveResponse`、`AsyncStager`、`BlockingAsyncStager` 和 `DefaultStager` 说明。自己的理解：暂存要得到对后续训练更新隔离的表示，保存则消费这份表示；异步保存和异步暂存是两个维度。Future 结果、资源关闭和缓冲区复用，都要按实际实现契约处理。

文档协议描述中出现 `stage_data` 的称呼，但本次源码与现有安装包实际调用的方法为 `stage`。本文的完整练习按真实签名运行，没有直接照抄旧描述的方法名。API 仍标为实验性；单看类构造默认值不能覆盖入口中显式参数和 writer 选择。

文档对两个完成 Future 的正常语义与本篇正常路径一致，但它不是对所有异常分支的逐行证明。已失败 staging Future 的探针中，当前安装包的 staging 报告与 upload 报告不同。该现象另与 main／stable 的 already-done 分支核对，本文明确记录这个版本边界，不把它宣传成可用快照，也不替上游做修复或外部报告。

## 官方异步 checkpoint recipe

读了 [Asynchronous Saving with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) 的 overview、checkpoint management 和 pinned memory 部分。自己的理解：把存储工作移出主线程，仍要给 CPU buffer 留预算，尤其是 pinned memory；限制一份未完成请求能降低内存和复用复杂度。示例在下一次请求前等待上次 Future，与本文固定时长模型的一次一个策略相符。

CPU memory 应按实际节点上参与的 ranks 和持有快照数预算，不能把全局 world size 机械地乘到每个节点。本文 `q×sum(B_r)` 是有效载荷的初步估算，不包含完整进程内存、序列化缓存或所有格式开销。

教程的发布日期、最后验证日与当前 API 版本不相同，故本文用当日源码界定具体默认值。Parallel 对 API 页返回的缓存获取日为 2026-11-17，recipe 为 2026-11-24；本次读取日期是 2026-11-24，两者分别保留在 [元数据](web-retrieval.json)，没有把缓存页面说成当天实时抓取。

## Python Future 与清理

读了 [Python 3.12 concurrent.futures](https://docs.python.org/3.12/library/concurrent.futures.html) 的 executor shutdown 相关摘录，并在实验中直接检查失败任务的 done 与 result。`done` 不能独自表示成功；获取结果会传播失败。关闭线程池时等待任务与安全释放门控，是本篇正常／异常清理的基础。等待超时与取消任务是不同操作。

[检索元数据](futures-retrieval.json) 记录页面与 provider 获取日期；独立复核还实际注入提前 TimeoutError，观察门控释放、保存任务完成、线程退出和临时目录清理。此证据仅适用于受控本地实验，不把它说成可以强制取消任意卡住的存储 I/O。

## 模型与论文边界

本文没有介绍某篇论文提出的 checkpoint 调度算法或最优保存间隔。`S+max(kC,W)` 是一次一个策略、固定时长、下一次 staging 前等待旧保存的事件模型；`λB≤R` 与主机内存估算是基础容量关系。没有凭此计算真实故障下的最优间隔，也没有假装读了不相关论文。

公式、CPU 合成状态、DCP 实测与生产能力分开标注：真实实验只验证 40 字节有效载荷的快照和完成语义；不验证 GPU、分布式、I/O 吞吐或掉电持久性。
