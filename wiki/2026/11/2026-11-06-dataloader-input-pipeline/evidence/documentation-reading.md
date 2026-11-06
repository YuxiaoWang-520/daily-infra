# 官方文档阅读笔记

2026-11-06 实际请求官方 2.15 数据文档、CUDA 语义与传输教程。provider 返回的 acquisition 日期分别为 11 月 2 日、11 月 2 日和 11 月 5 日。实现细节另用当日获取的固定 Git commit 核验。

## DataLoader 2.15

阅读单进程／多进程、平台相关行为、内存 pinning 与构造参数。

默认单进程加载在主进程执行数据访问。多进程方式把 dataset、变换和 collate 交给 worker，允许主进程消费与取数并行。spawn 会启动新解释器并导入入口，dataset 与相关函数需可序列化，模块顶层定义和 main guard 是本篇实验的组织方式。

参数文档区分每个 worker 的 prefetch factor 与总任务规模，正 worker 时默认 factor 2。persistent workers 保留 dataset 实例，in_order 的默认值及乱序风险都需结合样本分布与可复现性理解。Python 对象的多进程副本可能显著增加内存。[官方 DataLoader 文档](https://docs.pytorch.org/docs/2.15/data.html)

## CUDA semantics 2.15

阅读 asynchronous execution 与 CUDA streams。

stream 内有顺序，跨 stream 需要显式同步依赖。CPU 提交返回与设备完成是不同测量时刻。使用 side stream 时既要建立消费者等待，也要管理 allocator 与缓冲区生命周期。设备 event 和完整边界同步用于相应的计时目标。[官方 CUDA 语义](https://docs.pytorch.org/docs/2.15/notes/cuda.html)

## pinned memory 与 non_blocking 教程

阅读内存页固定、异步传输、并发条件和输入修改风险。

手动 pinning 涉及新分配与复制，主线程中的准备可能成为成本。DataLoader 多进程路径能通过独立线程承担这部分工作。non_blocking 参数影响主机同步，设备重叠仍需合适的 stream、硬件与页锁定来源。异步读取完成前保留稳定源数据，并核验目标消费依赖。[官方传输教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)

## 与本文实验对应

实验设置 pin_memory=False，无 CUDA 数据拷贝。两种 worker 配置的人工等待保持相同，观察主进程 next 等待、样本完整性和启动影响。官方教程的设备侧结论用于解释和排障协议，CPU 记录支撑输入流水线行为。
