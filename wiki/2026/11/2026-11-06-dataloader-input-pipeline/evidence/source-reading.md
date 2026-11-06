# 固定源码阅读笔记

本次读取 main 提交 `b79f21bab5e9e221b05bc903f7d8171a8c04a8a0`，源码文件为本篇镜像。以下行号对应该提交，升级后需要重新定位。

## 从入口追踪执行者

`DataLoader.__iter__` 在第 497 行附近选择复用或新建迭代器。单进程 `_SingleProcessDataLoaderIter` 在第 756 行附近，其 `_next_data` 直接执行 fetch，随后按条件 pin。多进程迭代器在第 790 行开始，注释中的队列示意与后续构造代码核对过。主进程组织任务与接收数据，后台 worker 执行 dataset 访问。

`worker.py::_worker_loop` 第 244 行开始设置进程状态，第 274 行调用 `torch.set_num_threads(1)`，第 309 行附近创建 fetcher，第 375 行附近执行 `fetcher.fetch(index)`。这项 PyTorch 线程设置没有替用户外部解码库指定线程策略。

`fetch.py::_MapDatasetFetcher.fetch` 在自动合批条件下尝试 dataset 的 `__getitems__`，否则逐索引访问，随后统一调用 `collate_fn(data)`。本文 CPU dataset 逐项读取 JSON，default collate 将字典中同键的数值整理成 Tensor。逐项源／解码／变换计时之和省略了 collate 和跨进程搬运，主进程 next 则包含交付等待。

## 任务、就绪结果与顺序

`dataloader.py` 第 286—297 行校验 worker 和 prefetch 组合。0-worker 模式要求省略多进程预取配置，正 worker 默认 factor 为 2，persistent workers 要求正 worker。

`_reset` 第 1234 行开始重置接收、发送和任务状态，第 1276 行附近预填充 `num_workers×prefetch_factor` 次任务。`_try_put_index` 第 1532 行开始核验容量并发出索引，`_process_data` 接收处理后再补充任务。

`_tasks_outstanding` 追踪已发出且尚未收到的任务。有序模式也能把已返回的后续任务放在 `_task_info`，等待 `_rcvd_idx` 对应任务。第 1469—1530 行的 `_next_data` 先处理这个交付顺序。正文把配置槽位、已发未收和可立即消费的结果分别解释。

因此任务乘积只适合规划并发与粗略载荷。队列覆盖暂停的 Q 需要是已就绪且按当前策略能消费的数据量。默认有序时，后续已完成任务也可能暂时无法交付。

## pinning 与资源生命周期

第 656—685 行把 pinning 开关与可用加速器条件结合，所选 main 明确关闭 MPS pinning。第 1192—1211 行创建主进程中的 pin-memory 线程，未启用时直接使用 worker result queue。

`pin_memory.py::_pin_memory_loop` 消费结果并调用递归 `pin_memory`。Tensor 调用相应方法，自定义对象和容器走各自分支。`Memory.cpp` 的 `pin_memory` 对已经 pinned 的对象可直接返回，否则 `_pin_memory` 创建 pinned storage 并复制。主机内存峰值要包含新旧数据的实际生命周期。

DataLoader 的 `_shutdown_workers` 第 1604 行附近依次处理 pinning 线程、退出事件、worker 和队列，并在必要时终止未退出进程。该函数以状态保证重复调用的收尾行为。本文固定版本实验在 finally 中调用它，避免异常 traceback 持有迭代器时延后退出。此调用属于实验对私有内部接口的明确依赖。

## 稳定版本与运行版本

main 和 v2.15.0 的 DataLoader 差异文件已归档。所比较的 fetch、worker、pin_memory 文件及已安装运行文件以 [source-manifest.json](source-manifest.json) 中的实际字节比较为准。整个运行构建与上游 tag 的 Git SHA 分开列出，逐文件一致性结论仅覆盖清单中的文件。

## 推导边界

worker 一个任务的读取和 CPU 处理连续执行，所以教学模型使用 `(R+C)/w`。共享存储服务另给出 R 约束。真实存储排队可能进一步占用 worker，模型省略这些附加延迟，作为理想下界。

CUDA 拷贝与模型执行在同一 stream 时依次执行，正文用 H+G。独立资源模型的 max(H,G) 要求跨 batch 排程及真实并发条件。CPU 练习只把这一项当作显式输入的模型假设。
