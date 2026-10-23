# 异步执行资料与代码笔记

访问与审查：2026-10-23，Asia/Shanghai。正文把逻辑时间、mock 调用验证、实际 GPU 执行严格分开。

| 论断 | 一手来源 | 本篇验证 |
| --- | --- | --- |
| 提交与完成、并行取决于硬件和资源 | [CUDA Asynchronous Execution，2.5.1](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | 合成时间线验证先后，不模拟真实调度。 |
| 同流顺序、跨流依赖、异步计时 | [PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution) | 用本机 CPU 与 mock 检查参考答案协议。 |
| 分配器寿命与依赖不同 | [CUDA streams 示例](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams) | 解释 wait_stream 与 record_stream，不声称实测跨流内存回收。 |
| 默认 event.record 选择当前流 | [固定源码 208—218 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/cuda/streams.py#L208-L218) | timer 限定当前设备与同一当前流。 |
| wait_stream 影响未来工作，不阻塞 CPU | [固定源码 57—86 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/cuda/streams.py#L57-L86) | 队列模型区分设备依赖与主机等待。 |
| elapsed_time 是毫秒，event.synchronize 阻塞主机 | [固定源码 243—263 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/cuda/streams.py#L243-L263)、[Event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html) | mock 测试先等待再取值与毫秒转换；没有 GPU 时间戳验证。 |

源码实际刷新到官方 `pytorch/pytorch` 的 `main`，固定 `d52f3687e72fac81e20fd7741ab1fd10e0b3899d`。这是本次检索时看到的分支快照，不是本地 PyTorch 2.9.1 的构建来源，也不是把 2.14 文档当成本地版本。

[源码清单](evidence/source-manifest.json) 记录固定 API 下载命令、时间、SHA256 和字节数；[完整 streams.py](evidence/upstream/pytorch/pytorch/torch/cuda/streams.py) 与 [许可证](evidence/upstream/pytorch/pytorch/LICENSE) 一起归档。分析到 Python 包装层及其 CUDA API 职责，没有追称完整 C++／驱动内部执行路径。

自己的实现对应：每条流尾部决定下一任务最早开始时间，事件是前序尾部的值快照，设备等待更新消费者下界，主机同步推进 host 时钟。参考 timer 先预热并同步，再以同流 start/end 包围批次，等待 end 后读取毫秒。实际 event 区间仍可能含提交空档和干扰，不能冒称逐条 kernel 纯计算时间。

本主题是基础执行语义，没有独立的论文算法需要附加引用。未来讨论调度器或 CUDA Graph 优化时应另研究其实现与证据，不从这里推导服务吞吐。

运行：[summary.json](evidence/runs/summary.json)、[9 项测试](evidence/runs/tests.txt)、[逻辑时间](evidence/runs/lab.json)、[GPU 示例 SKIP](evidence/runs/gpu-demo.json)。归档：[SHA256SUMS](evidence/SHA256SUMS)、[archive-check.txt](evidence/runs/archive-check.txt)。
