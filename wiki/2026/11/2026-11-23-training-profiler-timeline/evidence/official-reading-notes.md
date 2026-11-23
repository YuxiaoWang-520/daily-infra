# 官方资料阅读笔记

访问日：2026-11-23，Asia/Shanghai。检索用于定位一手资料，核心实现另由当日实际 fetch 的固定源码支持。没有复制整份网页或厂商手册。

## PyTorch API 与采集边界

读了 [2.15 profiler API](https://docs.pytorch.org/docs/2.15/profiler.html) 的 activities、schedule、step、export、shape／stack 开销和 CUDA fallback 说明，以及 [官方 recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 的长期训练抽样部分。

自己的理解：采样器按调用者给出的 step 编号选择动作。训练循环须发出 `prof.step()`；它不负责执行优化器更新。`wait`、`warmup` 和 active 解决的是采集窗口问题，不能替代训练自身充分 warmup 或无 profiler 性能基线。多轮 trace 应在回调内保存。

CPU、设备可用与设备活动被成功采集需要分别验证。API 对 CUDA 的说明包含 CUPTI 不可用时的 legacy 回退：表中可能仍有设备计时，导出 JSON 却没有设备活动。这为“先核查采集能力，再解释空轨道”提供依据。本次本机只支持 CPU activity，没有实测该 CUDA 回退。

shape 和 stack 会增加成本；shape 记录会保留引用，可能影响依赖引用计数的优化。本文明确关闭这些额外选项，且仍不把带采集／导出的墙钟当成生产吞吐。

版本注意：检索结果中的 `/stable/` 页面出现过旧版标题，未用该浮动别名证明当前版本；记录明确的 `/2.15/` 页面，并以 fetched Git SHA 为实现证据。Parallel 对 API 返回的获取日期为 2026-11-17，对 Nsight 2026.4 归档页为 2026-10-23；本次读取在 2026-11-23，不能把缓存获取日说成当天实时抓取。详见 [检索元数据](web-retrieval.json)。

## Nsight Systems：关联与通信

读了 [2026.4 User Guide 的 Timeline Correlation](https://archive.docs.nvidia.com/nsight-systems/2026.4/UserGuide/index.html#timeline-correlation)、[当前 User Guide 的 NCCL Execution Concepts](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[2026.3 Analysis Guide 的 CUDA API／kernel 报表和关联查询](https://archive.docs.nvidia.com/nsight-systems/2026.3/AnalysisGuide/index.html)。检索摘录直接覆盖 API→runtime 调度→kernel launch→GPU operation 的分层，以及 API 与设备活动的关联。

自己的理解：CPU API 范围与设备活动是两种记录，关联比时间邻近更可靠。某些报表使用 ns，Chrome JSON 使用 μs，读数前先核对单位。汇总报表百分比必须核查分母，不能直接解释成端到端墙钟占比。

分组、非阻塞 communicator 和 Graph 会改变 NCCL 的提交方式及线程。高级 NCCL tracing 还有采集模式、Nsight 与 NCCL 版本的条件；本文没有把某一版本高级 tracing 的字段当成所有 PyTorch trace 的通用字段，也没有在本机验证此模式。

“早到 rank 的长通信可能包含等对端推进”“通信与计算会竞争资源”是依据异步依赖与分布式执行提出的机制解释，不是本篇 trace 实测的根因。必须补充同一 collective 的身份、输入就绪、各 rank 前置工作、消息和拓扑，再做控制实验。

## Perfetto：文件单位与支持范围

读了 [外部格式说明](https://perfetto.dev/docs/getting-started/other-formats) 的 Chrome JSON 部分和 [FAQ](https://perfetto.dev/docs/faq) 的兼容边界。`ts` 的微秒单位、`pid/tid` 轨道和 `X` 完整事件可用于本篇教学投影。常见 JSON 可直接载入工具，但 importer 不是对全部历史格式特性的无条件保证。

因此解析器显式只接受有 `ts/dur` 的完整事件，不装作支持 `B/E` 配对、flow 或通用关键路径恢复。本次没有实际打开查看器，不声称完成视觉验收。[独立检索元数据](trace-format-retrieval.json)

## 为什么没有算法论文？

本篇介绍活动采集 API、事件统计语义与基础区间集合运算，没有讲解某篇论文提出的训练或通信算法。区间合并与固定依赖 DAG 的最早完成递推属于基础模型，官方语义和可执行 oracle 足以核查；不引入与核心问题无关的论文。NCCL 协议、GPU 微架构和关键路径推断算法不在实现或验证范围内。
