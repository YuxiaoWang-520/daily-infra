# 官方文档与项目讨论笔记

所有请求日期为 2026-11-11，Asia/Shanghai。页面缓存获取时间与请求时间分别见 [research-log.json](research-log.json)。以下是自己的概括，未复制整页。

## Collective Operations

[官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) 返回页标为 2.32.3，provider acquired_at 为 2026-11-08。AllReduce 为每 rank 提供归约后的完整结果；Reduce 只有 root 接收；Reduce 后 Broadcast、ReduceScatter 后 AllGather 可以实现同一 AllReduce 语义。本文选择 SUM，平均在示例中另行除以 P，避免把所有框架的梯度分母混在一起。

## GPU／NIC 路径与拓扑

[GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html) 强调 GPU 间 peer access、GPU 到 NIC 的直接访问能力、拓扑暴露等因素。NCCL 使用系统信息发现 PCI 拓扑，虚拟化或容器暴露的路径可能影响判断。本文仅引用诊断思路，没有执行页面中任何驱动、内核或 BIOS 配置命令；生产修正需依据目标环境的实证与运维约束。

## Logging

[官方 Logging](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/logging.html) 返回页标为 2.32.3，provider acquired_at 为 2026-10-09。支持 NCCL_DEBUG 选择等级、NCCL_DEBUG_SUBSYS 筛选 INIT、GRAPH、NET、TUNING 等子系统，NCCL_DEBUG_FILE 可按主机和 PID 分日志。具体可见信息依赖版本、等级与构建，不能从“开了 INFO”就推断每种调优细节都有输出。

正文建议受控记录拓扑与实际选择，再以受控计时做比较；详细诊断日志会干扰性能，不能把日志运行当正式吞吐数据。这里没有修改当前机器或任务的配置。

## busbw 的解释边界

[官方仓库 issue 274](https://github.com/NVIDIA/nccl-tests/issues/274) 中，sjeaugey（页面标为 Member）的说明把换算定义为平坦点对点网络上的理论归一化；benchmark 本身不探测硬件内部拓扑。本文据此把 algbw、归一化 busbw 和 NIC 实际计数区分。具体因子由固定 benchmark 源码再次核对。

## 文档与当前 release

当日 GitHub latest release 为 v2.32.6-1，源码 commit 与 master 一致；返回的滚动文档标注 2.32.3，不能把 provider 缓存标题当作当前 release 版本。正文中有关今天实现的函数与路径以已 clone 的固定源码为准。
