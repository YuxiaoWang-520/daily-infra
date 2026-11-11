# 参考资料与证据

主题：AllReduce 的消息、链路与拓扑成本。实际检索日期：2026-11-11（Asia/Shanghai）。请求、provider 缓存获取日期、Git 获取与论文读取记录见 [research-log.json](evidence/research-log.json)。

## 关键论断与一手证据

| 编号 | 核心论断 | 一手资料与定位 | 本地归档 | 边界 |
| --- | --- | --- | --- | --- |
| S1 | SUM AllReduce 让每 rank 得到逐元素和；ReduceScatter＋AllGather 实现等价语义 | [官方 Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)，AllReduce／AllGather | [文档笔记](evidence/docs-notes.md) | 不自动等同任意训练损失的平均分母 |
| P1 | 环的分块、P−1 轮、无争用假设与带宽下界 | [作者原稿](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)，§2.2、§3、§4.1、图 1、定理 1 | [论文笔记](evidence/paper-notes.md) | 独立无压缩数据、结合／交换归约、大消息、无争用映射；浮点存在舍入 |
| S2 | API 校验入队、准备阶段调优 | [collectives.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/collectives.cc#L192-L211)、ncclEnqueueCheck、ncclTaskPrepare | [源码笔记](evidence/source-notes.md)、[manifest](evidence/upstream/manifest.json) | 固定版本调用路径，不是 Mac 上的 NCCL 运行 trace |
| S3 | runRing 分块并融合最后归约与首个结果转发 | [all_reduce.h](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/device/all_reduce.h#L14-L81) | [片段](evidence/upstream/nccl__src__device__all_reduce.h.txt) | 此处分析常规 Ring kernel；其他 kernel／protocol 受条件控制 |
| S4 | 候选选择与 Ring 成本读取硬件、节点、通道和协议信息 | [tuning.cc](https://github.com/NVIDIA/nccl/blob/ad2d192c1f279aefcaf340f67e07880b65a1116b/src/tuning/tuning.cc#L125-L239)、ring.cc、tuning_general.cc | 源码笔记、对应片段 | 教学 alpha／容量模型不能替代该实现或实际调优 |
| S5 | nccl-tests 按 2(P−1)/P 归一化 algbw | [AllReduceGetBw](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/src/all_reduce.cu#L59-L67)、[PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/afd59abab774a6a4b492eb344bd0c845a0b7bb4d/doc/PERFORMANCE.md) | [函数](evidence/upstream/nccl-tests__src__all_reduce.cu.txt)、[文档片段](evidence/upstream/nccl-tests__doc__PERFORMANCE.md.txt) | 归一化指标，不能直接充当物理 NIC 计数 |
| S6 | busbw 的平坦点对点解释有层次拓扑与算法边界 | [官方仓库 issue 274](https://github.com/NVIDIA/nccl-tests/issues/274)，sjeaugey 的说明 | 文档笔记 | 原始项目讨论；不把 benchmark 的归一化因子当作硬件端口测量 |
| S7 | GPU／NIC 路径、拓扑识别与诊断日志影响判断 | [GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)、[Logging](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/logging.html) | 文档笔记 | 返回的文档标注 2.32.3；未执行 GPU／系统配置命令 |
| E1 | 整数模拟器满足结果、所有权与 TX 记账；同机 Gloo 给出同样的 SUM | 本篇原创合成实验 | [lab.json](evidence/runs/lab.json)、[tests.txt](evidence/runs/tests.txt)、[验证说明](evidence/validation.md) | 28 项测试、4 个 CPU 子进程；没有 GPU 性能数据 |

## 源码与版本记录

官方 NVIDIA/nccl 默认分支为 master。今日从原先不存在的缓存路径 clone，固定 `ad2d192c1f279aefcaf340f67e07880b65a1116b`，之后 fetch 稳定 tag v2.32.6-1，解引用到同一 commit。稳定 release 发布于 2026-11-09T19:28:46Z。默认分支与稳定版本次完全相同，不把这个结果扩展到未来版本。

NVIDIA/nccl-tests 的当日 master API 返回 `afd59abab774a6a4b492eb344bd0c845a0b7bb4d`，随后读取三个固定原始文件。选段与原始 blob SHA256 见 [manifest.json](evidence/upstream/manifest.json)，许可证见 [NCCL](evidence/upstream/LICENSE-nccl.txt) 与 [nccl-tests](evidence/upstream/LICENSE-nccl-tests.txt)。源码选段保留原始行号和版权头；NCCL 为 Apache-2.0，benchmark 文件按归档许可证再分发。

所有页面请求均在今天发出；Parallel 返回的缓存获取日期分别保留。滚动文档显示 2.32.3，不能把它当作 v2.32.6-1 源码的逐字规范。实际 CPU 运行包为 torch 2.15.0、build Git d0f2af8b66b20da4baa50644c0cd3f887a2eae2e，未运行 NCCL。完整版本见 [versions.json](evidence/versions.json)。

## 论文与机制对应

Pitch Patarasuk、Xin Yuan，Bandwidth optimal all-reduce algorithms for clusters of workstations，Journal of Parallel and Distributed Computing 69(2)，117–124，2009 年 2 月。[DOI](https://doi.org/10.1016/j.jpdc.2008.09.002) 元数据经 Crossref 核对。实际读取作者公开稿，而非假设期刊页码等于 PDF 页码。

论文原文缓存于 .cache/upstream/papers/09jpdc.pdf，SHA256 为 ff950c8ab65a79ca840d41f4b8ddcb4cfd635465654c2de0ac0a7da88af5f294。通过文本提取读 §2.2、§3、§4.1，并渲染阅读 PDF 第 12—13 页的图 1 与定理 1。公开下载不自动等于可再发布，故 Git 中保存自己的笔记、定位和哈希，不再分发论文 PDF 或截图。

## 复现与图源

[代码入口](code/README.md) 给出完整题目、答案、命令与边界。[验证说明](evidence/validation.md) 对照真实日志和模型估算；[代码哈希](evidence/runs/summary.json) 固定已测版本。两个 Mermaid 图为原创教学机制图，分别保存在 [ring-phases.mmd](assets/ring-phases.mmd) 和 [shared-nic.mmd](assets/shared-nic.mmd)，没有把它们描述为真实网络 trace。
