# 固定 NCCL 与 benchmark 源码阅读笔记

日期：2026-11-11（Asia/Shanghai）。NCCL master 与稳定 v2.32.6-1 同为 ad2d192c1f279aefcaf340f67e07880b65a1116b。nccl-tests master 固定 afd59abab774a6a4b492eb344bd0c845a0b7bb4d。原始路径、行号和哈希见 [manifest](upstream/manifest.json)。

## 入口、队列与准备

src/collectives.cc 的 ncclAllReduce 通过 ncclAllReduceConfigImpl 建立 ncclInfo；count、datatype、op、stream 等信息进入 ncclEnqueueCheck。src/enqueue/enqueue.cc 中该函数检查 communicator 与参数，再 taskAppend，结束内部 group。该动作不代表 GPU 上通信已经完成。

src/group.cc 的 ncclTaskPrepareJobFunc 调用 ncclTaskPrepare。src/enqueue/task_prep/task_prep.cc 把原始任务预处理为 tuning input，再为可调优 collective 调用 ncclTuningCompute，之后分类与 post-tuning。这是今日版本的实际目录，未沿用历史 src/enqueue.cc 或 src/graph/tuning.cc 的旧路径。

## 调优选择与 Ring 模型

src/tuning/tuning.cc 枚举 tuningMask 允许的有效候选。ncclTuningSelectBestTuning 按 selectionTimeUs（若有）或 timeUs 比较；配置的 plugin 可以修改通用成本表，后续还存在通道选择、注册与 kernel 类别等条件。本文只说明实际候选与成本选择机制，没有穷尽全部平台路径或声称总选 Ring。

src/tuning/ring.cc 从通信图读取 bwIntra／bwInter、nChannels；根据 protocol、节点数和计算能力加入修正。src/tuning/tuning_general.cc 的 ncclTuningGetNsteps 对 AllReduce 返回 2(nRanks−1)，而 ncclTuningGetTime 使用延迟、消息字节与带宽等计算估计。正文 alpha＋资源容量模型只保留基础因果关系，不复刻这些常数、修正或候选有效性规则。

## 常规 Ring device 路径

src/device/all_reduce.h::runRing 使用当前 channel 的 ring.index、prev 和 next；channel 数据被拆成 chunk，并处理末尾数据对齐。directSend 推出首块，directRecvReduceDirectSend 传播部分归约；directRecvReduceCopyDirectSend 得到自己块的最终结果、保存并转发；之后 directRecvCopyDirectSend 和 directRecv 收齐块。

教学练习把 ReduceScatter 与 AllGather 分开，以展示块所有权。实际代码在交界合并动作，按通道和传输协议推进；同步轮次 trace 不是 GPU 全局 barrier 或逐 kernel trace。存在 SIMPLE／LL／LL128 与其他 kernel 类别，本文不声称在任意平台都经过同一实例化。

## nccl-tests 的归一化

src/all_reduce.cu::AllReduceGetBw 用 count×typesize / 1e9 / seconds 得到 GB/s 的 algbw，再乘 2(nranks−1)/nranks 得 busbw。固定 doc/PERFORMANCE.md 给出相同因子和口径。

该换算没有采集每个物理 NIC 或共享 PCIe 路径的计数。正文因此同时引用官方仓库 issue 274 的原始项目讨论，说明平坦点对点归一化与层次硬件之间的解释边界。没有把归一化数值与某个端口带宽直接画等号。

## 实际执行范围

只实际运行 Python 整数模拟、容量模型与四 rank CPU／Gloo SUM。没有在 Mac 上编译或运行 NCCL／nccl-tests、CUDA、RDMA 或跨节点传输。源码分析有固定版本，运行结论限于自己执行的代码。
