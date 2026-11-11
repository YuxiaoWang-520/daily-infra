# 实际运行与验证范围

日期：2026-11-11（Asia/Shanghai）。机器 wyx-macmini.local，Apple M6，macOS 27 arm64。使用已有 Python 3.12.16（/Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python）、torch 2.15.0，build Git d0f2af8b66b20da4baa50644c0cd3f887a2eae2e；CPU Gloo 可用，CUDA 不可用。没有安装或升级依赖。

## 测试优先与实际修正

先编写测试和 NotImplementedError 接口，[red-tests.txt](runs/red-tests.txt) 记录 23 项测试运行，45 个 error（包括 subtest），接口未实现不能通过。完成答案后 [first-pass-tests.txt](runs/first-pass-tests.txt) 中 23 项通过。

独立 Python 审查发现数值边界：中间 float 转换提前溢出，单 rank 的树估算还可能得到 NaN。新增 5 项回归，修正前 [numeric-red-tests.txt](runs/numeric-red-tests.txt) 中 28 项运行得到 1 failure、4 errors；修复整数负载、有理数比值、最终 float 范围及单 rank 分支后，[tests.txt](runs/tests.txt) 中 28 项全部通过，退出码 0，unittest 用时 0.001 秒。此耗时不用于任何 GPU 性能结论。

独立审查者另外复验 5 项回归全部通过，范围见 [python-review.md](python-review.md)。Ruff [check](runs/ruff-check.txt) 与 [format --check](runs/ruff-format.txt) 均退出 0。

## 最终命令

从本篇根目录实际运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff check code
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff format --check code
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/daily-infra-nov11-archival-lab.json
```

四条最终命令退出码均为 0。JSON 原样复制归档至 [lab.json](runs/lab.json)，包含原始消息 trace；stdout 原样保存在 [lab-stdout.txt](runs/lab-stdout.txt)。运行发生在 drafts/ 下，原始失败日志保留当时路径，没有改写成仿佛在发布目录执行过。已测代码哈希见 [summary.json](runs/summary.json)。

## 真实四进程 CPU 实验

run_lab 调用 gloo_probe，以 spawn 启动 4 个真实子进程。每进程设置 1 个算子线程，以临时 FileStore rendezvous 初始化 Gloo，超时 20 秒。每 rank 输入 float64 向量 arange(8)+10×rank，使用 SUM AllReduce，全部结果为 [60,64,68,72,76,80,84,88]，与精确整数模拟器一致；随后手动除以 4 得到均值。

每个 rank 验证自己的结果，并写入本次临时目录；父进程读取后再逐 rank 对照模拟器。工作进程 finally 释放 process group，临时目录由上下文管理器回收。没有读取外部训练数据、发消息给他人或触发 GPU 任务。

本实验验证归约语义，未测延迟、GPU 带宽、真实 ring 排序、NCCL protocol 或跨节点流量。Gloo 的正确结果不能证明走了 NCCL runRing。

## 模型输入与结果

所有时间来自自拟 alpha=5 μs、独立有向带宽 50e9 字节／秒、节点内边 200e9 字节／秒，payload 为 4096 字节或 268435456 字节。GB/s 使用十进制，KiB／MiB 使用二进制。

| 情景 | 模型估计 |
| --- | --- |
| flat，P=4，4 KiB | 30.12288 μs |
| flat，P=8，4 KiB | 70.14336 μs |
| flat，P=4，256 MiB | 8.08306368 ms |
| flat，P=8，256 MiB | 9.46524096 ms |
| 两节点相邻环，256 MiB | 8.08306368 ms |
| 两节点交错环，256 MiB | 16.13612736 ms |

模型忽略归约计算、逐跳启动、包调度、pipeline fill 与实际协议。共享有向资源在同轮累计负载，充分流水线容量项为 max(load/B)；不把它说成任意网络的精确完成时序。整块 binomial tree 基线另有独立假设，不代表 NCCL Tree。

## 论文与源码读取

NCCL 实际 clone 与 tag fetch 均成功，固定 master／v2.32.6-1 的相同 commit。benchmark 三个固定文件通过原始 URL 读取。source manifest 保留原始 blob 与选段哈希；许可证随片段归档。

原始论文通过 Parallel 提取阅读，并下载作者公开 PDF，pdftoppm 渲染第 12—13 页后实际查看图 1 与定理 1。第 12 页渲染报告 Fontconfig 配置警告，但退出码为 0，页面可读；未将警告写成论文提取失败。原文哈希与自己的阅读笔记归档，不发布版权许可不明的 PDF／截图。

未执行：CUDA、NCCL／nccl-tests、RDMA、跨节点 collective、真实 GPU 拓扑、硬件吞吐、其他浮点归约顺序或生产训练扩容。正文数字和结论按以上范围分别标注。
