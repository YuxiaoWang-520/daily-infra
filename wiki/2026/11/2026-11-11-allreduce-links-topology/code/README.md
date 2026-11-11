# 两道练习与 CPU 验证

依赖为现有 Python 3.12.16、PyTorch 2.15.0（CPU Gloo 可用），没有新增依赖。纯模拟测试仅使用标准库；run_lab.py 的实际归约部分需要已有 torch。

## 练习一：从消息 trace 还原环归约

起始文件：[exercise_ring.py](exercise_ring.py)。输入是 P 行等长整数向量，每行元素数必须是 P 的正整数倍，element_bytes 为逻辑元素字节数。输出 RingResult 包含每个 rank 的完整 SUM、ReduceScatter 后 rank r 持有的第 r 块、两阶段消息列表、每个 rank 发送字节数。题设使用整数精确算术，不模拟固定宽度溢出、浮点舍入和真实序列化。

每轮开始时先冻结所有发送消息，再修改接收者。ReduceScatter 第 s 轮（s 从 0 开始），rank r 向 r+1 发送第 (r−s−1) mod P 块，接收后与本地对应块相加。P−1 轮后，各 rank r 拥有第 r 块的完整归约值。AllGather 第 s 轮发送第 (r−s) mod P 块，接收后复制。再经过 P−1 轮，所有 rank 都有完整结果。

验收：[test_answers.py](test_answers.py)。4 个 rank、8 个元素、逻辑元素 4 字节时，各 rank 发送 48 字节；全系统发送 192 字节。结果必须是 [60,64,68,72,76,80,84,88]。还覆盖单 rank、3 个 rank、负数、输入不变及非法输入。练习使用等块；实际 NCCL 支持尾块与对齐，不能把练习的整除约束当成 NCCL 限制。

<details>
<summary>完整答案与常见错误</summary>

[ring.py](ring.py) 先把输入复制成各 rank 的块，再以不可变 tuple 捕获本轮消息。若边收边取其他发送者的值，容易让同一轮错误地依赖本轮更新。只验证最终 SUM 还不够，测试也检查 ReduceScatter 的块所有权、两阶段每轮消息和 TX 字节数。

完成模板后复制到 ring.py 再执行测试；测试默认导入答案文件，单改模板不会改变被测实现。输出 trace 是教学同步轮次，实际 NCCL 把最后一次归约与转发融合，并按 channel／slice 流水线推进。

</details>

## 练习二：一条逻辑边为何不能独占 NIC？

起始文件：[exercise_costs.py](exercise_costs.py)。输入每 rank 的 payload 字节数 S、rank 的环顺序、逻辑有向边到物理资源的 route，以及每个有向资源的字节／秒容量和每轮启动时间 alpha。P 个 rank 均分 S，每轮每条逻辑边发送 S/P 字节。

先把同轮经过同一物理资源的流量相加，再用最慢资源的 load / bandwidth 加 alpha，最后乘 2(P−1) 轮。输出每轮资源负载、容量时间和模型估计总时间。要求 rank 顺序是 [0,P) 的排列、S 可等分、route 非空且无重复资源、使用到的容量有限且为正，alpha 有限且非负。P=1 的通信时间为零。

输入情景：两个节点各 2 个 rank，节点内有向边 200 GB/s，每节点 NIC 每个方向 50 GB/s；比较环 [0,1,2,3] 与 [0,2,1,3]。后者同方向有两个跨节点流共享 NIC，不能都赋予完整 50 GB/s。数字为自拟教学输入；GB/s 是十进制，256 MiB = 268435456 字节。

<details>
<summary>完整答案、模型边界与树基线</summary>

[costs.py](costs.py) 以整数累计物理资源的并发负载，先用有理数求比值，再把最终时间转成浮点数，避免中间整数过早转浮点而溢出。容量和 alpha 要能表示为有限 float；最终时间无法表示时抛 ValueError，单 rank 返回零。它是同步轮次、理想流体化／充分流水线的容量模型，忽略归约计算、逐跳启动、调度、头部与流水线填充成本；未模拟完整网络时序，不是 NCCL 调优器。

文件另提供整块 binomial reduce＋broadcast 的教学基线，限制 P 为 2 的幂、每层一对 peer 通信、链路无争用、没有分块流水线，时间为 2 log2(P) × (alpha + S/B)。该基线方便看到启动轮数和关键路径字节数的取舍，不能代表 NCCL Tree 或 double binary tree。

完成后复制到 costs.py 再执行测试。[run_lab.py](run_lab.py) 生成模型结果并调用 [gloo_probe.py](gloo_probe.py)，后者启动四个真实同机 CPU 进程验证 SUM，再手动除以 P 得到均值。它没有执行 NCCL，也没有测 GPU／NIC 带宽或延迟。

</details>

## 运行

从本篇根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/allreduce-lab-new.json
```

第二条要求输出文件不存在。原始输出与报告见 [运行记录](../evidence/runs/)，边界见 [验证说明](../evidence/validation.md)。
