# 练习、答案与真实双进程验证

两道练习建立 CP 的数值与数据所有权模型。使用已有 CPU PyTorch，不依赖 Transformer Engine、CUDA 或新增包。

| 文件 | 分工 |
| --- | --- |
| [exercises.py](exercises.py) | 两题起始接口；按正文约束实现。 |
| [answers.py](answers.py) | online softmax、布局辅助函数与单进程 CP 完整答案。 |
| [test_answers.py](test_answers.py) | `26` 项测试，独立 SDPA、有限差分、mask、空块、布局与梯度。 |
| [example_data.py](example_data.py) | 自编标量例子与固定 seed 的公开合成张量。 |
| [run_answers.py](run_answers.py) | 可复算输出、有效配对数及明确标为假设的前向字节预算。 |
| [distributed_cpu.py](distributed_cpu.py) | 两个本地 CPU／Gloo rank，AllGather 可微基线与 P2P 环形前向。 |

从本篇目录运行（`python` 绑定到已有环境）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py
python -m ruff check code
python -m ruff format --check code
```

`streaming_attention` 允许 K／V 块按任意顺序抵达，调用者保证覆盖目标 key 一次。全局位置决定 mask；全遮蔽输出为零、LSE 为 `-∞`。接口允许空块，拒绝不匹配形状、类型及已经溢出的分数。教学实现支持 float32／float64，实测主要为 CPU float64，不宣称具有 fused kernel 的内存效率。

四 token 例子输出 `[10,50/3,140/6,30]`。`S=8、p=2` 的连续／zigzag 因果配对数为 `[10,26]`／`[18,18]`。恢复全局顺序后两种布局输出一致，`S=16、p=4` 的 zigzag 梯度也与独立 SDPA 对齐。原始 `24` 项接口测试先失败；增加的两个边界测试真实暴露空 key 反向和分数溢出问题，修正后的 `26` 项在普通与优化模式均通过。

接入起始文件时，在测试中仅把 `streaming_attention` 和 `cp_attention` 的导入改为 `exercises`，其余辅助接口可保留参考版本。若练习布局辅助函数，可另外把相应接口一并切换为自行实现。

## 真实双进程

本次 macOS 的回环接口为 `lo0`；其他系统按本机配置调整：

```bash
PYTHONDONTWRITEBYTECODE=1 GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1 \
python code/distributed_cpu.py
```

脚本用临时 FileStore 完成 rendezvous，实际启动两个 CPU／Gloo 进程。先通过可微 AllGather 收集 K／V，在每个 rank 计算本地 query 并反向，再比较所有本地 Q／K／V 梯度；随后用 `batch_isend_irecv` 真正旋转 K／V，验证环形前向。它固定 `S=8、D=Dv=2、p=2`、连续布局、无 dropout、float64。每份 loss 为本地 `SSE/16`，相加等于完整输出平方均值。

四条 `CP_RESULT` 均通过，最大绝对误差 `1.1102230246251565e-16`，容差 `rtol=atol=1e-12`。[原始输出](../evidence/runs/09-distributed.txt)、[结果](../evidence/runs/09-distributed.json)、[环境与最终代码哈希](../evidence/runs/verification.json)。两个进程分别打印结果时可能相邻，因此归档解析按 `CP_RESULT ` 标记加 JSON 解码，不假设一行一个完整 JSON。

可微 AllGather 的旧接口已弃用，实际警告保留；这次固定环境仍可运行。首次 `torchrun --standalone` 启动在 `100` 秒超时且未留下完整原始输出，相关缺失和恢复方式保存在[失败记录](../evidence/runs/02-distributed-probe.json)。成功的 FileStore 运行有完整输出，未把最初超时归因到网络、Gloo 或硬件。

每个进程为对照另构造完整合成输入，没有测量实际训练内存。未执行 GPU／NCCL／TE kernel、分布式环形反向或计算通信重叠。[正文](../README.md)、[资料](../references.md)、[审查](../review.md)。
