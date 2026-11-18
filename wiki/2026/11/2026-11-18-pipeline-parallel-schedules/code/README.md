# 练习、答案和实际两进程验证

两道基础练习在已有 CPU PyTorch 环境执行。`exercises.py` 是待实现接口，`answers.py` 是完整参考答案。

| 文件 | 内容 |
| --- | --- |
| [answers.py](answers.py) | 固定局部次序的依赖调度，以及使用全局 MSE 分母的梯度累积 |
| [exercises.py](exercises.py) | 两道练习的起始接口 |
| [test_answers.py](test_answers.py) | `21` 项测试，覆盖时序依赖、状态峰值、不同 stage 耗时、梯度及一步更新 |
| [example_data.py](example_data.py) | 自编 float64 模型 `3→5→2` 和八行合成样本 |
| [run_answers.py](run_answers.py) | 实际数值输出，以及明确标为假设的耗时、容量和通信预算 |
| [distributed_cpu.py](distributed_cpu.py) | 两个实际 CPU／Gloo stage，运行 PyTorch GPipe 与 1F1B |

从本篇目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py
python -m ruff check code
python -m ruff format --check code
```

两种调度模型在 `p=3、m=4`、每个 `F／B` 耗时 `1` 的假设下都需 `12` 个时间单位，待反向份数峰值分别为 `[4,4,4]` 与 `[3,2,1]`。模型支持 `m<p`；本次 PyTorch `Schedule1F1B` 要求 `m≥p`，不能据此推定框架也支持该模型的全部输入。

CPU 梯度练习验证 `[1,3,4]` 等分块与完整 batch 对齐，全局 MSE 约 `0.37916318132772353`。模型必须具有相应的逐行分块语义。全局有效元素数是统一分母，函数清理旧梯度，更新由调用者在最后完成。

实际进程使用以下命令，本次 macOS 的回环接口为 `lo0`，其他系统按本机网络配置调整：

```bash
PYTHONDONTWRITEBYTECODE=1 GLOO_SOCKET_IFNAME=lo0 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
    code/distributed_cpu.py
```

该脚本固定两 stage、四个等长 microbatch。每份 loss 为 `SSE/16`，配合 `scale_grads=False`。显式静态元数据包含跨段输入及输出的 `requires_grad`；普通调用记录不包含额外元数据推导前向。GPipe 调用峰值为 `[4,4]`，1F1B 为 `[2,1]`，两者输出、全部参数梯度和一步 SGD 更新均与完整模型对齐。

计数是未配对前后向调用的峰值，不是分配器内存峰值，脚本还为数值对照保存了最后一段输出。没有在本次实验测量 GPU、NCCL、通信带宽、真实激活内存或分布式不等长分块。[实际结果](../evidence/runs/09-distributed.json)、[完整命令与环境](../evidence/runs/verification.json)、[正文](../README.md)
