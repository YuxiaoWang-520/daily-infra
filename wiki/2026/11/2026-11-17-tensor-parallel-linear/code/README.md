# 练习、答案与真实 CPU 进程验证

从本篇目录使用已有 PyTorch 环境运行。

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py
```

| 文件 | 用途 |
| --- | --- |
| [exercises.py](exercises.py) | 两道练习的起始接口，保留未实现入口供读者作答 |
| [answers.py](answers.py) | 单进程分片 MLP 与理想 ring 通信预算的完整答案 |
| [example_data.py](example_data.py) | 合成输入、数学权重、bias 与目标值 |
| [test_answers.py](test_answers.py) | `20` 项验收，包含全部梯度、`26` 个有限差分导数及预算边界 |
| [run_answers.py](run_answers.py) | 输出教学数值、软件版本及三个有明确范围的预算 |
| [distributed_cpu.py](distributed_cpu.py) | 实际 PyTorch TP 的两个 CPU／Gloo rank，比较默认配对和中间收集方案 |

第一题数学权重使用 `[in,out]`。完整输出为 `[[11,-5],[10,6]]`，MSE 为 `51.25`，输入梯度为 `[[19,1],[1.5,15.5]]`。答案中各分片的操作仍在同一进程，通过 Tensor 运算合成数值并保留 autograd。

第二题为均分 chunk 的理想 ring 模型。一个 MLP 前后向的预算分别取 `2 AR`、仅布局转换的 `2 AG + 2 RS`，以及加入 wgrad 输入重新收集后的 `3 AG + 2 RS`。`N=2048、H=8192、p=8、s=2 B` 时，每 rank 发送量依次为 `112、112、140 MiB`，接收量相同。模型不包含其他参数的梯度同步及 MLP 之外的通信。

两进程脚本的本次 macOS 运行命令如下。

```bash
PYTHONDONTWRITEBYTECODE=1 GLOO_SOCKET_IFNAME=lo0 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
    code/distributed_cpu.py
```

接口名 `lo0` 按系统网络配置调整。脚本检查前向、参数梯度和输入梯度，使用 `CommDebugMode` 分别记录前后向计数，初始化参数的通信在计数范围之外。默认配对每阶段各 `1 AR`，中间完整收集方案每阶段各 `1 AG + 1 AR`。这些是实际 CPU／Gloo 结果。

本次已有环境还运行了以下代码检查。

```bash
python -m ruff check code
python -m ruff format --check code
```

普通与 `-O` 模式均通过 `20` 项测试。命令、环境、退出码与代码 SHA256 保存在 [验证记录](../evidence/runs/verification.json)，结构化进程结果见 [两 rank 输出](../evidence/runs/09-distributed.json)。SP 的真实设备路径和 GPU 性能由正文明确其证据范围。[正文](../README.md)
