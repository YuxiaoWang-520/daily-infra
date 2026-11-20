# 两道练习、完整答案与双进程验证

使用已有 CPU PyTorch。合成输入与专家参数不含任何内部项目数据；无需安装 Megatron、CUDA 或新增包。

| 文件 | 用途 |
| --- | --- |
| [exercises.py](exercises.py) | 路由与返回合并的起始接口。 |
| [answers.py](answers.py) | Top-k／容量、负载辅助项、owner 打包、两层专家、带权合并完整答案。 |
| [example_data.py](example_data.py) | 四 token 概率与固定 seed 的公开合成张量。 |
| [test_answers.py](test_answers.py) | 23 项测试，包含独立 dense 对照、有限差分、空行和门控边界。 |
| [run_answers.py](run_answers.py) | 手算例子、split 矩阵、aux 值与明确标为假设的远端负载。 |
| [distributed_cpu.py](distributed_cpu.py) | 真实两 rank 的不等长 AllToAll、返回及第一阶反向。 |

从本篇目录运行，`python` 指向已有环境：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py
python -m ruff check code
python -m ruff format --check code
```

默认选择集合内 Softmax，支持全专家 Softmax 后截取的对照。容量按一次调用的 routing batch 计算，按 gate 保留最高项、不重新归一化。相同分数选更小专家编号、相同容量分数保留更早 token，这些是教学 tie 规则。float32／float64 是接口范围，真实验证主要使用 float64。

四 token 标量输出 `[11/8,11/2,87/8,116/15]`，专家计数 `[2,2,3,1]`；容量 2 后为 `[2,2,2,1]`，第三个 token 的输出为 `7.5`。门控、输出、token／logits／专家梯度与独立全专家参考对齐，并通过有限差分。全部分配丢弃或某专家未使用时，保留零梯度路径。

接入起始文件时仅替换测试中对 `route_tokens` 和 `combine_tokens` 的导入，其他辅助接口可复用参考答案；若自行重写 `dispatch_plan`，也可切换对应导入。题目边界与完整详解见[正文](../README.md)。

## 真实双进程

本次 macOS 回环接口为 `lo0`，其他系统按本机配置调整：

```bash
PYTHONDONTWRITEBYTECODE=1 GLOO_SOCKET_IFNAME=lo0 OMP_NUM_THREADS=1 \
python code/distributed_cpu.py
```

临时 FileStore 完成 rendezvous，启动两个真实 CPU／Gloo 进程。每 rank 原有两个 token，拥有两个专家；专家为无 bias 的 `2→3→2` tanh MLP，Top-2，无容量丢弃、无辅助 loss。gate 留在源 rank，专家输出返回后加权。输出平方和统一除以全局元素数 8。

两种场景各产生两条 `EP_RESULT`：混合路由中每 rank 接收 4 条分配；空 owner 场景中 rank 0 接收 8 条，rank 1 接收 0 条。两者输出、dx、dlogits、dW1／dW2 及一步 `W-.05dW` 与独立 dense 对照一致，最大绝对误差 `4.336808689942018e-19`，容差 `rtol=atol=1e-12`。SGD 检查不包含优化器状态。

原始输出在[日志](../evidence/runs/07-distributed.txt)，按 `EP_RESULT ` 标记加 JSON 解码得到[四条结果](../evidence/runs/07-distributed.json)，不假设每条结果总独占一行。[环境、实际命令和六文件哈希](../evidence/runs/verification.json)。

程序为了独立 dense 对照保留完整合成输入，不提供真实内存容量测量。没有执行分布式 capacity／aux 训练、TP、GPU／NCCL、shared expert、FP8 或 overlap，自定义通信只验证第一阶反向。CPU 正确性不能推出 GPU 性能收益。[资料](../references.md)、[审查](../review.md)。
