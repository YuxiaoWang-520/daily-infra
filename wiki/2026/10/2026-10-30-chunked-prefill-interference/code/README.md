# 运行与练习

本目录包含两道可在 CPU 上运行的练习。第一道验证分块 Attention 的因果关系，第二道用假设成本模型比较调度预算。两者都不需要安装 vLLM、Sarathi-Serve 或 CUDA。

本次使用已有 Python 3.10.13、PyTorch 2.9.1、macOS arm64 环境。普通执行需要 Python 3.10+ 和已有 PyTorch；Ruff 仅用于可选静态检查，不参与计算。实际命令、环境、退出码见 [运行记录](../evidence/runs/summary.json)。不要为运行本篇自动升级全局依赖。

## 文件与执行方式

| 文件 | 用途 |
| --- | --- |
| [exercises.py](exercises.py) | 两个待实现接口，默认抛出 `NotImplementedError` |
| [models.py](models.py) | 调度结果的不可变记录及整数检查 |
| [answers.py](answers.py) | 独立保存的完整参考答案 |
| [test_answers.py](test_answers.py) | 20 个测试：独立 SDPA、手算答案、边界与守恒 |
| [run_lab.py](run_lab.py) | 输出实际 CPU 数值对照，以及四组假设调度轨迹 |

所有命令均在**本篇文章目录**执行，即 `code/` 的上一层。

```bash
python -m unittest discover -s code -p test_answers.py -v
python code/run_lab.py
```

在起始文件中完成自己的版本后，可以这样运行同一套验收：

```bash
EXERCISE_IMPL=exercises python -m unittest discover -s code -p test_answers.py -v
```

未实现的接口会报错。也可以把自己的文件放在 `code/` 下，例如 `my_solution.py`，改成 `EXERCISE_IMPL=my_solution`，不必修改参考答案。运行 `run_lab.py` 时固定使用参考答案。

可选静态检查（仅在已有 Ruff 的环境）：

```bash
python -m ruff check --select E,F,I,UP --ignore E501 code
python -m ruff format --check code
```

## 练习一：跨块读取前缀 KV

实现 `chunked_attention(q, k, v, chunk_size)`，不需要阅读参考答案。

- Q、K 的 shape 为 `[L,D]`，V 为 `[L,Dv]`，所有维度为正。
- 只接受 CPU `float32`／`float64`，三者 dtype 相同，输入不得包含 `NaN` 或正负无穷；不符合契约时抛出 `ValueError`。
- `chunk_size` 为正整数，拒绝 `bool`、浮点数、0 和负数。
- 返回 `[L,Dv]`，不修改输入；支持非连续输入以及最后一块不足整块长度。
- 每块只生成自己的 Query 输出，KV 包含已处理前缀与当前块；按全局位置 `key_position <= query_position` 构造因果关系。
- 不设置 dropout。允许用 `torch.cat` 简化缓存内容，不考核原地分页写入或性能。

用以下独立基线验收：完整 Q／K／V 增加 batch 和 head 两个长度为 1 的维度后，调用 `torch.nn.functional.scaled_dot_product_attention(..., dropout_p=0.0, is_causal=True)`，取回 `[L,Dv]`。

float64 容差为 `rtol=atol=1e-12`；float32 为 `rtol=1e-5, atol=1e-6`。测试采用常规尺度输入；任意有限浮点数仍可能乘加溢出，不保证极端输入输出有限。

手算验收：Q／K 全为 0、V 为 `0..5`，块长为 3。位置 3 应平均读取 `0,1,2,3`，输出 `1.5`。错误地丢弃前缀会得到 `3.0`；保留前缀却对矩形输入直接用左上角三角关系，会得到 `0.0`。请能解释这两种错误。

完整推导在 [正文第 7 节](../README.md)。测试还检查默认设备改成 meta 时，显式 CPU 输入仍在 CPU 上正确计算；实现中的临时位置张量应跟随输入设备。

## 练习二：模拟完整工作集合的预算取舍

实现 `schedule_prefill(prompt_tokens, decode_requests, decode_steps, token_budget)`。

输入规则：`prompt_tokens=L`、`decode_steps=N`、`token_budget=T` 为正整数，`decode_requests=D` 为非负整数；全部拒绝 `bool`。本模型要求 `T>=D`，否则抛出 `ValueError`。

所有请求在时刻 0 就绪。旧请求的上一字在时刻 0，各还需要 N 个新输出；每个可用轮次，先给每条未完成的旧请求安排一个 Decode 位置，再把剩余预算给新 Prompt。假设 KV 与请求槽位充足，没有排队或抢占。

每轮耗时**人为规定**为 `1.0 + 0.5*d + p` 毫秒，d 为本轮 Decode 数，p 为本轮 Prefill 数。在轮末记录输出。Prompt 输入处理完整后记录其首字，但**不继续安排它的后续 Decode**；旧请求仍须生成完剩余输出后，模拟才能结束。

返回 [Schedule](models.py)：

- `rounds`：`Round` 的 tuple，字段顺序为起始时刻、结束时刻、Decode 数、本轮 Prefill 数、累计已处理 Prompt 数。
- `prompt_first_token_ms`：新 Prompt 第一个输出的时刻。
- `decode_output_times_ms`：每条旧请求的新输出时刻 tuple；没有旧请求时为空 tuple。

对 `L=16,D=2,N=4`，预期结果为：

| 预算 | 新 Prompt 首字 | 旧请求各自新输出时刻 | 最后事件时刻 |
| --- | --- | --- | --- |
| 18 | 18 | 18、20、22、24 | 24 |
| 6 | 24 | 6、12、18、24 | 24 |
| 4 | 26 | 4、8、12、16 | 26 |
| 2 | 32 | 2、4、6、8 | 32 |

表中单位均为模拟毫秒。额外检查每轮预算、总工作量、起止时刻连续，以及 `total_ms=len(rounds)+0.5*D*N+L`。这里既没有真实 GPU 计时，也没有模型权重复用或 context length 成本，因此不能用表格声称服务性能提高。

完整推导在 [正文第 7 节](../README.md)。本篇讨论调度取舍，不模拟新请求持续到达、优先级、分页回收或多 GPU。
