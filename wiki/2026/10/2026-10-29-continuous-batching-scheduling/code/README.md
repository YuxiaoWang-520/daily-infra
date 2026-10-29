# 练习：每轮调度与准入预算

这里有两道独立练习，使用 Python 3.10+ 标准库即可。题目从这里开始读，不必先打开答案。

- [exercises.py](exercises.py)：待实现的两个函数，默认抛出 `NotImplementedError`。
- [models.py](models.py)：题目提供的输入／返回类型及基础参数检查。
- [test_answers.py](test_answers.py)：手算轨迹、容量账本与边界测试。
- [answers.py](answers.py)：完整参考答案。
- [run_lab.py](run_lab.py)：打印正文示例的实际 CPU 运行结果。

在本篇目录运行参考答案：

```bash
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/run_lab.py
```

在 `exercises.py` 写好答案后，运行同一套测试：

```bash
EXERCISE_IMPL=exercises python3 -m unittest discover -s code -p 'test_*.py' -v
```

起始文件未实现时，此命令应失败；这是正常状态。测试只动态选择实现模块，输入与返回类型相同。需要自己另建练习文件时，可以复制这个目录中的起始文件及类型定义到自己的工作目录；本任务没有写入或改动仓库已有 `playground/`。

## 练习一：simulate

```python
def simulate(requests: Sequence[Request], capacity: int,
             policy: Literal["static", "continuous"]) -> Simulation:
    ...
```

`Request(request_id, arrival_tick, output_tokens)` 表示一条合成请求。`request_id` 非空且全局唯一；到达时刻是非负整数；输出数是正整数；容量是正整数。布尔值不作为整数计数。输入序列不能被重排或修改。

模型假设：每个 Prompt 长 1 个 token，第一轮处理这个 Prompt；之后每轮处理上一个输出。每条入选请求每轮都输出 1 个 token，耗时恒为 1 tick。token 预算至少为容量，KV 充足，无前缀命中、投机、取消或抢占。这些假设只为隔离调度粒度。

规则：

1. 请求按到达时刻排序；相同时保持输入顺序。输出长度只决定何时结束，不能影响顺序。
2. 时刻 t 到达的请求，可以参加从 t 开始的执行，输出记在 `t+1`。
3. `static` 仅在已接纳批次全部完成后补充新成员；已完成请求的输出仍及时交付。
4. `continuous` 每轮开始时补足空位。新成员不能在前一轮中途加入。
5. 无工作时可以跳到下一次到达时刻，但不能把绝对 tick 重置为 0。空输入返回两个空 tuple。
6. 无效容量、策略、重复 ID 等须报 `ValueError`；请求自身字段由类型定义检查。

返回：`Simulation.steps` 是 `Step(start_tick, request_ids)` 的 tuple；只记录实际执行的轮次，不记录空闲轮次。`Simulation.output_ticks` 是按**原输入顺序**返回的 `(request_id, 输出时刻 tuple)` 的 tuple，可以用 `dict(result.output_ticks)` 读取。

验收输入：

```python
requests = (
    Request("A", 0, 1), Request("B", 0, 4),
    Request("C", 0, 2), Request("D", 0, 1),
)
```

容量 2：固定模式依次为 `AB、B、B、B、CD、C`；连续模式为 `AB、BC、BC、BD`。相应 C 输出时刻分别为 `(5,6)` 与 `(2,3)`。再测 A 在 0 到达需 3 个输出、B 在 1 到达需 1 个输出：连续模式 B 于 2 输出，固定模式于 4 输出。

这些是逻辑 tick，不是毫秒或 GPU benchmark。代码里的 `filled_slot_fraction` 仅统计执行轮次的槽位填充比例，不计空闲期间；输出吞吐的时间窗口必须另外写清。

## 练习二：admit_prefills

```python
def admit_prefills(waiting: Sequence[Prefill], *, running_decodes: int,
                   max_num_seqs: int, token_budget: int,
                   free_blocks: int, block_size: int) -> Admission:
    ...
```

输入是一组按等待顺序排列的 `Prefill(request_id, prompt_tokens)`。Prompt 长度必须为正整数，ID 非空且等待队列内不重复。`running_decodes` 和 `free_blocks` 非负，其余预算及块长为正整数；全部计数拒绝布尔值和浮点数。等待序列不可修改。

`running_decodes` 是本轮已经安排好的运行中请求数，每条消耗 1 个 token；它们必须已经符合请求上限和 token 预算，否则报 `ValueError`。**`free_blocks` 是已经为这些 Decode 本轮分配好 KV 后剩下的块数**，不能再次为它们扣账。

本题只安排一轮，使用以下明确的简化：

1. 新请求无缓存命中／共享，完整处理 Prompt，不拆分。每个新请求需要 `ceil(prompt_tokens / block_size)` 块。
2. 检查有效请求数上限，再检查剩余 token，最后检查剩余块。若同时失败，按这个优先顺序报告。
3. 所有检查都通过，才接纳该请求并扣两项预算；不能留下失败的部分分配。
4. 第一条放不下就停止；不能跳过它、按长度排序，或回收运行中请求。未来 Decode 的增长不在本轮预留。
5. 空等待队列仍保留 running 的请求／token 计数，停止原因为 `None`。输入无效或已有运行工作本身不可行时报 `ValueError`。

返回的 `Admission` 各字段：

| 字段 | 含义 |
| --- | --- |
| `admitted` | 本轮新接纳的请求 ID，tuple |
| `num_scheduled_requests` | 已安排的 Decode 数加新接纳数 |
| `num_scheduled_tokens` | Decode 所用 token 加新接纳的完整 Prompt 长度 |
| `remaining_tokens` | 原 token 预算减上述处理量 |
| `remaining_blocks` | 输入空闲块减新接纳请求使用的块 |
| `stopped_request` | 第一条失败的请求 ID；全部成功则为 `None` |
| `stop_reason` | `sequence_limit`、`token_budget`、`kv_capacity`；全部成功则为 `None` |

验收输入：

```python
waiting = (Prefill("C", 4), Prefill("D", 3), Prefill("E", 1))
parameters = dict(running_decodes=2, max_num_seqs=4,
                  token_budget=8, free_blocks=3, block_size=4)
```

默认只接纳 C，D 被 token 预算挡住；剩余 token 为 2、剩余块为 2。只把 token 预算改为 16，接纳 C、D，共 4 条请求、9 个 token，E 被请求槽挡住。只等一条长度为 5 的 C、剩余块改成 1 时，KV 不够，不能扣掉 C 的 token。

本题没有实现真实 vLLM 的片段 Prefill、预留、异构缓存、优先级或公平性。它用于检验一轮预算的单位和事务性，不能用来保证未来不会 OOM。

## 详解、原始输出与运行范围

逐步推导见 [正文第 7 节](../README.md#7-两道代码练习把看起来能排变成可检查的规则)，完整答案见 [answers.py](answers.py)。[真实输出](../evidence/runs/lab.json) 与 [测试记录](../evidence/runs/tests.txt) 可直接核对上述轨迹。

本机使用已有 Python 3.10.13 环境；练习没有第三方依赖。环境清单中的 PyTorch 2.9.1 来自统一记录工具，练习本身没有调用 PyTorch。未执行 vLLM、CUDA 或生产负载，没有 GPU 吞吐与延迟结论。
