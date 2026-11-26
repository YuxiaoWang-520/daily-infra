# 两道练习与 CPU 验证

本目录使用合成 token ID、成本算术与一个极小的 float64 CPU Attention 模型，验证布局、样本边界和损失权重。使用现有 ai-infra 的 Python 3.12.16、PyTorch 2.15.0、Ruff 0.17.3，不安装或升级依赖。

| 文件 | 职责 |
| --- | --- |
| [exercises.py](exercises.py) | 两题 starter；仅复用给定的数据类，机制函数留作学习空白 |
| [answers.py](answers.py) | 完整装箱、成本、metadata、CE sum／count 与 DDP 缩放答案 |
| [toy_model.py](toy_model.py) | 单 head、绝对位置、残差与词表投影；强制 CPU SDPA math，无 dropout |
| [test_answers.py](test_answers.py) | 25 项测试，含手算、错误对照、梯度及边界 |
| [run_answers.py](run_answers.py) | 汇总严格 JSON，仅向指定目录写 results.json |

从本篇目录执行，python／ruff 指已有 ai-infra 环境：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v
ruff check code
ruff format --check code
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir /tmp/sequence-packing-demo
```

测试默认导入完整答案。练习时先实现 starter 中的布局／成本函数，再实现 packed metadata 与损失处理；可在自己的副本中将相应测试导入换成已完成实现。`CostEstimate` 和 `PackedBatch` 数据类已提供，starter 的 NotImplementedError 是有意保留，不代表完整答案未完成。

第一题验收 `[8,2,7,3]` 的 30／22／20 个槽、226／146／200 平方面积，以及 FFD 非最优反例。模型没有把排序／装箱时间算进 GPU 吞吐，也不自动截断超长输入。

第二题按预测位置直接构造已 shift 的 labels：`[4,6,-100,5,-100]`。不要再移动一次。metadata 保留有效输入 token 的计算，labels 决定哪些位置参与损失；二者不是同一 mask。全部监督被忽略时只返回可微的 sum=0 和 count=0，最终归一化策略由调用方协调。

CPU 张量工厂与参数初始化显式固定设备；回归测试覆盖应用将默认设备设为 meta 的情境。CPU 数值测试采用 `1e-12` 绝对误差上界，结果不是跨后端 bitwise 保证。普通 DDP 归一化只用本地 autograd 模拟标准平均后的梯度，没有建立分布式进程组。GPU varlen 接口只做固定源码核验；运行脚本也没有调用该接口。

本次原始执行输出写到 `evidence/runs`，上方命令改用临时目录保留原档。测试、真实 stdout、结构化结果和源码摘要见 [verification.json](../evidence/runs/verification.json)。测试先行的两项初始 stub 失败见 [red-tests.txt](../evidence/runs/red-tests.txt)。
