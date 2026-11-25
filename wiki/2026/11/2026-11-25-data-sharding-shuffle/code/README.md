# 练习与 CPU 复现

本目录只处理合成 ID、静态 shard 归属和可重建的本地批次计划，没有生产数据、训练模型、GPU 或远程存储依赖。现有 ai-infra 环境提供 Python 3.12.16、PyTorch 2.15.0 和 Ruff 0.17.3，无需安装或升级。

| 文件 | 用途 |
| --- | --- |
| [exercises.py](exercises.py) | 两题起始 API；NotImplementedError 是有意保留的学习空白 |
| [answers.py](answers.py) | 完整模型、严格输入检查、无状态合成值和进度 token |
| [test_answers.py](test_answers.py) | 20 项测试，含官方采样器对照、边界与计划身份拒绝 |
| [runtime_demo.py](runtime_demo.py) | 真正启动 CPU worker，观察预取和每 worker 尾批，重建输入后缀 |
| [run_answers.py](run_answers.py) | 两题结果汇总为 JSON；仅向指定输出目录写 results.json |

从本篇目录运行，命令中的 `python` 指现有 ai-infra 解释器：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v
ruff check code
ruff format --check code
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir /tmp/data-sharding-demo
```

学习时先实现 `rank_indices`，再实现 `DataPlan`／`ResumeToken`；辅助 `batch_indices`、`assigned_shards` 与 `sample_value` 的完整参考实现单独放在答案中。测试模块默认导入 `answers`，验收自己实现时可在自有副本中替换对应导入，保留未选为题目的辅助函数。不会把未完成 starter 的预期失败算成答案失败。

验收应得到三 rank 补齐结果 `[[0,3,6,9],[1,4,7,0],[2,5,8,1]]`。真实 worker 探针得到 `dispatched_at_start=4`、`dispatched_at_checkpoint=6`、`committed_batches=2`；正确恢复 8 批，错误恢复只剩 4 批并跳过 8 个样本。IterableDataset 的两个 worker 各自产出 3 个数，batch 大小 4 且丢尾时得到 0 批。

探针以 `spawn` 启动 worker，类和函数在模块顶层定义，入口受 `if __name__ == "__main__"` 保护。使用 Python 列表传输合成值，避免把 Tensor 共享内存传输混入本题；生产 Tensor IPC 性能未验证。创建后的原迭代器被完整消费，以得到对照后缀并正常退出 worker；没有模拟崩溃。token 只代表数据计划，必须与真实训练的完整状态一起保存才构成训练 checkpoint。

实际归档运行使用输出目录 `evidence/runs`，上方复现命令改用临时目录。原始日志、结构化结果与源码摘要见 [验证清单](../evidence/runs/verification.json)。初始 stub 的失败见 [摘要](../evidence/runs/red-tests.txt) 和 [无损压缩完整日志](../evidence/runs/red-tests.txt.gz)；新增 starter 骨架不改变已验证答案行为，清单单独记录该变更。
