# 两道练习与复现方法

需要已有 PyTorch 2.15 环境；本次用 Python 3.12.16、PyTorch 2.15.0、Darwin arm64。没有 CUDA 依赖，没有安装或升级包。从本篇目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir /tmp/public-async-checkpoint-demo
```

第二条命令只在指定输出目录写 `results.json`，同名文件会被覆盖，请选自己的目录。真实 checkpoint 仅在函数自建的 `TemporaryDirectory` 中保存、读回并清理，不从外部路径加载 checkpoint。默认序列化涉及 pickle，因此不要把本示例改成读取不可信文件的接口。

## 练习一：有界保存流水线

起始点：[exercises.py](exercises.py) 的 `simulate_checkpoints`；完整答案：[answers.py](answers.py)。输入为 `num_steps,compute_ms,interval_steps,stage_ms,write_ms`，全部是整数；拒绝 bool、float 和其他类型。步数、stage、write 成本可为 0；compute 和间隔须大于 0。Python 整数用于精确时间运算。

模型只有一份未完成保存：计算与 staging 共用训练轨道，保存用后台轨道；下一次 staging 前必须等待上一份保存完成。输出每个 checkpoint 的 step、等待、staging 和 write 区间，及 `training_finished_ms`、`all_saved_ms`、`backpressure_ms`、`final_drain_ms`、`sync_total_ms`。

验收：输入 `(300,100,100,2000,6000)`，应为训练结束 36000、全部保存 42000、同步基线 54000、背压 0、最后等待 6000 ms。将 write 改为 16000，分别为 48000、64000、84000、12000、16000 ms。没有整除间隔的最后一步不自动补一份保存；`250` 步的快存储例子训练到 29000 ms，保存到 30000 ms。

按 checkpoint 时刻推进计算，取上一份保存结束与当前时间的最大值，再安排 staging。最后推进训练尾段并排空后台任务。不要把它改成无界排队或异步 staging 后仍沿用同一组预期值。模型固定时长，没有资源竞争，因此不构成实际优化收益预测。

## 练习二：稳定快照与两个完成边界

起始点：[exercises.py](exercises.py) 的 `run_checkpoint_case`；完整答案：[dcp_demo.py](dcp_demo.py)。公开合成状态为两个长度为 2 的 float64 Tensor 与一个 int64 标量：`weight=[1,2]`、`momentum=[0.1,0.2]`、`step=7`，有效载荷 40 字节，不包括序列化／metadata 成本。

暂存成功后，将 live 状态分别加 100、10、1。保存成功并读回后，三个值都应与原始快照 `torch.equal`，而 live 状态已经变化。浅字典复制与 live weight 的 data pointer 相同，应记录为别名而不是隔离。

用 Event 控制异步 CPU staging 和 `FileSystemWriter.write_data` 的入口；不使用 `sleep`。复制门关闭时，两个完成句柄均待完成；复制放行并成功后，upload 仍待完成、metadata 不存在；写门放行后，等待保存结果再 load。同步 staging 返回单 Future，也必须等保存结果。

`ControlledStager` 为一次性教学实现，内部使用不缓存的 `BlockingAsyncStager`，不代表 GPU `DefaultStager`、pinned memory、共享内存或 D2H copy 已被验证。`GateWriter` 将真实文件写入暂时挡在门后；失败用例在写入前主动抛异常。它没有模拟中途写满磁盘、掉电或存储服务故障。

失败用例的 `upload.done()` 为 True，但 `result()` 抛 `CheckpointException`，没有 metadata。单独的 `probe_failed_staging` 记录已完成、带异常 staging Future 的版本行为：当前安装包的 staging 句柄报成功，但 upload 以 RuntimeError 失败。该探针只断言失败不能变成保存成功，不将这个报告差异当成所有版本的稳定保证。

调用者自有 stager 在保存任务不再使用它之后关闭。异常路径也释放门控、处理仍运行的上传并关闭线程池。本示例只读回自己刚生成的文件；临时二进制 checkpoint 不归档，公开数值、文件名、阶段状态和结果保存在 [results.json](../evidence/runs/results.json)。

参考答案的验收脚本见 [test_answers.py](test_answers.py)，默认导入答案模块。学习者验证自己的实现时可替换相同函数的导入；题目不能只靠“生成了文件”通过。

真实执行清单、18 项测试及 `-O` 模式、Ruff 结果在 [verification.json](../evidence/runs/verification.json)。单进程／macOS 提示如实保存，不表示进行了分布式实验。独立代码复核与验证边界见 [review.md](../review.md)。
