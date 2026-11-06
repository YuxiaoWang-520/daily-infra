# 练习与运行

本篇两道题分别建立流水线容量模型，并在真实 CPU DataLoader 上记录样本、启动与取数等待。输入均为合成教学数据。

## 现有环境

解释器为 `/Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python`，Python 3.12.16、PyTorch 2.15.0。实验使用 Mac mini 的 CPU，JSON 文件由每轮自己的 `TemporaryDirectory` 创建，写入使用独占模式。文件准备好后才启动计时，清理发生在迭代结束后。

在本篇目录执行完整测试与三轮实验。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py
```

## 练习一

在 [exercises.py](exercises.py) 实现 `pipeline_budget` 和 `exposed_pause`。

一份 batch 对共享存储的服务需求为 `R=storage_ms`，其 CPU 解码、变换与合批成本为 `C=cpu_ms`，每个 worker 依次完成 `R+C`。worker 数为正整数 `w`。传输与模型计算耗时分别为 `H=transfer_ms` 和 `G=compute_ms`，普通同一 stream 的设备端间隔为 `H+G`。

取以下理想稳态间隔下界。

```text
I_storage = R
I_worker = (R + C) / w
I_device = H + G
I = max(I_storage, I_worker, I_device)
```

只有显式指定 `allow_device_overlap=True`，才把传输和计算视作独立资源，用 `I_device=max(H,G)`，并分别报告 `transfer` 和 `compute` 的限制。它要求实际实现具备相应硬件、stream 与依赖条件。

模型假定均匀 batch、足够 CPU 核、理想调度，省略存储排队附加延迟、IPC、pinning、线程竞争和调度开销。原本完全串行的成本为 `R+C+H+G`。限制项表示该模型内的服务能力约束，需要结合实际阶段观测定位根因。

任务槽位 `K=w×prefetch_factor`，按每槽位恰有一份 batch 载荷估计 `K×batch_bytes`。这是队列载荷规划值，真实主机峰值还包含其他复制、dataset、worker、缓存及分配器成本。

输入时间均有限且非负，`G>0`。worker 数、prefetch factor 和 batch 字节数均为正整数，布尔值拒绝作为整数。非法输入或合成时间超出有限浮点范围时抛 `ValueError`。输入时间转换到浮点后，字节数与任务数保持整数计算。相同最大值的所有限制项按存储、worker、设备的顺序返回。

手算验收使用 `R=4 ms,C=16 ms,w=2,H=3 ms,G=10 ms,prefetch_factor=2,batch_bytes=64 MiB`。worker 间隔为 10 ms，同一 stream 的理想间隔为 13 ms，串行成本为 33 ms。4 个任务槽位的载荷规划值为 256 MiB。允许独立拷贝与计算时理想间隔为 10 ms，worker 与模型计算同时达到约束。

`exposed_pause` 输入为孤立的供给暂停 `S`、已经就绪的 batch 数 `Q`、每批消费间隔 `D>0`，返回 `max(0,S−Q×D)`。假定暂停开始时消费者立刻使用第一份就绪 batch，暂停结束后立刻恢复供给。`Q` 是非负整数，来自实际就绪状态。`S=40,Q=2,D=13` 得 14 ms，`Q=4` 时为 0。

完整答案位于 [answers.py](answers.py)，独立手算及边界测试位于 [test_answers.py](test_answers.py)。题目模板可以接受相同测试。

```bash
PYTHONDONTWRITEBYTECODE=1 EXERCISE_IMPL=exercises /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
```

## 练习二

在 [probe_exercises.py](probe_exercises.py) 实现 `run_trial`，结果类型见 [models.py](models.py)。还需实现模块顶层的 `JsonDataset`，供 spawn worker 导入。

每轮生成 40 个合成 JSON 文件，第 `i` 个文件包含 ID `i` 和数值 `[i,i+1,i+2,i+3]`。逐项读取并校验 ID 和数值后，解析为 FP64 Tensor，输出平方和 `4i²+12i+14`。每条数据记录源阶段、JSON 解码与校验阶段、Tensor 变换阶段的墙钟时间，以及实际 worker ID。源阶段包含人为等待，用来模拟供给延迟。

DataLoader 使用 batch size 4、`shuffle=False`、`drop_last=False`、`in_order=True`。分别用 0 个与 2 个 worker。多进程明确使用 `spawn`、prefetch factor 2，保留模块顶层类和 `if __name__ == "__main__"` 入口保护。单进程省略 multiprocessing-only 参数。给 DataLoader 独立的 CPU Generator，保护调用者随机状态。

计时从创建迭代器之前开始，单独保存创建耗时，每次 `next(iterator)` 等待和消费阶段耗时。消费阶段做一次 checksum 求和，再等待 20 ms，模拟消费者占用。每条源阶段等待 5 ms。人工等待参数允许 0—1000 ms。默认排除前 2 个 batch 后计算 next 等待中位数。总迭代耗时包含创建与耗尽时的 worker 收尾，阶段观测保留全部 batch。

一次 task 通常由一个 worker 读取一组索引并合批。统计中的某批源／解码／变换耗时为该批各样本观测之和。跨 worker 的工作可能重叠，汇总的工作时间与主线程的 next 墙钟等待分别解释。

验收要求如下。

1. ID 完整且顺序正确，每个 ID 恰好一次。所有转换结果逐项相同。默认 40 个样本的 checksum 总和为 92080，10 个样本的测试总和为 1820。
2. `sample_count=10,batch_size=4` 时保留尾批，三个 batch 的样本数为 4、4、2。0-worker 的来源标记为 -1，2-worker 的来源标记实际包含 0 和 1。
3. 全部时间观测非负，保留首批和稳态记录。计时结果随调度变化，测试只断言正确性、范围和结构。
4. 校验非法索引、错误 ID、错误数值结构、无效维度及延迟配置。warmup 必须留下至少一批用于统计。
5. 独立记录 iterator 创建、首批等待、稳态 next 等待，避免把进程启动摊入稳态。每个配置重复 3 次，输出原始 batch 观测和逐轮中位数。

完整答案位于 [loader_probe.py](loader_probe.py)，测试位于 [test_loader_probe.py](test_loader_probe.py)，组合执行位于 [run_lab.py](run_lab.py)。自写模板通过 `PROBE_IMPL=probe_exercises` 接受相同测试。

异常收尾也属于练习契约。当前实验固定 PyTorch 2.15.0，在 `finally` 中显式调用多进程迭代器的私有 `_shutdown_workers()`，确保异常 traceback 或调用者持有迭代器时也关闭 worker，再清理临时文件。此处依赖固定版本的内部接口，升级 PyTorch 时需要重新核验。故障注入测试保留真实迭代器和异常，检查两个 worker 已退出。

## 验证范围

实际验证包含 CPU JSON 读取、解码、Tensor 变换、DataLoader 顺序、多进程 spawn 和取数计时。小文件在迭代前刚创建，文件缓存状态未控制，人为等待主导负载。该实验展示的流水线行为依赖此配置，磁盘设备带宽、CUDA H2D、pin-memory 和真实模型性能需要各自的测量协议与设备验证。
