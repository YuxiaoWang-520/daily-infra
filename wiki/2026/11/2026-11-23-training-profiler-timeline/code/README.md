# 两道练习：把时间轴与计时口径对上

使用已经具备 PyTorch 的 Python 环境，不需要 CUDA、TensorBoard 或新增依赖。本文实跑环境为 Python 3.12.16、PyTorch 2.15.0、Darwin arm64；只支持 CPU activity。命令从本篇目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir /tmp/public-profiler-demo
```

第二条命令在指定目录写入 `synthetic_gpu_trace.json`、`cpu_trace.json`、`cpu_table.txt` 和 `results.json`。重复执行会覆盖这些同名教学产物，请使用自己的输出目录。本次真实记录保存在 [运行清单](../evidence/runs/verification.json) 和 [答案输出](../evidence/runs/answers-run.txt)。

## 练习一：通信时间到底有多少被计算覆盖？

起始代码：[exercises.py](exercises.py) 中的 `overlap_summary`。输入为同一设备、同一时间轴上的两组半开区间，以及一个正长度窗口；单位全部为微秒。调用者已经知道哪些区间属于计算、哪些属于通信，本题不靠 kernel 名称自动猜测类别。

输入：`compute=[(2000,8000),(8000,11000)]`、`communication=[(5000,13000)]`、`window=(0,15000)`。输出 8 项：窗口长度、计算并集、通信并集、交集、通信减去计算的长度、两类总并集、窗口内未覆盖部分，以及裁剪后的各原始活动时长之和。键名见参考实现签名和正文。

验收值依次为 `15000,9000,8000,6000,2000,11000,4000,17000`。必须处理无序、嵌套、相邻、重复、零长度、越过窗口、空列表和生成器。负时间戳可以表示相对时间；负 duration、倒置区间、非有限数和非正窗口必须拒绝。非数字类型抛出 `TypeError`，非法数值抛出 `ValueError`；整数时间戳须能无损转换成 binary64，避免大整数被静默舍入。浮点输入本身的量化误差仍由调用者承担。多个设备或未对时的主机不能直接合并。

解题时先裁剪，再分别排序、合并两组区间，最后用双指针计算交集。重复事件应增加“原始时长之和”，但不能增加并集。完整答案在 [answers.py](answers.py)，重要边界与独立整数格 oracle 在 [test_answers.py](test_answers.py)。区间模型不推断因果、不估算 SM 利用率，也不把未覆盖区间判为 GPU 空闲。

## 练习二：采到的究竟是哪几步？

起始代码：[exercises.py](exercises.py) 中的 `profile_training`。用公开合成张量构造四步 SGD：`X[16,8] @ W[8,4]`，对 `target[16,4]` 求均方误差，`float64`、单线程、学习率 `0.01`。初始化完全由 `arange` 决定，无随机数据和生产输入。

每步分别标记 `phase::input`、`phase::forward`、`phase::loss`、`phase::backward`、`phase::update`。输入准备只是 CPU clone，不能据此验证磁盘、DataLoader 或 H2D 吞吐。使用 `activities=[CPU]`，`wait=1,warmup=1,active=2,repeat=1`，在每步末尾调用 `prof.step()`，在 `on_trace_ready` 中导出。关闭 shape、stack、memory 和 FLOP 采集。

验收：只出现 `ProfilerStep#2`、`ProfilerStep#3`；回调看到 `step_num=4`；每种阶段标记各 2 个；`phase::forward` 有 CPU 子事件且 self time 不大于 total time；四步损失和最终权重与相同算术的无 profiler 基准完全一致。退出时恢复调用者线程数。不要断言某算子必须耗时多少微秒。

完整答案在 [cpu_profile.py](cpu_profile.py) 的 `run_cpu_profile`，命令入口为 [run_answers.py](run_answers.py)。学习者的 `profile_training` 可以调用自己实现的相同数据契约；参考测试默认导入答案模块，验证自己的实现时应相应调整导入。

本次输出为 [真实 CPU trace 投影](../evidence/runs/cpu_trace.json) 和 [聚合表](../evidence/runs/cpu_table.txt)。投影只保留 `cpu_op`／`user_annotation` 的完整事件，删除参数和全局元数据、重映射进程／线程号、平移时间原点，并原样保留 `dur`。它没有 flow、原始关联参数或完整元数据，不能替代完整生产 trace。原始临时 JSON 未归档，仅保留其摘要和投影变换记录。事件名称保留，公开前已经检查；此函数不是适用于任意生产 trace 的通用脱敏器。

解析器仅处理 `ph="X"`；不会配对 `B/E`，也不会从 flow 推导依赖。时间戳平移溢出会拒绝。可在支持 Chrome JSON 的本地查看器中打开两份 JSON；GPU 那份始终标注为合成，CPU 那份标注为真实投影。本次未执行查看器视觉渲染或 CUDA/NCCL 实测。

`capture_wall_seconds` 包括四步带标记训练、采集阶段切换与导出；它不等于两个 active step 的范围之和，也不构成无采集器的训练吞吐基准。
