# 两道代码练习

参考答案面向本机已有 Python 3.12.16、PyTorch 2.15.0，普通 CPU 即可；只使用标准库和 torch。练习对象是调用方拥有的本地 checkpoint 路径。加载使用显式 `weights_only=True`；该选项不能代替对文件来源和训练配置的检查。

## 练习一：发布一个完整文件

起始文件：[exercise_1.py](exercise_1.py)。目标是在已有父目录中保存一个已冻结的字典；字典只含 Tensor、字符串、数字、容器等基础状态。返回值为空。先在目标同目录写唯一临时文件，再 flush、fsync、关闭，最后用 os.replace 发布。完成后，把两个函数复制到 checkpoint_io.py 的对应位置再运行测试；测试默认导入该答案文件，单独编辑起始文件不会改变被测实现。

验收：[test_checkpoint_io.py](test_checkpoint_io.py)。正常覆盖后读到新版本；序列化写出半截后失败、fsync 失败、替换失败时旧版本仍可读；首次写入失败没有目标文件；不自动创建父目录。题设只有一个写者。模拟错误会执行 Python 的 finally，强杀进程可能留下临时文件；目录项断电持久性未覆盖。

<details>
<summary>完整答案与推导</summary>

[checkpoint_io.py](checkpoint_io.py) 的关键是让最后一次可见名字变更发生在序列化结束后。临时文件与目标在同一目录，避免跨文件系统 rename。finally 只删除本次创建的唯一临时文件，不删除旧目标。成功返回的 os.replace 提供原子可见性；调用方仍要保证快照在写入期间不被训练修改。

</details>

## 练习二：让恢复后的第 6 步确实接着第 5 步

起始文件：[exercise_2.py](exercise_2.py)。沿用 [resume_probe.py](resume_probe.py) 的固定模型、20 个合成样本、显式排列、每批 2 个样本、Adam 与 StepLR。模型含 Dropout，每步输入还叠加 Python random 产生的小扰动。第 5 步保存，第 8 步放弃内存中的工作；新进程从文件恢复，再运行 5 步。

输入为 schema 1 的快照字典，输出为可继续训练的 Trainer。检查配置和样本排列，要求 cursor 等于 completed_steps × 2。恢复模型、优化器、调度器、排列和已消费进度；对象构造结束后恢复 Python 与 torch CPU 随机状态。比较完整恢复及四个缺项对照。

验收：[test_resume_probe.py](test_resume_probe.py)。完整恢复的最终参数、Adam 矩、逐步 loss、样本 ID、学习率及随机状态与连续训练一致；四个缺项对照必须产生可观察差异。另检查快照不随后续训练改变、schema／配置／排列／进度错误被拒绝。

<details>
<summary>完整答案与推导</summary>

[resume_probe.py](resume_probe.py) 中 Trainer.snapshot 对状态字典执行 deepcopy，固定第 5 步的值。restore 先校验 schema、配置和排列，再创建模型、优化器与调度器。调度器对象已经创建后才装入 optimizer 状态，避免初始化覆盖学习率。先加载调度器状态，再加载优化器状态，最后装回 RNG。装入优化器时再次 deepcopy，使恢复后的训练不会污染调用方保留的输入快照；回归测试验证同一快照可重复恢复。

从文件恢复时，剩余更新数为 10 − 5 = 5；数据起点为已消费的 10 个样本。漏恢复 cursor 会从第一个样本重新读，却仍声称已经完成 5 步。漏恢复 optimizer 会丢掉 Adam 的矩与步数；漏恢复 RNG 改变 Dropout 和输入扰动；漏恢复 scheduler 改变下一次衰减的时点。新子进程的构造 seed 故意设为 999，使遗漏 RNG 能被检测。

run_probe 为每种模式各启动一个全新 Python 进程，不复用生产者的对象。它保留调用者的随机状态和线程数；训练 fixture 本身会设置并消费全局 RNG，请勿并发使用。模型结构与参数组顺序固定，配置标签需随数据或模型语义变化而更新；这不是通用 checkpoint 兼容性验证器。

</details>

## 运行与证据

从本篇根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/checkpoint-lab-new-report.json
```

第二条命令要求输出文件尚不存在。实测报告与原始输出保存在 [运行记录](../evidence/runs/)，详见 [验证说明](../evidence/validation.md)。本实验没有 GPU、AMP、DataLoader worker、梯度累积或分布式恢复；这些系统需要额外状态和协调。
