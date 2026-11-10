# 真实运行与验证边界

日期：2026-11-10（Asia/Shanghai）。机器 wyx-macmini.local，Apple M6，macOS 27 arm64。现有 Python 3.12.16 路径为 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python，torch 2.15.0（build Git d0f2af8b66b20da4baa50644c0cd3f887a2eae2e），CUDA 不可用。未安装或升级依赖。

## 测试与修复经过

先编写测试和 NotImplementedError 接口，执行 unittest 得到真实失败：[red-tests.txt](runs/red-tests.txt) 共运行 17 项、18 个 error（包括 setUpClass），证明未实现接口不能通过。

完成答案后 [first-pass-tests.txt](runs/first-pass-tests.txt) 中 24 项通过。独立代码审查发现恢复可能反向修改输入快照；新增两项回归在修正前都失败，见 [restore-red-tests.txt](runs/restore-red-tests.txt)。使用 deepcopy 隔离被加载的 optimizer state 后，最终 [tests.txt](runs/tests.txt) 中 26 项全部通过，退出码 0，unittest 用时 5.664 秒。计时只是该测试命令的实际耗时，没有作为训练性能测量。

实际最终命令（从本篇根目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff check code
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff format --check code
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/daily-infra-nov10-final-lab.json
```

上述四条最终命令退出码均为 0。Ruff 原始记录：[检查](runs/ruff-check.txt)、[格式](runs/ruff-format.txt)。实验输出：[JSON](runs/lab.json)、[stdout](runs/lab-stdout.txt)。输出原样归档，路径保留实际运行位置；未把草稿路径改写成仿佛在发布目录运行过。

## 实验设计与观察

合成数据 20×3，batch size 2；Linear(3,4) → Tanh → Dropout(0.4) → Linear(4,1)，CPU float64，Adam lr 0.03、foreach=False，StepLR 每 3 步乘 0.8。显式排列 seed 77，连续／保存生产者 seed 2026，恢复对象构造 seed 999。每步还有 Python random 产生的输入扰动。实验固定 1 个算子线程。

基线连续训练 10 步；生产者在第 5 步保存，再运行至第 8 步后丢弃内存对象。五个真实新 Python 进程分别从同一文件读取，执行全部恢复、漏 optimizer、漏 RNG、漏 cursor、漏 scheduler。每个进程只在自己临时目录内产生输出，使用当前 sys.executable，参数列表调用 subprocess，不经 shell。

完整恢复的参数、Adam 矩和后 5 步 loss 最大绝对差均为 0；样本 ID、学习率、优化器步数和最终 RNG 一致。四个缺项对照都产生差异，正文数字来自 lab.json。模拟“已完成计算随后丢失”的边界；没有向操作系统发送崩溃／强杀信号。

## 安全与范围

I/O 测试注入半写、fsync、replace 异常，确认旧 checkpoint 可读且临时文件清理；它们属于进程内可捕获异常。文件发布接口只用于调用方自有路径、单写者；没有验证父目录同步、断电、强杀残留、NFS、对象存储或并发写者。

公开 restore 接口校验固定 schema、配置、排列和 cursor／step 一致性。它不是任意模型、数据集或不可信输入的通用验证器。weights_only=True 为加载方式的一部分，来源和配置仍由调用方负责。

本文没有 GPU／AMP、CUDA RNG、独立 NumPy RNG、梯度累积中途、DataLoader worker、多 rank DCP、跨 world size 或跨版本恢复实验。保存间隔公式与 9.09% 数字为教学估算，未用本机运行时间替代生产 I/O 性能。

完整代码文件哈希与实际环境见 [summary.json](runs/summary.json)。源码与文档证据独立归档，不把阅读结果计为实验通过。
