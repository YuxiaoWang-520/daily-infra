# 线程映射与扇区练习

先做正文两道题；完整答案为 [answers.py](answers.py)。已有 Python 3.10+ 标准库即可运行：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

9 项测试应通过。诊断给出 10 元素、每块 4 线程的全部 12 行映射，以及 32 个 FP32 lane 在连续对齐、错开 4 字节、跨步 128 字节时的扇区数 4／5／32。

这是 CPU 上的编号与区间模型，不模拟线程执行时间、block 调度、缓存、实际内存事务或 occupancy。数学 block_size 不验证设备上限；n=0 表示不启动。扇区模型跨界用例不代表任意未对齐访问都能由单条 GPU 加载执行。实测记录见 [evidence/runs/](../evidence/runs/)。
