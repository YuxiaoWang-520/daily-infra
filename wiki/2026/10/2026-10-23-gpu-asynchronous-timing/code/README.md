# 异步与计时练习

已有 Python 3.10+、PyTorch 即可运行 CPU 练习和计时协议测试：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
python3 gpu_demo.py
```

9 项测试应通过。lab 的 0.1／3.1／5.1 全是合成逻辑毫秒。计时器的 CUDA 调用通过 mock 检查，未在 GPU 上验证。本机 gpu_demo 实际输出 SKIP，无计时数字；在目标 CUDA 环境运行后才会输出真实样本与正确性检查。

GPU recipe 范围：当前设备、同一不变的当前 stream，5 次预热、每批 20 次、5 批，热输入与预分配输出。事件时间除以调用数是批次均值；样本中位数不是逐请求 P99。调用者负责选择足够预热、目标数据与隔离条件，不得把内部 side stream、主机处理或传输默认计入。

完整答案：[answers.py](answers.py)。数据与日志：[evidence/runs/](../evidence/runs/)。
