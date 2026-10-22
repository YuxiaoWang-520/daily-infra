# Roofline 练习

先读正文第 8 节题目，再看 [answers.py](answers.py)。只需已有 Python 3.10+ 标准库。

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

预期 12 项测试通过。诊断中向量加法是模型里的 memory 限制，1024 GEMM 是 compute 限制，题设下界两倍的合成时间得到 50 TFLOP/s。全部是虚构硬件参数上的算术结果，不是运行 GPU 取得的时间。

账本限定：独立连续向量；GEMM `C=A@B` 不读取旧 C，理想 A/B 各读一次；计数采用一次乘加 2 FLOPs；速率使用十进制 TFLOP/s 与 GB/s。空工作强度返回 0.0 是占位约定；balanced 使用浮点相等判断。实测记录在 [evidence/runs/](../evidence/runs/)。
