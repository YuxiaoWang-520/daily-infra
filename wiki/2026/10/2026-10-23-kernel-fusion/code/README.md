# 融合成本与 CPU 语义练习

已有 Python 3.10+、PyTorch 即可，先做正文题目再看 [answers.py](answers.py)。

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

9 项测试应通过。`pipeline_model(2,3,4)` 得到 156／60 字节；1024 方形输入在融合 slowdown=1／5 的两种假设下，合成成本选择发生反转。参数是虚构的，不是 GPU 实测。

`chunked_affine_relu` 依序进行相同 CPU 逐元素操作，只在自己的新 buffer 上原地更新，释放每块后处理下一块，完整输出仍然需要。它是推理参考，不提供梯度；既没有 GPU kernel 融合，也没有峰值内存或速度验收。模型假定空任务不启动，不对应每个框架的实际 empty 处理策略。[日志](../evidence/runs/)。
