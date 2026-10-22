# 广播练习

先读正文第 7 节的题目，再实现自己的版本。答案在 `answers.py`，可以直接阅读、复制到你自己的练习文件中；这里不自动安装依赖。

| 文件 | 内容 |
| --- | --- |
| [answers.py](answers.py) | 广播 shape 推断、推理专用的分块平方距离 |
| [test_answers.py](test_answers.py) | 12 项验收：shape、零 stride、共享与复制、重叠写入、梯度、分块与边界 |
| [run_lab.py](run_lab.py) | 固定小输入，输出存储属性、广播值、梯度与距离矩阵 |

在本目录运行：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

需要 Python 3.10+ 与已经安装的 PyTorch；实测为 Python 3.10.13、PyTorch 2.9.1 CPU。测试应报告 `Ran 12 tests` 和 `OK`。诊断输出中扩展 stride 为 `[0,1]`，原 bias 和扩展视图的 storage 均为 12 字节，repeat 为 24 字节，求和损失的 bias 梯度为 `[2,2,2]`。

`pairwise_sqdist` 显式关闭 autograd，只支持普通二维 float32／float64、相同设备与特征宽度。测试覆盖 CPU float32／float64，不宣称已验证 CUDA 或 Tensor subclass。小块差值尺寸受块大小控制，未测进程 RSS 或 GPU 显存峰值。

真实结果见 [运行目录](../evidence/runs/)。

返回值是推理 Tensor，本题不支持把结果直接当成可训练计算图的中间量。
