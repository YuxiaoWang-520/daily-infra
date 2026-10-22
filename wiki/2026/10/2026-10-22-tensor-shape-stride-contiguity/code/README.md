# 练习、答案与复现

从文章的第 9 节开始做题，再查答案。这里没有自动安装依赖的脚本。

| 文件 | 用途 |
| --- | --- |
| [layout_answers.py](layout_answers.py) | 练习一、二：地址、默认连续性、按逻辑次序物化；只需标准库 |
| [torch_answers.py](torch_answers.py) | 练习三：拆头与合头，写明零复制约定与允许复制的边界 |
| [test_layout.py](test_layout.py) | 12 项标准库验收，含标量、空 Tensor、单例轴、边界与非法输入 |
| [test_torch.py](test_torch.py) | 14 项 PyTorch CPU 验收，核对值、stride、view／copy 与内存共享 |
| [run_lab.py](run_lab.py) | 输出稳定的布局属性与逻辑值，不输出随机内存地址，不测性能 |
| [ruff.toml](ruff.toml) | 固定本篇模块的导入分类，避免从不同工作目录检查时结果不同 |

在本目录运行：

```bash
python3 -m unittest discover -s . -p 'test_layout.py' -v
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

第一条不需要 PyTorch；后两条要求环境中已有 PyTorch。已验证环境为 Python 3.10.13、PyTorch 2.9.1、macOS arm64、CPU。代码要求 Python 3.10 或更高版本；不宣称已验证全部 PyTorch 版本或 GPU backend。

全部测试应报告 `Ran 26 tests` 和 `OK`。`run_lab.py` 重点检查输出中的：

- `view_flatten.step_slice.success` 为 `true`，一维 stride 为 `[2]`。
- `view_flatten.crop.success` 为 `false`。
- `head_layouts.merge_split_view.same_live_storage_as_base` 为 `true`。
- `head_layouts.merge_packed_bhsd.same_live_storage_as_base` 为 `false`。
- `channels_last` 的逻辑 shape 为 `[2,3,4,5]`，stride 为 `[60,1,15,3]`。

真实运行结果见 [诊断输出](../evidence/runs/lab.json) 与 [测试记录](../evidence/runs/tests.txt)。测试名数量不包含每个 `subTest` 内部的多个数据案例。

如果改写参考答案，先运行基础测试，再运行 PyTorch 对照测试。不同设备上的性能需要另设计实验；本目录输出只验证语义。
