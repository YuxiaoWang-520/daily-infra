# CPU 练习与完整答案

本目录包含两个练习的起始合同和完整参考实现。使用已有 PyTorch 环境，不安装依赖；本次环境为 Python 3.12.16、PyTorch 2.15.0、Darwin arm64，无 CUDA。

| 文件 | 用途 |
| --- | --- |
| [exercises.py](exercises.py) | 两题起始合同，刻意保留 NotImplementedError |
| [answers.py](answers.py) | 缩放转换、反缩放裁剪及累计的完整数学答案 |
| [runtime_demo.py](runtime_demo.py) | 真实 CPU GradScaler、SGD、恢复、错误对照和 BF16 autocast |
| [test_answers.py](test_answers.py) | 23 项验收测试 |
| [run_answers.py](run_answers.py) | 完整运行并导出严格 JSON，不覆盖已有 results.json |

在本篇目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v
PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v
ruff check code
ruff format --check code
PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir /tmp/amp-wiki-nov27-example
```

`python` 应为已有 ai-infra 环境；归档清单使用 `<python>`／`<ruff>` 表示实际环境的可执行文件，避免发布个人绝对路径。临时输出目录需没有 `results.json`；第二次使用另一个自有目录。归档命令 cwd 为 `code`，所以 CLI 采用等价的相对路径。

## 题一合同

`Quantization` 为不可变结果，字段是原梯度、scale、dtype、直接低精度保存值、缩放后低精度保存值、恢复值。输入要求在正文与函数 docstring 中列明。这里模拟保存点的舍入，不能代表每种 GPU 内核的 FTZ 行为。缩放结果可以有意成为 Inf；导出时转为字符串。

`unscale_and_clip` 采用不含 epsilon 的理想 L2 裁剪。真实 PyTorch 裁剪用 `1e-6`，按 `1e-6` 绝对容差比较其示例，不要求理想小数逐位一致。

## 题二输出合同

`training_trace(faults, restore_after=None, init_scale=8)` 返回字典：

- `records`：每次尝试的 `data_batches`、`applied_updates`、`scheduler_steps`、`injected_nonfinite`、`scale_before`／`scale_after`、`growth_tracker`、`scaled_gradient`／`unscaled_gradient`、`optimizer_step_called`、`step_return_is_none`、`parameter_before`／`parameter_after`、`momentum_before`／`momentum_after`。
- 顶层汇总：`data_batches`、`applied_updates`、`scheduler_steps`、`scheduler_last_epoch`、`final_scaler`、`scaler_enabled`。
- `faults` 必须全为 bool。恢复位置为 `0..len(faults)` 中的整数，拒绝 bool；初始 scale 必须在 FP32 中有限且大于零。

五次受控输入的 scale 为 `[8,16,8,8,16]`，有效更新为 `[1,2,2,3,4]`，最终参数约 `0.2268`。`CountingSGD` 计数只针对普通 SGD 的实际调用；不要泛化到任意 fused／自定义 AMP 优化器。调度器使用常数 LambdaLR，验证的是更新计时策略和状态恢复。

恢复只使用同进程内存深拷贝，没有序列化不可信文件，没有落盘、崩溃或随机数据恢复实验。参数和输入固定，显式指定 CPU，不改变默认 RNG。协议标量 loss 为 FP32，Inf 是检查前手工注入。CPU BF16 linear／MSE 是另外一个 autocast 实验；没有 GPU 性能或 FP16 训练下溢实测。

实际结果：[JSON](../evidence/runs/cpu/results.json)、[普通测试](../evidence/runs/unittest.txt)、[优化模式](../evidence/runs/unittest-optimized.txt)、[命令与源码摘要](../evidence/runs/manifest.json)。
