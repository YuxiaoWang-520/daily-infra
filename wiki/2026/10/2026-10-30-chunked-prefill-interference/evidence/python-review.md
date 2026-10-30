# Python 独立审查记录

日期：2026-10-30（Asia/Shanghai）。范围仅为本篇五个原创 Python 文件：`models.py`、`exercises.py`、`answers.py`、`test_answers.py`、`run_lab.py`。不含第三方源码、尚未完成的文章正文或引擎 GPU 实现。

## 执行者、方式与限制

首先尝试调用窄范围 `python-reviewer` 专职审查，但其固定模型 gpt-5.4 在当前账户返回 `400 model_not_found`，没有完成审查，不能计作通过。

随后由已有独立审查代理 `capacity_python_fallback`（gpt-5.6-terra、max effort）进行只读 Python 审查。主代理提供了输入输出契约、明确的数值与模拟边界、既有测试记录；审查者独立读取五个文件，并实际运行当时的 `test_answers.py`（19 tests，退出 0）与 `run_lab.py`（退出 0）。审查者读取已有 Ruff 通过记录，没有重新运行 Ruff。没有委派正文技术或教学审查，也没有称该替代审查为固定专职模型的审查。

## 实际返回的发现

独立审查结论无 CRITICAL／HIGH／MEDIUM，返回两项 LOW 和一个数值边界提醒：

| 程度 | 审查者观察 | 主代理处理与验证 |
| --- | --- | --- |
| LOW | 构造位置的 `torch.arange` 未绑定输入设备；在 `with torch.device("meta")` 内传显式 CPU QKV，会触发 CPU／meta 不匹配 | 先新增回归测试，真实复现失败，再给两处 `arange` 显式绑定 `q.device`；测试输入工厂与 lab 输入显式使用 CPU；修复后 20 tests 通过 |
| LOW，测试保障 | 已测 `NaN` 和正无穷，没有测负无穷；现有 `isfinite` 实现已能正确拒绝 | 把负无穷加入三种输入位置的非有限值测试，实际通过 |
| 数值边界 | 有限但极大的 float32 Q/K 点积会溢出；审查者在相同极端输入下观测到完整 SDPA 也产生 `NaN` | 未把参考算法改为跨精度实现；正文练习详解与代码说明明确有限输入检查不保证所有极端数值的输出有限，精度结论只覆盖记录的普通尺度输入 |

修复前的默认设备回归原始错误保存在 [default-device-red.txt](runs/default-device-red.txt)：20 个测试中该测试报 `RuntimeError: Tensor on device cpu is not on the expected device meta!`，其余通过。更早的 [initial-red-tests.txt](runs/initial-red-tests.txt) 是编写参考答案前导入缺失模块的失败记录，不当作数值正确性证明。

## 修复后复核与接受决定

主代理按发现逐项修改，随后运行完整的 CPU 测试、lab、Ruff check 与 format 检查，均退出 0。最终归档目录的实际命令与结果见 [summary.json](runs/summary.json) 和 [tests.txt](runs/tests.txt)。修复反馈已发送给原审查代理；没有再声称它做了第二次全范围审查。

主代理接受修复：新增回归保护了临时张量设备选择；负无穷覆盖补齐；数值溢出限制已说明。当前无未解决 CRITICAL／HIGH，也无影响核心教学的 MEDIUM。最终正文、资料和交付状态仍由主代理进行技术与教学自审，见 [review.md](../review.md)。
