# Python 独立审查

2026-11-06，由独立代理 `/root/checkpoint_python_fallback` 以只读方式审查本篇 8 个 Python 文件及 code/README.md。该代理与主执行者分离，承担 Python 代码审查职责。固定 python-reviewer 角色在本会话此前执行时返回模型不受账户支持，实际完成审查的是可运行的独立代理。主执行者负责修复和最终验收。

审查范围包括资源预算的串行与重叠假设、数值类型和溢出、任务槽位与就绪批数、临时文件、spawn 导入与收尾、样本完整性、尾批、独立 Generator 和计时边界。审查者实际运行 4 项已有定向测试，并做两条边界复验。

## 初轮发现

| 等级 | 实际观察 | 修复与回归 |
| --- | --- | --- |
| MEDIUM（info） | `pipeline_budget(1e308,1e308,1,0,1,1,1)` 在有限性检查前转换 Fraction，抛出裸 OverflowError，偏离 ValueError 契约 | 合成时间有限性检查移到 worker 间隔转换前，补单 worker 回归 |
| MEDIUM（info） | next 失败后 traceback 保留迭代器，单纯 del 局部变量不能保证 worker 在临时目录删除前退出。故障注入观察到 2 个活跃子进程 | finally 显式执行固定版本的幂等收尾，回归保留真实 iterator 与异常，再断言 worker 退出 |

两条新增回归在修复前分别真实失败，记录见 [溢出红测](runs/review-overflow-red-test.txt) 与 [收尾红测](runs/review-cleanup-red-test.txt)。红测自身也在 finally 中清理故意保留的 worker。

## 定向复审

原独立审查者只读复核这两项修复及新增回归，实际运行两项测试，全部通过，用时 0.759 s。审查者确认两项 MEDIUM 均可关闭，本轮没有遗留 CRITICAL、HIGH 或 MEDIUM。

主执行者随后完成 33 项全量测试，以及 Ruff 检查和格式检查。最终原始记录见 [tests.txt](runs/tests.txt)、[lint.txt](runs/lint.txt) 与 [format.txt](runs/format.txt)。

## 接口边界

实验固定 PyTorch 2.15.0，多进程收尾使用其私有 `_shutdown_workers()`。此依赖已经在题目详解、代码注释与运行说明中声明。升级版本时需要核验该接口及异常路径。CPU 合成实验没有覆盖 CUDA、真实存储或持久 worker 模式。
