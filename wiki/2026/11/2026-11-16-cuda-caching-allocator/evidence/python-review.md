# Python 独立代码审查

日期为 2026-11-16，时区 Asia/Shanghai。范围为本篇 `code/` 中 7 个 Python 文件，包含不可变类型、统计答案、分配器答案、两份起始代码、测试与实验入口。

## 执行角色与范围

固定 python-reviewer 代理返回了 `The 'gpt-5.4' model is not supported when using Codex with a ChatGPT account.`，未执行审查。随后由已有通用代理 `checkpoint_python_fallback` 独立完成只读代码审查，主代理负责修复和完整验证。正文技术证据与教学表达两轮审查由主代理自审。

## 发现、修正与复验

独立审查发现 1 项 MEDIUM。实验入口把 `complete_deferred()`、`release_free_segments()` 等必要状态变更写在 `assert` 中。`python -O` 会删除这些调用，导致延迟释放案例抛出 MemoryError，缓存循环报告归还 1 个段时实际还保留 8,388,608 字节。主代理先实际运行优化模式复现，并保存修复前代码哈希和失败输出。[失败命令与哈希](runs/01-optimized-red-verification.json) · [失败输出](runs/01-optimized-red-lab.txt)

修正把状态变更移到断言之前，先保存返回值，再检查结果。缓存循环增加实际 `released_bytes` 与 `final_reserved_bytes` 输出，释放段次数根据实际结果产生。新增 `LabTests.test_optimized_execution_performs_state_transitions`，用子进程执行 `-O`，检查最终 reserved 为 0、归还 8 MiB、延迟完成后分配 8 MiB，以及预期 MemoryError 场景。

独立代理复验该回归，1 项测试通过，用时 1.993 秒，确认原 MEDIUM 已解决，限定修复范围内无遗留或新增问题。主代理的完整 24 项测试通过，最终命令、退出码与代码哈希统一归档。[完整运行清单](runs/06-verification.json)

## 独立代理实际执行的验证

初轮执行 5 项定向 unittest，全部通过，用时 0.007 秒，覆盖快照隔离、未知／重复／等待中 handle、500 步随机字节守恒、失败后缓存回收及延迟完成。另用普通与优化模式的探针复现上述实验入口问题。

复验命令为 `python -m unittest test_answers.LabTests.test_optimized_execution_performs_state_transitions -v`。执行器为已有 ai-infra 环境 Python，设置 `PYTHONDONTWRITEBYTECODE=1`。审查代理执行只读检查，修正由主代理完成。

## 验证边界

结论覆盖 CPU 单池普通段的教学实现、状态转移、预算、异常恢复与实验输出。CUDA stream、真实分配舍入、graph pool、expandable segments 与 GPU 性能由正文明确限定，仍需对应设备验证。
