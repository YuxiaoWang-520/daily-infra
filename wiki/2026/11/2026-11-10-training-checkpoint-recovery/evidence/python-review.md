# 独立 Python 审查

日期：2026-11-10（Asia/Shanghai）。实际审查者：独立默认代理 checkpoint_python_fallback。最初调用专用 python-reviewer 时，底层 gpt-5.4 被当前 ChatGPT 账号拒绝，代理在读取代码前退出；随后由可用代理实际执行审查。没有把失败的专用调用记作已完成审查。

## 实际范围与方法

只读检查本篇 7 个 Python 文件与 code/README.md，关注快照隔离、optimizer 装载、RNG、数据游标、新进程对照、临时文件与失败清理。没有改写代码或安装依赖。五进程完整实验由主执行者运行，独立审查者对其启动与比较逻辑做静态核验，没有重复声称运行全套。

独立审查者实际执行两段 Python 脚本：一段复现 optimizer 输入状态被修改；另一段运行正常 roundtrip、序列化半写失败、fsync 失败、replace 失败，共 4 项通过。

## 发现、修正与关闭

| 级别 | 实际发现 | 修正 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | restore 的 optimizer.load_state_dict 可能复用 CPU 同 dtype 的输入 Tensor；恢复后训练一步把输入 Adam step 从 5 改为 6，矩也改变。重复使用该快照相对未污染副本出现约 0.00209515 参数偏差 | 装入 optimizer 时 deepcopy，保留输入快照；新增输入不变和重复恢复等价两项回归 | 修正前两项均真实失败，见 restore-red-tests.txt；修正后独立运行两项均通过，问题关闭 |
| LOW（note） | 练习一模板与测试导入答案文件之间缺少复制说明，单改模板不会改变被测实现 | code/README.md 与 exercise_1.py 都说明复制到 checkpoint_io.py 后再测试 | 独立复核两处说明，问题关闭 |

修正前失败输出：[restore-red-tests.txt](runs/restore-red-tests.txt)。主执行者最终 [26 项测试](runs/tests.txt) 全部通过。

## 独立复验的实际命令与结果

工作目录为本篇 code/，实际执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest \
  test_resume_probe.SnapshotTests.test_restored_training_does_not_mutate_input_optimizer_snapshot \
  test_resume_probe.SnapshotTests.test_repeated_restores_from_same_snapshot_are_equivalent -v
```

代理报告实际结果：2 项通过，0.376 秒。此小范围复验未另保存标准输出文件，报告以代理实际返回为依据；完整主执行者测试有原始输出。

最终结论：限定审查范围内未发现未解决的 CRITICAL／HIGH／MEDIUM／LOW。边界仍为固定 CPU、单写者、调用方自有路径、完整更新边界；未验收 GPU、DCP、worker、断电或多写者恢复。
