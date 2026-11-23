# 参考资料与证据

主题：训练 profiler 的 CPU 提交、设备活动与通信等待。访问日为 2026-11-23，Asia/Shanghai；实际源码获取时间为 `2026-11-23T11:07:41.918461+08:00`。

## 关键论断与来源

| 编号 | 需要证实的论断 | 一手来源与定位 | 本地归档 | 范围与限制 |
| --- | --- | --- | --- | --- |
| S1 | step 编号驱动采样，回调编号不等于被采样编号 | PyTorch `profiler.py::schedule/step` 与动作转换表 | [schedule](evidence/upstream/pytorch/schedule.txt)、[转换表](evidence/upstream/pytorch/schedule-transitions.txt)、[step](evidence/upstream/pytorch/step-advancement.txt) | 固定 main，另读 stable；真实 CPU 验证 |
| S2 | CPU 插桩经过原生回调记录字段与关联 ID | `prepare_trace` → `_start_trace` → C++ TLS callback → `begin_op` | [准备入口](evidence/upstream/pytorch/prepare-kineto.txt)、[原生入口](evidence/upstream/pytorch/native-start.txt)、[C++ 注册](evidence/upstream/pytorch/callback-registration.txt)、[进入](evidence/upstream/pytorch/record-function-callback.txt)、[关联](evidence/upstream/pytorch/correlation-push.txt) | 普通 Kineto 路径，不覆盖 Cuspy 实现 |
| S3 | CPU self 扣除同步子范围，device total 不做时间并集 | `profiler_util.py::_populate_cpu_children/FunctionEvent` | [父子条件](evidence/upstream/pytorch/cpu-parent-tree.txt)、[total／self](evidence/upstream/pytorch/self-and-device-time.txt)、[self device](evidence/upstream/pytorch/self-device-time.txt) | CPU 线程范围，不能替代 OS CPU 利用率或 GPU 硬件计数 |
| S4 | 关联记录须过滤无效 external ID，不能按时间邻近乱配 | `profiler.py::_parse_kineto_results` | [main 关联](evidence/upstream/pytorch/correlation-link.txt)、[stable 关联](evidence/upstream/pytorch/stable-correlation.txt) | 未在本机实测 CUDA 对应关系 |
| S5 | CPU API、调度、kernel、GPU operation 分层 | Nsight Systems User Guide，NCCL Execution Concepts／Timeline Correlation | [官方阅读笔记](evidence/official-reading-notes.md)、[检索元数据](evidence/web-retrieval.json) | 长通信的根因需控制实验，本文只提出可检验机制 |
| S6 | shape／stack 开销与 CUDA legacy fallback 需明确 | PyTorch 2.15 profiler API，activities／Note／export／schedule | [官方阅读笔记](evidence/official-reading-notes.md) | 读取官方说明，未运行 CUDA fallback |
| S7 | Chrome JSON 时间字段为 μs，X 是完整事件 | Perfetto 外部格式说明与 FAQ | [格式检索](evidence/trace-format-retrieval.json)、[官方阅读笔记](evidence/official-reading-notes.md) | 解析器只处理 X；未执行查看器渲染 |
| S8 | 累计、并集、交集与独占时间是不同口径 | 基础集合运算及独立整数格 oracle | [完整实现](code/answers.py)、[测试](code/test_answers.py)、[结果](evidence/runs/results.json) | 合成 GPU 时间，不推断硬件利用率或实际加速 |

## 源码与版本

- 官方远端：`https://github.com/pytorch/pytorch.git`；默认分支为 `main`。
- 当日 fetch commit：`8aa44e99457abde4f2c0dd6156a09f5407ac90e0`。
- 最新正式 release：[`v2.15.0`](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，发布于 `2026-10-28T17:09:49Z`；tag commit 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。
- 本机 CPU 实测安装包：`2.15.0`，构建 SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。未把安装包、tag 和 main 混为同一构建。
- 获取命令、输出与退出码：[source-update.json](evidence/source-update.json)。使用 `fetch --no-tags` 和明确 tag ref；没有 checkout、reset 或清理上游缓存工作树。
- 19 个原样源码片段，共 500 行；每项都含完整 SHA、源文件路径、行号、永久链接及双重摘要：[source-manifest.json](evidence/source-manifest.json)。完整 [LICENSE](evidence/upstream/pytorch/LICENSE)、[源文件头](evidence/upstream/pytorch/source-headers.json) 保留。
- 对 main 与 stable 实际阅读的路径、函数、分支和统计语义：[源码阅读笔记](evidence/source-reading-notes.md)。

## 官方文档与论文边界

[PyTorch 2.15 profiler API](https://docs.pytorch.org/docs/2.15/profiler.html)、[官方 recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[Nsight Systems 当前 User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[2026.4 归档 User Guide](https://archive.docs.nvidia.com/nsight-systems/2026.4/UserGuide/index.html)、[2026.3 Analysis Guide](https://archive.docs.nvidia.com/nsight-systems/2026.3/AnalysisGuide/index.html)、[Perfetto 外部格式](https://perfetto.dev/docs/getting-started/other-formats) 与 [兼容 FAQ](https://perfetto.dev/docs/faq) 用于语义和观测边界。

网页不是固定 Git 版本；访问日和 provider 已缓存获取日分别记录。没有复制完整手册，自己的关键阅读结论在 [官方阅读笔记](evidence/official-reading-notes.md)。本文不介绍论文算法，故未寻找无关原始论文；这一选择及基础 DAG／区间模型的边界已在笔记说明。

## 可复现证据

练习、完整参考代码、命令和验收条件见 [code/README.md](code/README.md)。[verification.json](evidence/runs/verification.json) 记录实际环境、命令、退出码、5 个 Python 文件摘要和测试日志位置；最终为 22 项测试，普通与优化模式通过。首次实现和审查回归的失败记录保留，不能用它们代表最终状态。

[results.json](evidence/runs/results.json)、[答案输出](evidence/runs/answers-run.txt)、[CPU 表](evidence/runs/cpu_table.txt) 和 [真实 CPU trace 投影](evidence/runs/cpu_trace.json) 属于最终运行。[合成 GPU trace](evidence/runs/synthetic_gpu_trace.json) 的 provenance 明确标注合成。原始临时 CPU JSON 未归档，摘要及精确投影规则保留；名称保留并经过人工隐私检查。

历史 `pre-review-*` 是独立审查前的 20 项测试与一次 CPU 输出，`initial-*` 保留首次 19 项测试／lint 阶段，最终结论使用无此前缀的 22 项测试记录。未保留全部临时重复成功运行，未伪造原始日志。

[Python 审查报告](evidence/python-review.json)、[两轮审查](review.md)、[技术核验](evidence/technical-audit.json)、[交付检查](evidence/delivery-check.json) 和 [SHA256SUMS](evidence/SHA256SUMS) 共同描述发布范围；清单排除自身及交付报告，避免循环摘要。
