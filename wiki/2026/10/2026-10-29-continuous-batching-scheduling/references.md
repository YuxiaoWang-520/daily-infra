# 参考资料、版本与证据

主题：请求为何需要逐步调度，batch 大小由什么决定？检索日期：2026-10-29，Asia/Shanghai。实际获取时间、命令、版本、字节数及 SHA-256 分别见 [versions.json](evidence/versions.json)、[源码清单](evidence/source-manifest.json) 和 [论文获取记录](evidence/paper-acquisition.json)。

## 论断与证据对应

| 核心论断 | 一手来源与定位 | 本地归档及限制 |
| --- | --- | --- |
| 可以把调度边界移到每轮模型执行之间，允许完成退出与新请求加入 | [Orca，OSDI 2022，第 3 节 Figure 3／4、第 4.2 节 Algorithm 1](https://www.usenix.org/conference/osdi22/presentation/yu)。 | [原论文阅读笔记](evidence/paper-reading.md)；A—D 为自选数字，未引用性能倍率。 |
| 当前引擎计划按轮交给执行器，再处理模型输出 | [EngineCore.step](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/engine/core.py#L674-L704)。 | [原文件](evidence/upstream/vllm-project/vllm/vllm/v1/engine/core.py)；以普通 step 说明机制，不作为全部异步路径的同步保证。 |
| 请求数与本轮处理的输入 token 数是不同约束 | [配置语义](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/config/scheduler.py#L48-L76)、[SchedulerOutput](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/output.py#L234-L247)。 | [配置](evidence/upstream/vllm-project/vllm/vllm/config/scheduler.py)、[输出结构](evidence/upstream/vllm-project/vllm/vllm/v1/core/sched/output.py)；包含本次 main 的 active／scheduled 可选细分，不断言稳定版存在同样字段。 |
| 一轮新处理量与已知／计算进度差额相关，普通 Decode 多为 1 个新输入位置 | [统一进度](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L699-L722)、[实际差额及裁剪](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L839-L857)。 | [scheduler.py](evidence/upstream/vllm-project/vllm/vllm/v1/core/sched/scheduler.py)、[源码笔记](evidence/source-reading.md)；异步占位与投机不能直接套用基础差额公式。 |
| 分配失败会阻碍准入；running 缺 KV 时还可能抢占重算 | [容量检查](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/kv_cache_manager.py#L517-L587)、[抢占更新](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1812-L1842)。 | [管理器原文件](evidence/upstream/vllm-project/vllm/vllm/v1/core/kv_cache_manager.py)；练习的 ceil 页数是无共享、无命中、完整 Prefill 简化，不包含未来输出容量保证。 |
| continuous batching 不自动保证严格 FCFS，也不保证每轮都选中所有 running 请求 | [跳过分支](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L894-L909)、[结果约束](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1618-L1637)。 | 当前源文件；不能继承 Orca 特定算法的公平性保证。 |
| TTFT、逐事件 ITL 与请求 TPOT 的起止点、聚合权重不同 | [官方指标文档](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/docs/design/metrics.md#L51-L81)、[时间戳代码](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/metrics/stats.py#L525-L558)、[请求级 TPOT](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/metrics/stats.py#L594-L615)。 | [stats.py](evidence/upstream/vllm-project/vllm/vllm/v1/metrics/stats.py)、[loggers.py](evidence/upstream/vllm-project/vllm/vllm/v1/metrics/loggers.py)、[文档原文](evidence/upstream/vllm-project/vllm/docs/design/metrics.md)；用户端另有网络／缓冲，合成时间戳不是服务端实测。 |
| 调参须同时观察预算、延迟及重算，不能只看 batch 上限 | [官方优化指南](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/docs/configuration/optimization.md#L29-L80)。 | [文档原文](evidence/upstream/vllm-project/vllm/docs/configuration/optimization.md)；正文用适用条件讨论，未采用通用最优值或实测吞吐结论。 |

## 源码版本与获取

- 官方远端 `https://github.com/vllm-project/vllm.git`；通过 GitHub 元数据核验默认分支为 `main`。
- 检查缓存无改动后，实际执行 `git fetch --depth=1 origin main` 及 `git switch --detach origin/main`，退出码均为 0。缓存从 `84e455367336b62547d1f1273e9b50bc7f8edb47` 更新到 `0c81a0371f586cc50a0daad6d216fbac21cf1a2d`；该 commit 时间为 2026-10-29T02:41:42Z。
- 每个归档文件均由 `git show SHA:path` 导出，保留原字节与许可头；11 项完整清单记录路径、命令、时间与哈希。源码受 [Apache 2.0 许可证](evidence/upstream/vllm-project/vllm/LICENSE) 约束。
- 2026-10-29 查询 GitHub latest release 为 [v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，发布于 2026-10-18T06:50:48Z，tag SHA 为 `5df63a7203cdb9328d824a27b682e77c7b9928be`。仅核对稳定版发布元数据，未逐行比较；本篇实现分析明确绑定上述 main。
- 实际入口、状态、分支及教学简化逐项写在 [source-reading.md](evidence/source-reading.md)。没有安装、构建或执行 vLLM，没有模型权重或真实 GPU 运行证据。

## 论文、文档与图源

Orca 的作者、正式出版版本、阅读章节及差异见 [paper-reading.md](evidence/paper-reading.md)。正式 PDF 本次完整下载成功，提取成功，SHA-256 记录在获取清单；原文只放忽略的研究缓存，发布目录保存自己的机制笔记与精确定位，未复制论文图或整段原文。

本次实际访问了 [官方优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/) 和 [官方指标文档](https://docs.vllm.ai/en/latest/design/metrics/)，正文最终引用使用同日固定 commit 的文档源，避免浮动页面成为实现版本依据。原论文原则与当前默认分支代码分开解释；未使用二手博客支撑核心事实。

正文 Mermaid 为自绘教学流程，源文件是 [scheduling-loop.mmd](assets/scheduling-loop.mmd)；时间线和 A—D 请求均为合成示例。图不是生产 trace，也没有包含内部日志、代码、项目标识或用户数据。

## 练习、答案与验证边界

[练习入口](code/README.md)、[起始代码](code/exercises.py)、[类型](code/models.py)、[答案](code/answers.py)、[测试](code/test_answers.py)、[演示](code/run_lab.py) 均可本地阅读。

实际环境与每条命令见 [summary.json](evidence/runs/summary.json)：Python 3.10.13、macOS arm64，使用已有 Conda 环境，未安装依赖。环境记录工具检测到 PyTorch 2.9.1、CUDA 不可用；练习只用标准库。

- [真实测试输出](evidence/runs/tests.txt)、[逻辑排程及预算账本](evidence/runs/lab.json)、[Ruff check](evidence/runs/lint.txt)、[Ruff format](evidence/runs/format.txt)。
- [首次 RED](evidence/runs/initial-red-tests.txt)：先建立测试、尚未创建答案模块时的真实预期导入失败；不作为最终验证状态。
- [独立 Python 审查与修正](evidence/python-review.md)、[两轮文章自审](review.md)。固定模型审查未启动成功与后续实际完成的替代审查分开记录。
- [核心文件校验和](evidence/SHA256SUMS)、[归档链接与源码锚点检查](evidence/runs/archive-check.txt)。原始上游行尾空格保留，空白检查例外仅覆盖 `evidence/upstream/**`；本文与练习没有豁免。

CPU 运行验证的是教学规则、时间记账和预算守恒；固定每轮成本不是 GPU 测量。未验证 vLLM 集成、异步／多卡、显存峰值、线上 TTFT／ITL 或吞吐提升。
