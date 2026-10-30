# 参考资料与证据

主题：长 Prompt 为什么会干扰其他请求，如何拆分 Prefill？检索日期为 2026-10-30，源码及版本采集时间为 `2026-10-30T11:15:25+08:00`。

## 核心论断与来源

| 编号 | 需要证实的论断 | 一手来源与精确定位 | 本地归档 | 证据范围与限制 |
| --- | --- | --- | --- | --- |
| P1 | 分块后要保持原因果关系，后块仍访问已有 KV | [SARATHI v1](https://arxiv.org/html/2308.16369v1) §4.2、Figure 6 | [论文笔记](evidence/paper-reading.md) | 算法机制；本篇自己的数值图解另行验证 |
| P2 | Decode、部分 Prefill、新请求在预算内组织，并权衡延迟和开销 | [OSDI 2024 出版方全文](https://www.usenix.org/system/files/osdi24-agrawal.pdf) §4.1—4.3、Algorithm 3、§5.4.1 | [论文笔记](evidence/paper-reading.md)、[获取记录](evidence/paper-acquisition.json) | 历史算法和实验讨论，不外推历史加速比 |
| S1 | 作者代码扣除本轮已占 token 后决定 Prefill 数量 | [SarathiScheduler](https://github.com/microsoft/sarathi-serve/blob/96f9911790ecc00af12ee9fae47cb8fa9ba0d199/sarathi/core/scheduler/sarathi_scheduler.py#L74-L96) 及 124—235 | [原始源码](evidence/upstream/microsoft/sarathi-serve/sarathi/core/scheduler/sarathi_scheduler.py) | 当天 main 快照的静态路径，有状态／内存限制及动态分支 |
| S2 | 当前 vLLM 按进度差、预算和可选阈值安排新计算位置 | [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L753-L855) 及 1331—1346 | [原始调度器](evidence/upstream/vllm-project/vllm/vllm/v1/core/sched/scheduler.py) | 不声称当前所有模式都是严格全局 Decode 优先 |
| S3 | 块的逻辑位置延续历史，KV 长度不等于当前 Query 长度 | [MRV1 位置构造](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/worker/gpu_model_runner.py#L1946-L1960)、[后端接口](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/attention/backends/flash_attn.py#L1404-L1428) | [源码阅读笔记](evidence/source-reading.md)、[runner 原文](evidence/upstream/vllm-project/vllm/vllm/v1/worker/gpu_model_runner.py)、[后端原文](evidence/upstream/vllm-project/vllm/vllm/v1/attention/backends/flash_attn.py) | MRV1、密集因果接口；没有执行 GPU 后端 |
| S4 | 部分 Prefill 不应交付新回答 token，computed 计数不能独立证明 GPU 已完成 | [进度更新](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L1844-L1867)、[输出处理](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L2737-L2756) | [调度器原文](evidence/upstream/vllm-project/vllm/vllm/v1/core/sched/scheduler.py)、[源码笔记](evidence/source-reading.md) | 区分调度进度与完成反馈，也区分不交付与不执行采样计算 |
| S5 | 分块不能代替 KV 准入与容量检查 | [KVCacheManager](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/kv_cache_manager.py#L517-L547) | [原文](evidence/upstream/vllm-project/vllm/vllm/v1/core/kv_cache_manager.py) | 当前完整输入 fit 检查不等于未来最大生成量预留 |
| D1 | PyTorch 矩形 SDPA 的 `is_causal=True` 有具体对齐语义 | [PyTorch 2.9 官方 API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) 的 causal 与 dropout 描述 | [API 笔记](evidence/source-reading.md)、[手算与真实反例测试](code/test_answers.py) | 本地执行 2.9.1；不把 PyTorch API 与引擎后端混同 |
| E1 | 正确分块与完整方阵 Attention 在记录精度下相符 | 本篇合成输入、独立完整 SDPA 与手算均值对照 | [答案](code/answers.py)、[原始运行](evidence/runs/lab.json)、[测试](evidence/runs/tests.txt) | 一层单头、CPU、常规尺度，不是完整模型等价或 GPU 性能 |
| E2 | 预算变化同时影响旧请求间隔、新请求首字和轮数 | 明确给定的 `1+0.5*d+p` 成本模型 | [模拟器](code/answers.py)、[轨迹](evidence/runs/lab.json) | 人为模型，不是实际毫秒测量或资源复用收益 |

## 源码版本与归档

### vLLM

- 官方远端：`https://github.com/vllm-project/vllm.git`；本次核验默认分支 main。
- 默认分支完整 SHA：`e49ea84e929e3d48a2bc500454dfcd4377f9a722`；提交时间 `2026-10-29T21:53:35-05:00`。
- 实际 fetch／checkout 命令及退出码见 [versions.json](evidence/versions.json)，缓存位于忽略目录 `.cache/upstream/vllm/`。
- 稳定发行元数据：[v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，发布于 `2026-10-18T06:50:48Z`，tag commit 为 `5df63a7203cdb9328d824a27b682e77c7b9928be`。只核验发行元数据，未独立分析该稳定版源码。
- 保留 LICENSE、SchedulerConfig、Scheduler、SchedulerOutput、KVCacheManager、EngineCore、GPUWorker、MRV1 GPUModelRunner、FlashAttention 后端及优化文档，共 10 个原始文件。路径、读取命令与校验和见 [source-manifest.json](evidence/source-manifest.json)。
- 浮动的 [当前优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/) 仅作资料定位；正文行为以固定版本实现为准。[对应固定快照文档](evidence/upstream/vllm-project/vllm/docs/configuration/optimization.md) 已归档，不能只凭文档中的优先级概述推断 running 列表全是 Decode。

### Sarathi-Serve

- 官方远端：`https://github.com/microsoft/sarathi-serve.git`；本次核验默认分支 main。
- 完整 SHA：`96f9911790ecc00af12ee9fae47cb8fa9ba0d199`；提交时间 `2026-01-08T10:40:57+05:30`。
- 当天实际 clone，记录见 [versions.json](evidence/versions.json)，缓存位于 `.cache/upstream/sarathi-serve/`。
- 保留 LICENSE、`sarathi_scheduler.py`、`config.py` 三个原始文件；不是论文年代完整实验环境。

两项目源码均按各自 Apache-2.0 许可保留，包含原始版权头与 [vLLM LICENSE](evidence/upstream/vllm-project/vllm/LICENSE)、[Sarathi-Serve LICENSE](evidence/upstream/microsoft/sarathi-serve/LICENSE)。归档保留上游原字节，不把原始源码称为本篇原创；本篇原创图源为 [时间线](assets/scheduling.mmd) 与 [跨块 KV 图](assets/prefix-kv.mmd)。

## 论文与官方 API

两篇论文完整题名、作者、原始日期、版本、DOI／arXiv ID、官方链接、实际阅读章节、前提及代码对应关系均在 [论文笔记](evidence/paper-reading.md)。论文未作为完整文件再分发；下载超时与成功替代阅读方式见 [paper-acquisition.json](evidence/paper-acquisition.json)，没有使用不完整 PDF 作证据。

API 阅读与具体反例在 [源码及 API 笔记](evidence/source-reading.md)。本篇只用一手来源支撑核心结论，不依赖二手性能数字。

## 可复现性与审查

- [代码目录](code/README.md)：两道独立题目、可填写接口、完整答案、测试与实验入口。
- [运行摘要](evidence/runs/summary.json)：实际 Python／PyTorch／平台、命令、工作目录、退出码和原始记录。
- [测试](evidence/runs/tests.txt)、[实验 JSON](evidence/runs/lab.json)、[Ruff check](evidence/runs/lint.txt)、[Ruff format](evidence/runs/format.txt)。
- [Python 独立审查与修复](evidence/python-review.md)、[技术及教学两轮自审](review.md)。
- [源文件清单](evidence/source-manifest.json)、[核心文件 SHA-256](evidence/SHA256SUMS)、[归档检查](evidence/runs/archive-check.txt)、[校验和复核](evidence/runs/checksums.txt)、[交付检查](evidence/runs/delivery-check.txt)。

没有运行引擎、GPU kernel、完整模型或真实负载服务测试。正文的精度数字来自 CPU；时间线数字来自假设公式。GitHub 提交链接在任务完成通知中给出，避免文章内自引用尚未生成的提交。
