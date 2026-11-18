# 参考资料与源码证据

核验日期为 `2026-11-18`，时区为 `Asia/Shanghai`。核心问题是 PP 中 microbatch 的前后向次序、待反向状态窗口以及 batch／通信取舍。原始论文、固定源码、公式模型与本机实际执行分别记录。

## 1. 源码获取与版本

| 官方仓库 | 实际获取动作 | 固定默认分支 commit | 核验时最新 release |
| --- | --- | --- | --- |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | 现有过滤缓存 fetch `origin main` | `6e95b054945729c0edc7ecf9227386e1b63e45d0` | `v2.15.0`，发布于 `2026-10-28T17:09:49Z` |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 现有过滤缓存 fetch `origin main` | `91260b59e815f0be44d625d75ba22cb8473e5981` | `core_v0.20.0`，发布于 `2026-10-29T21:06:49Z` |

通过 GitHub REST API 分别核验默认分支为 `main`，并查询 `releases/latest`。更新前查看缓存状态，保留原有工作树和索引；Megatron 缓存最初以 `--no-checkout` 建立，本次仍只 fetch 与按 SHA 读取，未 checkout、reset 或清理。

```bash
git -C .cache/upstream/pytorch fetch --filter=blob:none origin main
git -C .cache/upstream/Megatron-LM fetch --filter=blob:none origin main
```

随后用 `git show <完整 SHA>:<路径>` 读取。共保存 `20` 段、`566` 行实现证据，包含原路径、行号、永久链接、完整源文件和片段 SHA256，以及许可与文件头的摘要。[源码清单](evidence/source-manifest.json)

实际 CPU 执行环境为安装的 PyTorch `2.15.0`，build commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`，并非从上述 main 快照重新构建。另只读核对了本机 `Schedule1F1B.__init__`，同样要求 microbatch 数不少于 stage 数。本文未执行 Megatron GPU 训练。

## 2. 核心论断与实际读取位置

| 问题 | 固定源码定位 | 本地证据与读取结论 |
| --- | --- | --- |
| GPipe 是否先全部前向再全部反向？ | PyTorch `torch/distributed/pipelining/schedules.py:1399–1418、1430–1464` | [前向](evidence/upstream/pytorch/gpipe-forward.txt)、[反向与缩放](evidence/upstream/pytorch/gpipe-backward-scaling.txt)，当前反向按编号递增 |
| 1F1B 接口有哪些前提？ | 同文件 `1505–1528` | [构造条件](evidence/upstream/pytorch/one-f-one-b-constraint.txt)，当前类要求 `m≥p` |
| 预热为什么比另一种划分多一份？ | 同文件 `1554–1580、1610–1645、1668–1698` | [预热](evidence/upstream/pytorch/one-f-one-b-warmup.txt)、[稳态](evidence/upstream/pytorch/one-f-one-b-steady.txt)、[排空](evidence/upstream/pytorch/one-f-one-b-cooldown.txt)，先做 `p−r` 次前向，再反向／前向交替 |
| 保存的状态在哪里取出？ | PyTorch `torch/distributed/pipelining/stage.py:1203–1216、1261–1279` | [保存](evidence/upstream/pytorch/save-forward-state.txt)、[消费](evidence/upstream/pytorch/consume-forward-state.txt)，按 chunk 身份匹配，最后一段返回输出还有独立保存路径 |
| 默认梯度缩放的数学作用？ | 同文件 `1051–1065` | [缩放](evidence/upstream/pytorch/gradient-scaling.txt)，配合调度调用者给定因子，需匹配 loss 的分母 |
| 静态元数据为什么会产生 `None` 梯度？ | 同文件 `2377–2399`，`_utils.py:331–353` | [静态分支](evidence/upstream/pytorch/static-gradient-metadata.txt)、[派生规则](evidence/upstream/pytorch/derive-gradient-metadata.txt)，无梯度标志的源元数据对应 `None` |
| stage 如何生成 P2P 发送？ | `stage.py:752–806、834–894` | [前向发送](evidence/upstream/pytorch/forward-p2p-ops.txt)、[梯度发送](evidence/upstream/pytorch/backward-p2p-ops.txt)，同时关注发送完成与状态释放的边界 |
| Megatron 普通非交错路径怎样选？ | `megatron/core/pipeline_parallel/schedules.py:162–169` | [路径选择](evidence/upstream/Megatron-LM/schedule-selection.txt)，限定普通 PP 与虚拟流水线配置 |
| 预热窗口与保存队列如何对应？ | 同文件 `2301–2316、2350–2396` | [计数](evidence/upstream/Megatron-LM/warmup-count.txt)、[队列](evidence/upstream/Megatron-LM/warmup-save-queue.txt)，预热 `p−r−1` 份，稳态再前向后反向 |
| 稳态和排空分别处理哪些对象？ | 同文件 `2449–2485、2487–2511` | [稳态](evidence/upstream/Megatron-LM/steady-forward-backward-queue.txt)、[排空](evidence/upstream/Megatron-LM/cooldown.txt)，队尾保存、队首取出并匹配梯度 |
| 组合收发最终向哪个方向通信？ | `megatron/core/pipeline_parallel/p2p_communication.py:521–545、554–578` | [前向发送／反向接收](evidence/upstream/Megatron-LM/send-forward-receive-backward.txt)、[反向发送／前向接收](evidence/upstream/Megatron-LM/send-backward-receive-forward.txt)，按相邻 stage 对应 |

所有范围的 GitHub 固定 commit URL 均在源码清单中，正文在相关论断附近直接引用。不同后端、DTensor、虚拟 stage、多模块及其他并行配置有额外条件，本文只分析明确列出的路径。

## 3. 原始论文和官方文档

| 一手来源 | 版本、日期与实际阅读位置 | 在本文中的用途 |
| --- | --- | --- |
| [GPipe](https://arxiv.org/html/1811.06965v5) | `1811.06965v5`，`2019-07-25`；阅读 §2.1—2.3 与图 2 | 分片、microbatch 累积、统一更新及重算的范围 |
| [Efficient Large-Scale Language Model Training](https://arxiv.org/html/2104.04473v5) | `2104.04473v5`，`2021-08-23`；阅读 §2.2.1—2.2.2、图 3—4，§3.1、3.2、3.4 | 普通同步 1F1B 的保存窗口、气泡定义与 microbatch 取舍 |
| [PyTorch Pipeline Parallelism](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html) | 显式 `2.15` 文档，访问于 `2026-11-18` | `PipelineStage`、`ScheduleGPipe`、`Schedule1F1B`、`PipelineScheduleSingle` 与 `scale_grads` 的 API 约定 |

官方文档标记该 pipelining API 为 alpha，本文按固定版本与真实运行限定结论。当前实现支持静态和运行时推导的元数据；本次只实际验证显式静态配置，没有据此推定动态 shape 行为。

作者、最初提交日期、实际版本、阅读笔记，以及 GPipe 摘要页与 HTML 版内题名差异见 [原始论文笔记](evidence/paper-notes.md)。使用 HTML 原文，未读取或归档 PDF。

## 4. 自编验证与边界

| 记录 | 实际结果与范围 |
| --- | --- |
| [初始测试](evidence/runs/01-red-verification.json) | `21` 项测试在参考接口尚未实现时运行，含子测试的预期错误，退出码 `1` |
| [元数据失败](evidence/runs/02-distributed-metadata-red.json) | 初始静态元数据未声明梯度标志，实际两进程反向失败，原始错误完整保留 |
| [修正后探测](evidence/runs/03-distributed-probe.txt) | 实际两进程运行通过，随后最终记录再验证冻结后的脚本 |
| [普通测试](evidence/runs/04-tests.txt)、[优化模式](evidence/runs/05-optimized-tests.txt) | 各 `21` 项通过，覆盖多组依赖、状态窗口、分母与参数更新 |
| [Ruff lint](evidence/runs/06-ruff.txt)、[format](evidence/runs/07-format.txt) | 已有工具的检查均通过 |
| [自编数值与预算](evidence/runs/08-answers.json) | 真实 CPU 模型输出、完整 batch 对照；时序、容量和通信预算明确采用教学假设 |
| [真实两 rank](evidence/runs/09-distributed.json) | 两种调度 × 两个 rank，检查输出、所有参数梯度、一次 SGD 更新和前后向调用次序 |
| [执行清单](evidence/runs/verification.json) | 实际命令、退出码、环境及六个 Python 文件 SHA256 |
| [独立代码复核](evidence/python-review.json) | 通用代理只读复核六文件并运行七项针对性测试，专用代理未启动的原因单列 |

日志写入前替换了绝对 home／仓库路径、机器名和网络地址。分布式进程的输出可能粘连，采集通过 `PP_RESULT` 标记及 JSON 解码边界提取。实际时间线表由模拟器结果生成，CPU 记录只有调用次序与数值，没有将其当作 GPU 耗时或显存 trace。

## 5. 许可、图源和交付检查

上游片段保留读取范围的原始字节，配套完整许可和所选文件头归档。[PyTorch LICENSE](evidence/upstream/pytorch/LICENSE)、[Megatron LICENSE](evidence/upstream/Megatron-LM/LICENSE)、[文件头](evidence/upstream/Megatron-LM/COPYRIGHT_HEADERS.txt)

数据依赖图、合成模型及单位时间表为自编。[Mermaid 图源](assets/dependencies.mmd)、[模拟器时间表](assets/timelines.txt)。图文一致性、源码与片段摘要、代码冻结摘要、本地链接、索引日期及脱敏检查由 [交付清单](evidence/delivery-check.json) 记录。指定原始日志与上游片段按局部 `.gitattributes` 保留空白，文件字节见 [SHA256SUMS](evidence/SHA256SUMS)。

返回 [正文](README.md) 或 [审查记录](review.md)。
