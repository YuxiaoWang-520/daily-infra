# 参考资料与证据

主题：长序列 CP 的 K／V 交换、全局归一化、因果布局与梯度归约。检索日期：`2026-11-19`，时区 `Asia/Shanghai`。源码更新完成时间为 `2026-11-19T11:06:20.160253+08:00`；页面读取和笔记整理在同日完成。

## 关键论断与来源

| 编号 | 论断 | 一手来源与精确定位 | 本地归档 | 范围与限制 |
| --- | --- | --- | --- | --- |
| S1 | CP 按序列切分，Attention 仍依赖全局 K／V；与 Megatron 语境的 SP 不同 | [官方 CP 概览](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)，Overview 与 Enabling Context Parallelism | [官方文档阅读笔记](evidence/reading-notes.md) | 浮动文档只支持概念；当日实现以固定源码为准。 |
| S2 | 块式 Attention 可保持 Q 本地，并旋转 K／V；通信隐藏有计算／带宽条件 | [Ring Attention v4](https://arxiv.org/html/2310.01889v4)，第 3 节、Algorithm 1、附录 A／Figure 4 | [论文笔记](evidence/paper-notes.md) | 原始 JAX 实现与当前 TE 不是同一套代码；没有复现论文性能。 |
| S3 | CP 组、stream、通信类型传给 TE | [Megatron 包装器](https://github.com/NVIDIA/Megatron-LM/blob/43a2ef6d35a0188a4f0601a5ce5e569d2c011562/megatron/core/extensions/transformer_engine.py#L2439-L2470) | [组与默认配置](evidence/upstream/Megatron-LM/cp-group-and-default.txt)、[forward 委托](evidence/upstream/Megatron-LM/delegate-qkv-to-te.txt) | `context_parallel_size>1`，受 TE 版本等条件限制。 |
| S4 | p2p／AllGather／A2A／层次式是不同路线 | [Megatron 配置](https://github.com/NVIDIA/Megatron-LM/blob/43a2ef6d35a0188a4f0601a5ce5e569d2c011562/megatron/core/transformer/transformer_config.py#L1309-L1332)、[TE 分派](https://github.com/NVIDIA/TransformerEngine/blob/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L5746-L5794) | [配置](evidence/upstream/Megatron-LM/cp-communication-and-layout.txt)、[分派](evidence/upstream/TransformerEngine/cp-dispatch.txt) | 本篇深入普通 softmax 的 P2P 路线，不保证所有 head、mask 或 backend 组合可用。 |
| S5 | zigzag 采用 `2p` 块的首尾配对 | [Megatron 契约与索引](https://github.com/NVIDIA/Megatron-LM/blob/43a2ef6d35a0188a4f0601a5ce5e569d2c011562/megatron/core/utils.py#L2644-L2702) | [契约](evidence/upstream/Megatron-LM/zigzag-contract.txt)、[索引](evidence/upstream/Megatron-LM/zigzag-chunk-selection.txt) | 等长完整因果序列下的负载直觉；packed 文档等需另外核对。 |
| S6 | P2P 前向使用通信缓冲与增量结果合并 | [TE 缓冲及循环](https://github.com/NVIDIA/TransformerEngine/blob/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L1913-L1949) | [前向缓冲](evidence/upstream/TransformerEngine/forward-ring-buffers.txt)、[P2P 操作](evidence/upstream/TransformerEngine/p2p-communication.txt)、[因果 tile 选择](evidence/upstream/TransformerEngine/causal-tile-selection.txt) | GPU 路线只读源码；CPU P2P 演示没有重叠计算。 |
| S7 | 用全局 LSE 重新缩放局部输出 | [TE LSE 与输出校正](https://github.com/NVIDIA/TransformerEngine/blob/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L156-L193) | [LSE](evidence/upstream/TransformerEngine/merge-logsumexp.txt)、[输出](evidence/upstream/TransformerEngine/merge-partial-output.txt)、[实际调用](evidence/upstream/TransformerEngine/apply-softmax-correction.txt) | 教学代码使用 `(m,ℓ,u)`；全遮蔽通用处理不能直接等同于 TE 子函数契约。 |
| S8 | K／V 梯度要汇总跨 query owner 的贡献 | [TE 反向通信](https://github.com/NVIDIA/TransformerEngine/blob/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L2700-L2740)、[梯度修正](https://github.com/NVIDIA/TransformerEngine/blob/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L3017-L3051) | [通信](evidence/upstream/TransformerEngine/backward-ring-communication.txt)、[累加](evidence/upstream/TransformerEngine/backward-kv-accumulation.txt) | 普通与 FP8 分支不同；本机没有运行 TE 分布式反向。 |
| S9 | backend、格式及滑动窗口限制影响分派 | [TE 条件检查](https://github.com/NVIDIA/TransformerEngine/blob/c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py#L5692-L5716) | [条件片段](evidence/upstream/TransformerEngine/backend-and-window-constraints.txt) | 只描述指定 main 快照，不外推所有稳定版。 |
| S10 | 可微 AllGather 的反向对 owner 求和；Gloo 走模拟路径 | [PyTorch `_AllGather.backward`](https://github.com/pytorch/pytorch/blob/55e45fb47412a2ffdab02ca0ee62da51c1d58f31/torch/distributed/nn/functional.py#L440-L467)、[`_AlltoAll.forward`](https://github.com/pytorch/pytorch/blob/55e45fb47412a2ffdab02ca0ee62da51c1d58f31/torch/distributed/nn/functional.py#L499-L522) | [上游反向](evidence/upstream/pytorch/autograd-gather-backward.txt)、[Gloo fallback](evidence/upstream/pytorch/gloo-scatter-fallback.txt)、[安装包实际片段](evidence/upstream/pytorch/runtime-all-gather.txt) | 该教学基线使用的旧接口已经弃用；警告和实际环境均保留。 |
| S11 | SDPA 的 Boolean mask 中 True 表示允许，默认按 `1/√D` 缩放 | [PyTorch SDPA 语义](https://github.com/pytorch/pytorch/blob/55e45fb47412a2ffdab02ca0ee62da51c1d58f31/torch/nn/functional.py#L6470-L6526) | [语义片段](evidence/upstream/pytorch/sdpa-semantics.txt)、[测试](code/test_answers.py) | 正确性对照实际运行安装包 CPU SDPA，不声称执行了抓取的 main。 |

## 源码版本与获取

| 官方仓库 | 默认分支及完整 SHA | 最新稳定 release／tag 的完整 SHA |
| --- | --- | --- |
| `https://github.com/NVIDIA/Megatron-LM.git` | `main`：`43a2ef6d35a0188a4f0601a5ce5e569d2c011562` | `core_v0.20.0`：`685fb8bd970185e9ac3283fe32a5d6667bf4b268` |
| `https://github.com/NVIDIA/TransformerEngine.git` | `main`：`c2b5a322fc0220770171b9dc6b9b922cb3ddcbdd` | `v2.21.0`：`7457e47d26d9d02dfb783007335dfbe75e1b90a8` |
| `https://github.com/pytorch/pytorch.git` | `main`：`55e45fb47412a2ffdab02ca0ee62da51c1d58f31` | `v2.15.0`：`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457` |

已核验上述官方仓库的默认分支为 main。缓存置于 `.cache/upstream/`，实际获取过程见 [source-update.json](evidence/source-update.json)。PyTorch 与 Megatron 的既有缓存得到保留；Megatron 原有 no-checkout 状态没有被 checkout、reset 或清理。Transformer Engine 使用 `--filter=blob:none --no-checkout --depth=1` 新建缓存，之后按固定 SHA 用 `git show` 阅读。

本次 fetch 的大量无关自动 tag 通告不全文归档；更新记录注明删节范围、原 stderr 行数与 SHA256，保留相关 main／稳定 tag 信息。它不是完整原始控制台转录。

稳定 tag 对照只覆盖本篇关心的 CP 配置、P2P 分派和 Softmax 合并；对应 `stable-*` 片段也在清单中。没有对稳定版进行 GPU 性能、全功能兼容或完整 diff 验收。

[source-manifest.json](evidence/source-manifest.json)逐片段记录源路径、行号、完整源文件哈希、片段哈希与永久链接。共 `24` 个固定版本片段，另有安装包实际 `_AllGather` 片段。各仓库 LICENSE 与引用文件的原版权头存于 `evidence/upstream/<repository>/`。片段保持原字节；本篇 `.gitattributes` 仅豁免这些上游原文和运行原始文本的空白检查。

## 论文与官方文档

- **论文**：Hao Liu、Matei Zaharia、Pieter Abbeel，*Ring Attention with Blockwise Transformers for Near-Infinite Context*，arXiv `2310.01889`。v1 为 `2023-10-03`，本篇主要阅读 v4（`2023-11-27`）的第 3 节、Algorithm 1 和附录 A／Figure 4。版本核对后不以先前定位的 v3 代替 v4。链接：[摘要与版本历史](https://arxiv.org/abs/2310.01889v4)、[v4 HTML](https://arxiv.org/html/2310.01889v4)。自己的符号、推导与差异记录见 [paper-notes.md](evidence/paper-notes.md)。
- **再分发边界**：arXiv 页面标示其永久非独占分发许可，并不等于向第三方授予完整再分发权。本仓库不打包完整论文 HTML／PDF，保留自己的笔记与精确版本定位；未复制论文插图。
- **官方文档**：Megatron Core 的 [Context Parallel Overview](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)。通过 Parallel 提取读取，页面获取快照日期为 `2026-11-18`；论文页面工具缓存的获取日期早于本次访问。检索与页面快照日期区分见 [web-retrieval.json](evidence/web-retrieval.json)。实现事实由同日 fetch 的固定源码支撑。
- **Release**：[Megatron core_v0.20.0](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0)、[TE v2.21.0](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.21.0)、[PyTorch v2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)。通过 GitHub API 查当日 latest release，发布日期和稳定 commit 在源码记录中。

## 可复现证据与验证边界

- 两道完整练习、六个 Python 文件：[code/](code/README.md)。数值均为公开机制的自编合成输入，没有内部训练数据、日志、路径或项目标识。
- [运行环境、命令、退出码与六文件哈希](evidence/runs/verification.json)：现有 Python `3.12.16`、PyTorch `2.15.0`（build `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`）、Darwin arm64、CUDA 不可用、Gloo 可用，未安装或升级依赖。
- [首次测试失败](evidence/runs/01-red-tests.txt)与[初始接口记录](evidence/runs/01-red.json)：先有测试、后实现。随后新增的空 key／分数溢出检查曾[真实失败](evidence/runs/03-edge-red-tests.txt)，修复后并入 `26` 项通过测试。
- [正常测试](evidence/runs/04-tests.txt)、[优化模式测试](evidence/runs/05-optimized-tests.txt)、[Ruff](evidence/runs/06-ruff.txt)、[格式检查](evidence/runs/07-format.txt)、[数值答案与假设字节预算](evidence/runs/08-answers.json)。
- [真实双进程输出](evidence/runs/09-distributed.txt)、[四条解析结果](evidence/runs/09-distributed.json)、[启动参数与退出码](evidence/runs/09-distributed-metadata.json)。原始日志仅替换本机绝对工作路径和机器名，其余输出包括弃用警告保留。原始 `torchrun` 尝试的输出未被初始包装器保存，该缺失明确记录在[失败记录](evidence/runs/02-distributed-probe.json)。
- [Python 代码审查记录](evidence/python-review.json)：固定模型专用审查器启动失败；通用独立审查者实际审查六文件并通过 `7` 项针对性测试，不冒称专用 Python 审查器成功执行。
- [图源](figures/ring.mmd)与正文 Mermaid 一致，为自编依赖示意图，没有测得的时间轴或上游图像复制。技术及教学自审见 [review.md](review.md)，最终文件校验见 [SHA256SUMS](evidence/SHA256SUMS)和[交付检查](evidence/delivery-check.json)。
