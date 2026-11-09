# 参考资料与证据

本篇在 2026-11-09 实际检索、阅读原始论文及官方资料，并从官方远端更新 PyTorch main。检索过程见 [research-log.json](evidence/research-log.json)，版本关系见 [versions.json](evidence/versions.json)。

## 关键论断与定位

| 编号 | 论断 | 原始来源与定位 | 本地证据 | 边界 |
| --- | --- | --- | --- | --- |
| P1 | 三阶段累加分片三类模型状态 | ZeRO v3 第 3.1、5.1—5.3 节，图 1 | [论文笔记](evidence/paper-reading.md) | 2／2／12 B 为论文式混合精度假设 |
| P2 | 分片参数可按需收集，梯度归约到所有者 | ZeRO v3 第 4.1、5.2—5.3 节 | [论文与实现对应](evidence/paper-reading.md) | 当前执行细节另由 FSDP2 源码核验 |
| P3 | 两次参数收集会增加通信 | ZeRO v3 第 7.1—7.2 节 | [推导与题解](code/README.md) | 本文按理想 ring 的每 rank 发送字节展开 N 与 dtype |
| S1 | fully_shard 配置参数组，默认自动策略区分 root | `_fully_shard.py` 100—137、260—325 行，`_fsdp_state.py` 179—232 行 | [入口片段](evidence/upstream/pytorch/main/_fully_shard.py.txt)、[状态片段](evidence/upstream/pytorch/main/_fsdp_state.py.txt) | 当日 main 与稳定 tag 分别核验 |
| S2 | 参数按 Tensor 维度分片并处理 padding | `_fsdp_param.py` 279—408 行 | [分片片段](evidence/upstream/pytorch/main/_fsdp_param.py.txt) | 练习的 flat 均匀补齐是教学布局 |
| S3 | 前向／反向 hooks 触发收集、重分片和归约 | `_fsdp_param_group.py` 450—495、580—592、627—724、788—835 行 | [参数组片段](evidence/upstream/pytorch/main/_fsdp_param_group.py.txt) | 路径依策略和设备而定 |
| S4 | collective 与缩放共同实现梯度同步语义 | `_fsdp_collectives.py` 342—400、648—746、1000—1064 行 | [集合通信片段](evidence/upstream/pytorch/main/_fsdp_collectives.py.txt) | 默认实现、dtype 与 divide factor 条件已区分 |
| S5 | 参数／归约混合精度会影响字节预算 | `_fsdp_api.py::MixedPrecisionPolicy` | [策略片段](evidence/upstream/pytorch/main/_fsdp_api.py.txt) | 本文没有实测 GPU 混合精度 |
| S6 | Adam m、v 与偏差修正 | 官方 Adam 2.15 算法说明，`torch/optim/adam.py` 对应更新片段 | [Adam 源码片段](evidence/upstream/pytorch/main/adam.py.txt) | 实验使用 CPU FP64，无 weight decay／AMSGrad |
| E1 | 分片更新保持全局 Adam 目标 | CPU／Gloo 两进程 3 步与负对照 | [lab.json](evidence/runs/lab.json) | 显式教学 collective 与本地优化器 |

当前实现永久链接包括 [fully_shard](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fully_shard.py)、[state](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_state.py)、[param](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param.py)、[param group](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py) 和 [collectives](https://github.com/pytorch/pytorch/blob/cbee6335ae15a556794e2552c553f5befda84516/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py)。

## 源码获取与再分发

官方远端为 `https://github.com/pytorch/pytorch.git`，GitHub 仓库 API 核验默认分支 main。干净的既有缓存执行 `git fetch --depth 1 origin main`，退出码 0，固定提交 `cbee6335ae15a556794e2552c553f5befda84516`。读取均使用固定 SHA 的 git show，缓存 checkout 保持原状。

稳定 latest release API 返回 v2.15.0，发布时间 `2026-10-28T17:09:49Z`，tag 指向 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。main 读取、稳定 tag 与安装包构建分开记录。根模块自动策略在稳定 tag 中的原始片段也已保存。[稳定发布](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)

本篇归档 9 份带原始行号的源码摘录和一份完整 LICENSE。每份摘录记录原路径、完整 SHA、行范围、完整 Git blob SHA-256 与摘录 SHA-256。可按清单重建原始选段，分隔空行和 L 行号前缀为归档格式。文件来源见 [source-manifest.json](evidence/source-manifest.json)，许可见 [LICENSE](evidence/upstream/pytorch/LICENSE)。

本机安装包报告 2.15.0，构建 Git SHA 与发布 tag 分别列出。选取 FSDP 入口和 state 两个安装文件做字节比较，二者均匹配稳定 tag，比较范围写在清单中。具体实现阅读见 [源码笔记](evidence/source-reading.md)。

## 论文与官方文档

ZeRO 原始论文为 Samyam Rajbhandari、Jeff Rasley、Olatunji Ruwase、Yuxiong He 合著，首发于 2019-10-04，本篇阅读 v3（2020-05-13），arXiv:1910.02054v3，DOI 为 10.48550/arXiv.1910.02054。完整章节、符号与实现对应见 [论文笔记](evidence/paper-reading.md)。原文入口列出 arXiv 非独占分发许可，本篇归档自撰笔记及精确版本定位。[版本入口](https://arxiv.org/abs/1910.02054v3)、[许可说明](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)

官方稳定资料实际阅读了 FSDP2 的 API 描述、mesh、reshard_after_forward 与生命周期，以及 Adam 的更新算法、超参数和状态。默认分支文档页面本次没有可读正文，当前实现直接使用当日获取的固定 Git 源码。阅读笔记与来源定位见 [documentation-reading.md](evidence/documentation-reading.md)。[FSDP2 2.15](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)、[Adam 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)

## 可复现交付

题目与答案见 [code/README.md](code/README.md)，真实命令、退出码、环境与源码哈希见 [runs/summary.json](evidence/runs/summary.json)，独立审查见 [python-review.md](evidence/python-review.md)。

Mermaid 图源由本文自绘，表示指定策略下的机制。完整性清单见 [SHA256SUMS](evidence/SHA256SUMS)，机器交付检查见 [delivery-check.json](evidence/delivery-check.json)，两轮内容审查见 [review.md](review.md)。
