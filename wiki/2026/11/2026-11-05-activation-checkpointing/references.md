# 参考资料与证据

主题：用重算换显存的代价如何计算？检索与核验日期：2026-11-05（Asia/Shanghai）；具体时间见 [versions.json](evidence/versions.json)。

## 关键论断与来源

| 编号 | 需要证实的论断 | 一手来源与精确定位 | 本地归档 | 证据范围与限制 |
| --- | --- | --- | --- | --- |
| S1 | 保存边界、反向重建内部数据可以缩短中间结果生命周期 | Chen 等人 v2，§4.1、Algorithm 1，第 4—5 页 | [论文笔记](evidence/paper-reading.md) | 对照 PyTorch 的按需重算机制；不声称两者是同一份实现。 |
| S2 | 均匀分段有边界与局部工作集两项成本 | 同论文 §4.3、公式（1），第 5 页 | 论文笔记与 [容量答案](code/answers.py) | 本篇 `ceil(n/s)+s` 是独立教学代理，不是实际总显存或论文完整规划器。 |
| S3 | 非重入 checkpoint 记录图，通过保存／重取 hooks 触发重算 | PyTorch main 的 `checkpoint`、`_checkpoint_impl`、`_checkpoint_without_reentrant_generator_impl`、`_checkpoint_hook` | [main checkpoint.py](evidence/upstream/pytorch/main/torch/utils/checkpoint.py) | 代码固定 commit；main 未构建或执行。 |
| S4 | 非重入路径默认可提前结束重算 | `_recomputation_hook` 的计数条件与 `_StopRecomputationError`；`early_stop=True` 参数 | 同上，1288—1334 行、437—449 行 | 入口计数只能表明进入了 block，不代表全部算子完整重跑。 |
| S5 | 保存 RNG 能使本例 Dropout 的重算条件保持一致 | `_checkpoint_without_reentrant_generator_impl` 保存 CPU 状态，`recompute_fn` 在 `fork_rng` 中恢复 | 同上，1975—2047 行；[CPU 实验](code/torch_probe.py) | 实测只覆盖 CPU eager；不推断多设备、任意 Generator 或编译模式。 |
| S6 | 默认 determinism check 不逐元素比较值 | `_default_meta_extractor`、`_CheckpointFrame.check_recomputed_tensors_match` | 同上，1244—1256 行、1029—1101 行 | 还检查保存数量；不能当作完整训练数值验收。 |
| S7 | 用户 hooks 在嵌套时仅观察最内层生效的保存 | `torch.autograd.graph.saved_tensors_hooks` 的嵌套和引用环说明 | [main graph.py](evidence/upstream/pytorch/main/torch/autograd/graph.py) | 本例外层统计不包含 checkpoint 全部内部成本，不能代替峰值测量。 |
| S8 | 本次保存 RNG 时梯度和 SGD 更新匹配，关闭时可出现反例 | 本机 PyTorch 2.15.0 的固定 CPU 实验与断言 | [lab.json](evidence/runs/lab.json)、[tests.txt](evidence/runs/tests.txt) | 正确性实验；未测 GPU 显存、吞吐或长期收敛。 |
| S9 | 当前 main 与 v2.15.0 的部分设备／SAC 行为已有差异 | 实际 `git diff v2.15.0 HEAD -- torch/utils/checkpoint.py` | [完整差异](evidence/upstream/main-v2.15.0.diff) | 本文不使用 SAC，不将其 hook 统计推广到其他模式。 |

## 源码记录

- 官方仓库：[pytorch/pytorch](https://github.com/pytorch/pytorch)，远端 `https://github.com/pytorch/pytorch.git`，默认分支 `main` 由当日 GitHub API 核实。
- 当日 main：`4892050dadb28d1ca1065268e35084747967e34e`。实际执行 shallow、blobless、sparse clone，随后按相关文件设置 sparse checkout；命令与退出码见 [versions.json](evidence/versions.json)。
- 稳定版本：当日 GitHub `releases/latest` 返回 [v2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，发布时间为 `2026-10-28T17:09:49Z`。实际 fetch 该 tag，固定 commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。
- 本机实测：Python 3.12.16，PyTorch 2.15.0，构建 SHA `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`；构建 SHA 与 tag SHA 分开记录。本机 `checkpoint.py` 与 tag 的同名文件逐字节相同，SHA-256 为 `73e1f61d3134fe67a00cdf466ed4db911b3806e0b4f3cbe3500e4376314019b1`。未核对完整二进制包与 tag 的一致性。
- 运行路径：[源码笔记](evidence/source-reading.md)。官方固定链接：[main checkpoint.py](https://github.com/pytorch/pytorch/blob/4892050dadb28d1ca1065268e35084747967e34e/torch/utils/checkpoint.py)、[main graph.py](https://github.com/pytorch/pytorch/blob/4892050dadb28d1ca1065268e35084747967e34e/torch/autograd/graph.py)、[v2.15.0 checkpoint.py](https://github.com/pytorch/pytorch/blob/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457/torch/utils/checkpoint.py)。
- 归档相关完整源文件和 [上游 LICENSE](evidence/upstream/pytorch/LICENSE)，保留原始空白；逐字节校验记录在 [source-manifest.json](evidence/source-manifest.json)。缓存 `.cache/upstream/pytorch/` 不是发布证据的唯一位置。

## 论文记录与阅读笔记

- 题名：Training Deep Nets with Sublinear Memory Cost。
- 作者：Tianqi Chen、Bing Xu、Chiyuan Zhang、Carlos Guestrin。
- 原始提交：2016-04-21；实际阅读版本：2016-04-22 的 v2，`arXiv:1604.06174v2`；DOI：`10.48550/arXiv.1604.06174`。
- 来源：[版本页](https://arxiv.org/abs/1604.06174v2)、[PDF 原文](https://arxiv.org/pdf/1604.06174v2)。实际下载成功，PDF 为 608189 bytes，SHA-256 和获取记录见 [paper-acquisition.json](evidence/paper-acquisition.json)。
- 实际核验：§4.1、Algorithm 1、§4.3 公式（1），以及 §5.1 对参数和临时内存的排除说明。用现有 Poppler 渲染并查看第 5 页，核对分段循环与公式，未安装依赖。
- [阅读笔记](evidence/paper-reading.md) 解释符号、假设及与 PyTorch 的对应和差别。没有移用论文中的历史 GPU 性能数字作为当前训练结论。
- 页面标注 [arXiv 非独占分发许可](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)，不能据此推断一般再分发授权。原 PDF 与渲染页仅作 `.cache/papers/` 本地阅读缓存，不进入 Git；本篇归档自写笔记与可定位、可校验的来源元数据。

## 官方文档与其他资料

[PyTorch 2.15 checkpoint 文档](https://docs.pytorch.org/docs/2.15/checkpoint.html)，2026-11-05 实际访问，用来定位 API 及非重入路径说明。实现结论以固定 commit 的代码和本地归档为准，避免浮动文档替代版本证据。未使用二手文章支撑核心结论。

## 可复现证据

- 练习、输入与验收：[code/README.md](code/README.md)；完整答案：[容量模型](code/answers.py)、[CPU 实验](code/torch_probe.py)；组合运行：[run_lab.py](code/run_lab.py)。
- CPU：Mac mini，Apple M6，arm64，macOS 27.0；解释器为 `/Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python`。实验显式使用 CPU FP64，JSON 运行设置 1 个 CPU 线程。
- 基础测试与实验输出：[tests.txt](evidence/runs/tests.txt)、[lab.json](evidence/runs/lab.json)；静态检查：[lint.txt](evidence/runs/lint.txt)、[format.txt](evidence/runs/format.txt)。最终命令、退出码与摘要见 [summary.json](evidence/runs/summary.json)。
- 先红后绿：[容量模板的初始失败](evidence/runs/initial-red-tests.txt)、[实验模板的初始失败](evidence/runs/torch-red-tests.txt)、[时间合成溢出复现](evidence/runs/overflow-red-test.txt)、[独立审查发现的边界复现](evidence/runs/reviewer-red-test.txt)。这些失败记录保留原路径和当时行号，最终源码行号可以不同。
- [Python 独立审查](evidence/python-review.md)、[技术与教学审查](review.md)。源码、代码、图源及关键元数据的校验和见 [SHA256SUMS](evidence/SHA256SUMS)。
- 没有安装或升级依赖；没有 GPU、编译、分布式或长期收敛结果。所有微秒数是容量练习的假设输入。
