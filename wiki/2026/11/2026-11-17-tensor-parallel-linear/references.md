# 参考资料与源码证据

核验日期为 `2026-11-17`，时区为 `Asia/Shanghai`。主问题围绕两层 MLP 的特征切分、部分和及通信边界，原始论文、当前源码和实际执行分别记录。

## 1. 官方源码的实际获取

| 对象 | 官方远端 | 获取动作 | 固定默认分支 commit | 最新 release |
| --- | --- | --- | --- | --- |
| PyTorch | `https://github.com/pytorch/pytorch` | 在现有缓存 fetch `origin main` | `479e2b59bebcc8f65283548837332b1ad42f99c3` | `v2.15.0`，发布于 `2026-10-28T17:09:49Z` |
| Megatron-LM | `https://github.com/NVIDIA/Megatron-LM.git` | 新建过滤 blob 的浅层缓存，不 checkout | `1a82f514b87d6bd4cc81d1a175a6c4ce6e54faec` | `core_v0.20.0`，发布于 `2026-10-29T21:06:49Z` |

两者默认分支均为 `main`，release 信息通过 GitHub REST API 的 `releases/latest` 核验。缓存分别位于 `.cache/upstream/pytorch/` 和 `.cache/upstream/Megatron-LM/`。现有 PyTorch 缓存工作树保留原状，仅更新官方远端 ref，用固定 SHA 的 `git show` 读取内容。

```bash
git -C .cache/upstream/pytorch fetch --filter=blob:none origin main
git clone --filter=blob:none --no-checkout --depth=1 \
    https://github.com/NVIDIA/Megatron-LM.git .cache/upstream/Megatron-LM
```

实现分析对应上述 main 快照，稳定 release 作为当时的发布参照。实际 CPU 执行使用安装的 PyTorch `2.15.0`，build commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。这三类版本分别记录，本文没有运行缓存源码构建出的框架或 Megatron GPU 训练。

获取时间、官方远端、commit 时间、release 与原文行号、源文件及片段 SHA256 位于清单中。共归档 `16` 段、`612` 行实现证据。[源码与版本清单](evidence/source-manifest.json)

## 2. 关键论断和读取位置

| 需要证实的问题 | 固定源码的定位 | 本地证据与阅读结论 |
| --- | --- | --- |
| 列切分默认输入输出与存储维度？ | PyTorch `torch/distributed/tensor/parallel/style.py:86–152` | [Colwise](evidence/upstream/pytorch/colwise-style.txt)，输入为副本，输出默认切最后一维，Linear 存储 weight 按第 `0` 维切分 |
| 行切分参数及 bias 如何分布？ | 同文件 `221–279` | [Rowwise 参数](evidence/upstream/pytorch/rowwise-parameters.txt)，存储 weight 按第 `1` 维切分，bias 为副本 |
| 部分和怎样按输出布局转换？ | 同文件 `294–307` | [Rowwise 输出](evidence/upstream/pytorch/rowwise-output.txt)，副本对应 AllReduce，切片对应 ReduceScatter |
| 布局类型怎样指向 collective？ | PyTorch `torch/distributed/tensor/_redistribute.py:180–204` | [重分布](evidence/upstream/pytorch/redistribute-collective-types.txt)，Partial／Shard／Replicate 组合决定通信类型或本地操作 |
| 当前普通列切分何时归约 dgrad？ | Megatron `megatron/core/tensor_parallel/layers.py:1319–1321`、`1404–1467` | [条件](evidence/upstream/Megatron-LM/column-dgrad-condition.txt)、[前向传参](evidence/upstream/Megatron-LM/column-forward.txt)，普通组大小、SP 与禁用梯度归约等影响分支 |
| 前向输入保存与 SP 收集在哪里？ | 同文件 `744–797` | [自定义线性前向](evidence/upstream/Megatron-LM/linear-forward-saved-input.txt)，保存原输入后，在 SP 分支取得完整输入 |
| 反向是否包含额外输入收集？ | 同文件 `801–876` | [自定义线性反向](evidence/upstream/Megatron-LM/linear-backward-regather-and-dgrad.txt)，当前计算 wgrad 且 SP 开启时重新收集输入，dgrad 则按普通归约或 RS 分支处理 |
| 行切分输出、SP 与 bias 的顺序？ | 同文件 `1730–1776` | [Row forward](evidence/upstream/Megatron-LM/row-forward-reduce-bias.txt)，本地 GEMM 不加 bias，归约后加或返回 bias，expert 等分支单列 |
| 基本边界的 autograd 怎样定义？ | Megatron `megatron/core/tensor_parallel/mappings.py:205–242` | [copy／reduce](evidence/upstream/Megatron-LM/copy-and-reduce-autograd.txt)，输入和输出边界的前后向通信互相对应 |
| 特征 AllGather 和 sequence AllGather 的反向怎样选择？ | 同文件 `264–282`、`304–356` | [特征收集](evidence/upstream/Megatron-LM/gather-feature-autograd.txt)、[sequence 收集](evidence/upstream/Megatron-LM/gather-sequence-autograd.txt)，按副本梯度或并行贡献选择本地切分或归约切分 |
| SP ReduceScatter 的反向是什么？ | 同文件 `359–385` | [sequence RS](evidence/upstream/Megatron-LM/reduce-scatter-sequence-autograd.txt)，反向沿 token 维收集 |
| 求和 primitive 和 MLP 调用者怎样连接？ | `mappings.py:26–41`，`transformer/mlp.py:239–274`、`393–406` | [求和](evidence/upstream/Megatron-LM/allreduce-primitive.txt)、[MLP 构建](evidence/upstream/Megatron-LM/mlp-builders.txt)、[MLP 输出](evidence/upstream/Megatron-LM/mlp-second-linear-output.txt)，检查 bias 返回及可配置 submodule 的边界 |

每段原始路径的固定 commit 链接均保存在清单的 `url` 字段，正文在相应解释处直接引用。阅读 Megatron 时限定普通线性层实现及明确列出的分支，未将这些路径推定为所有 Transformer Engine、expert、重算或推理配置的共同默认实现。

## 3. 论文与官方文档

两篇原始论文的作者、日期、版本、实际阅读章节、自己的推导及实现差异见本地笔记。[论文阅读笔记](evidence/paper-notes.md)

| 一手资料 | 版本和访问日期 | 对应的检查 |
| --- | --- | --- |
| [Megatron-LM](https://arxiv.org/html/1909.08053v4) | `1909.08053v4`，`2026-11-17` | §3、公式（1）—（3）、图 3 及附录 B，列／行配对与求导边界 |
| [Reducing Activation Recomputation](https://arxiv.org/html/2205.05198v1) | `2205.05198v1`，`2026-11-17` | §4.2.2、图 5—6、公式（3）—（4）及同节输入重新收集说明 |
| [PyTorch TP API](https://docs.pytorch.org/docs/2.15/distributed.tensor.parallel.html) | `2.15`，`2026-11-17` | `parallelize_module`、`ColwiseParallel`、`RowwiseParallel` 的参数与布局默认值 |
| [PyTorch TP Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) | 当日教程页面标识 `2.15.0+cu132`，`2026-11-17` | 查看 MLP 的配对、SP 布局转换及可组合的使用方式，执行环境另列 |

ring 的字节量和轮数采用已有通信章节的同一教学模型，本文重新计算目标形状，并运行预算答案。性能条件要结合实际通信实现和硬件验证。[已发布的环通信与拓扑章节](../2026-11-11-allreduce-links-topology/README.md)

## 4. 自编实验与执行证据

| 证据 | 实际内容与范围 |
| --- | --- |
| [初始 red](evidence/runs/01-red-verification.json) | `18` 项测试在答案未实现时运行，记录异常与退出码 |
| [输入重新收集的 red](evidence/runs/03-regather-red.txt) | 加入两个真实回归用例，验证预算接口尚未覆盖该策略 |
| [最终 tests](evidence/runs/04-tests.txt) | `20` 项通过，包含 `26` 个有限差分导数 |
| [优化模式](evidence/runs/05-optimized-tests.txt) | 相同 `20` 项通过 |
| [教学输出](evidence/runs/08-answers.json) | 输出值、梯度、软件版本和三种预算，预算为公式模型 |
| [两 rank 输出](evidence/runs/09-distributed.json) | 实际 CPU／Gloo 的默认配对及强制中间收集，各方案覆盖 rank 0 和 rank 1 |
| [命令清单](evidence/runs/verification.json) | 命令、退出码、作用域、环境与代码 SHA256 |

首次 CPU 进程探测的原始输出也保留在 `02-distributed-probe.txt`。两个进程的输出可能粘连，最终采集按 `TP_RESULT` 标识及 JSON 解码边界解析，最终结构化记录来自有完整退出码的新一次验证运行。前向和反向的 `CommDebugMode` 均排除参数初始化，所见计数属于框架 collective 调用。

写入日志前，将本机绝对目录、主机名和网络地址替换为占位符，张量值、错误、collective 名称及计数保持真实。源码均来自公开官方对象，例子和图为自编。

## 5. 许可与图源

上游片段保持读取范围的原始内容，配套许可与所选源文件头一并归档。局部 `.gitattributes` 对原始片段和指定运行日志保留空白，完整字节通过 SHA256 校验。[PyTorch LICENSE](evidence/upstream/pytorch/LICENSE)、[Megatron LICENSE](evidence/upstream/Megatron-LM/LICENSE)、[Megatron 文件头](evidence/upstream/Megatron-LM/COPYRIGHT_HEADERS.txt)

两幅 Mermaid 图为自编的教学数据流，图源与正文代码块一致。前向图对应合成数字，SP 图区分四个基本布局转换和额外输入重新收集。[前向图源](assets/forward.mmd)、[SP 图源](assets/sequence.mmd)

返回 [正文](README.md)。
