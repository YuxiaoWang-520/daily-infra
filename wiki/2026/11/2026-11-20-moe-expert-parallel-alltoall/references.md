# 参考资料与证据

主题：MoE Top-k 门控、专家所有权、AllToAll dispatch／combine、容量与负载。访问日期：`2026-11-20`（Asia/Shanghai）；准确源码获取时间在 [source-update.json](evidence/source-update.json)。

## 关键论断与来源

| 编号 | 关键论断 | 一手定位 | 本地证据 | 适用范围 |
| --- | --- | --- | --- | --- |
| S1 | 稀疏 gate 对专家输出加权，KeepTopK 与 Softmax 的先后顺序有意义 | [Shazeer 等，2017](https://arxiv.org/html/1701.06538v1)，第 2 节式（1），第 2.1 节式（3）—（5） | [论文笔记](evidence/paper-notes.md) | 原文带噪声；本文去噪声的确定性教学门控不是完整论文复现。 |
| S2 | Switch 的完整专家概率、Top-1、容量与辅助 loss | [Switch v3](https://arxiv.org/html/2101.03961v3)，第 2.1—2.2 节式（1）—（6），附录 F | [论文笔记](evidence/paper-notes.md) | 原文 Top-1 与本文默认 post-Top-k 门控区分；不挪用性能结果。 |
| S3 | 当前 main 与稳定版 dispatcher 默认不同 | [main 配置](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/transformer_config.py#L1144-L1155)、[稳定版配置](https://github.com/NVIDIA/Megatron-LM/blob/f7aa73b2d9cdda75421be7067d221a5cb90e0f43/megatron/core/transformer/transformer_config.py#L1010-L1020) | [main](evidence/upstream/Megatron-LM/main-dispatcher-default.txt)、[stable](evidence/upstream/Megatron-LM/stable-dispatcher-default.txt) | 两个指定 SHA 的默认值，不表示运行时一定使用此值。 |
| S4 | pre／post Softmax 路线与 FP32 中间概率 | [Top-k helper](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/moe/moe_utils.py#L1078-L1094) | [实际分支](evidence/upstream/Megatron-LM/topk-softmax-scope.txt)、[稳定版对照](evidence/upstream/Megatron-LM/stable-topk-softmax.txt) | 普通 Softmax、非融合、无分组／bias；教学使用 float64，不能当作 kernel bit-exact 复现。 |
| S5 | 容量按 routing batch 计算，保留 mask 后不重归一化 | [容量约束](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/moe/moe_utils.py#L1271-L1305) | [公式](evidence/upstream/Megatron-LM/capacity-formula.txt)、[mask](evidence/upstream/Megatron-LM/capacity-mask-no-renormalization.txt) | 统计 batch 范围由调用方确定；本文容量没有在真实分布式实验启用。 |
| S6 | 辅助 loss 的分母与输入范围必须匹配 | [公式说明](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/moe/moe_utils.py#L84-L116)、[实现](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/moe/moe_utils.py#L171-L175) | [归一化](evidence/upstream/Megatron-LM/aux-normalization-scope.txt)、[实现](evidence/upstream/Megatron-LM/aux-implementation.txt)、[独立构造的 aux 输入](evidence/upstream/Megatron-LM/independent-aux-routing-inputs.txt) | 本文实现 local-batch、无 padding 的简化；没有做分布式辅助 loss 训练。 |
| S7 | 按源／目标形成不等长 splits 并保存逆映射 | [splits](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/moe/token_dispatcher.py#L573-L606) | [计数](evidence/upstream/Megatron-LM/send-receive-splits.txt)、[排列](evidence/upstream/Megatron-LM/pack-and-record-inverse.txt) | 当前 AlltoAll dispatcher；TP>1 还有其他布局变换。 |
| S8 | dispatch、专家调用、combine 和反排列是不同阶段 | [发送 token 与概率](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/transformer/moe/token_dispatcher.py#L719-L740)及相关阶段 | [dispatcher 选择](evidence/upstream/Megatron-LM/dispatcher-selection.txt)、[发送](evidence/upstream/Megatron-LM/dispatch-tokens-and-probs.txt)、[专家](evidence/upstream/Megatron-LM/local-expert-call.txt)、[返回](evidence/upstream/Megatron-LM/inverse-alltoall-combine.txt)、[恢复](evidence/upstream/Megatron-LM/restore-token-order.txt) | 本文源端加权与上游传递 gate 的消息数不同；不冒称逐行复现。 |
| S9 | AllToAll 第一阶反向对调 splits | [自定义反向](https://github.com/NVIDIA/Megatron-LM/blob/2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a/megatron/core/tensor_parallel/mappings.py#L473-L488) | [前向](evidence/upstream/Megatron-LM/autograd-alltoall-forward.txt)、[反向](evidence/upstream/Megatron-LM/autograd-alltoall-backward.txt) | 真实教学包装器使用 CPU／Gloo；没有高阶梯度或 NCCL 性能结论。 |
| S10 | PyTorch `all_to_all_single` 的 input／output split 接口 | [API 源码文档](https://github.com/pytorch/pytorch/blob/3d9ded4cb906777b32d099ef17dfb28d50b623c2/torch/distributed/distributed_c10d.py#L6460-L6518) | [API 片段](evidence/upstream/pytorch/alltoall-single-splits.txt)、[真实两进程结果](evidence/runs/07-distributed.json) | 实测运行安装包，抓取 main 只作语义对照。 |

## 源码版本、获取与许可证

实际 fetch 的官方仓库：

- `https://github.com/NVIDIA/Megatron-LM.git`：默认分支 `main`，`2e7c0b68b3aa75c2ebfc7eb35afcc513b7aad72a`；最新稳定 release [core_v0.20.1](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.1)，tag commit `f7aa73b2d9cdda75421be7067d221a5cb90e0f43`，发布于 `2026-11-19T21:26:28Z`。本次稳定版默认 `allgather`，main 默认 `alltoall`；稳定版的 Softmax 范围也单独留有片段。
- `https://github.com/pytorch/pytorch.git`：默认分支 `main`，`3d9ded4cb906777b32d099ef17dfb28d50b623c2`；最新稳定 release [v2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，tag commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。

缓存位于 `.cache/upstream/`，先检查后执行 `fetch --no-tags --depth=1`，另取明确稳定 tag；没有 checkout、reset 或清理既有缓存。Megatron 原本的 no-checkout 状态被保留。获取命令、退出码和输出见 [source-update.json](evidence/source-update.json)。

[source-manifest.json](evidence/source-manifest.json)包含 `25` 个关键片段，共 `511` 行，逐一记录完整 SHA、源路径、行范围、完整文件哈希、片段哈希及永久链接。两仓库的完整 LICENSE 与所引文件版权头均归档；片段保留原始字节，不作措辞或空白改写。文章 `.gitattributes` 只豁免上游原文及原始运行文本的空白检查。

## 论文、官方文档与阅读方式

1. Shazeer 等，*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*，arXiv `1701.06538v1`，`2017-01-23`。阅读第 2 节加权专家式、第 2.1 节 Noisy Top-k 及训练门控讨论、第 4 节和附录 A 的 importance／load 区别。[固定 HTML](https://arxiv.org/html/1701.06538v1)、[版本页](https://arxiv.org/abs/1701.06538)。
2. Fedus、Zoph、Shazeer，*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*，arXiv `2101.03961v3`，v1 为 `2021-01-11`，v3 为 `2022-06-16`。阅读第 2.1—2.2 节与附录 F 中的 dispatch／combine 结构。[固定 HTML](https://arxiv.org/html/2101.03961v3)、[版本页](https://arxiv.org/abs/2101.03961)。
3. Megatron 官方 [MoE AlltoAll dispatcher API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html)：用于定位流程和 TP 条件，不把浮动文档默认值覆盖到某个固定源码版本。[自己的阅读笔记](evidence/reading-notes.md)。

使用 Parallel 搜索并提取上述一手页面。访问发生于 `2026-11-20`，工具的页面缓存日期较早，准确 metadata 见 [web-retrieval.json](evidence/web-retrieval.json)。固定论文不因缓存日期变成另一个版本；引擎实现事实以当日 fetch 为准。

论文页面标示 arXiv 永久非独占分发许可，没有取得在本仓库重新分发全文或插图的许可，因此只归档自己的[论文笔记](evidence/paper-notes.md)、公式重算与精确版本定位，不打包 PDF／HTML 原文。没有复用论文性能图和硬件结果。

## 可复现验证

- 代码：[六文件与命令](code/README.md)，均为公开机制和自编合成输入，没有内部代码、数据或日志。
- [实际环境及冻结哈希](evidence/runs/verification.json)：Python `3.12.16`、PyTorch `2.15.0`，包 build SHA `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`，Darwin arm64，CUDA 不可用、Gloo 可用。没有安装或升级依赖。
- [先写测试的 RED](evidence/runs/01-red-tests.txt)；实现后 [23 项普通测试](evidence/runs/02-tests.txt)与 [23 项优化模式测试](evidence/runs/03-optimized-tests.txt)通过，包含独立 dense 参考与有限差分。[Ruff](evidence/runs/04-ruff.txt)、[格式](evidence/runs/05-format.txt)、[数值及假设预算](evidence/runs/06-answers.json)。
- [真实双进程原始输出](evidence/runs/07-distributed.txt)与[四条解析结果](evidence/runs/07-distributed.json)：混合路由、零接收 owner 均通过；输出、token／logits／专家梯度及一步 SGD 对照有实际断言。最大绝对误差约 `4.34×10⁻¹⁹`。未测试分布式容量、aux 训练、GPU、TP 或重叠。
- [代码独立审查](evidence/python-review.json)：固定模型专用 Python 审查器因不受支持而未能启动；通用独立审查者实际检查六文件并通过 `7` 项针对性测试。没有把专用审查器说成成功执行。
- 原始日志仅替换本机绝对路径与机器名；图为自编 Mermaid 依赖图，[图源](figures/dispatch-combine.mmd)与正文一致。审查结论见 [review.md](review.md)，文件校验见 [SHA256SUMS](evidence/SHA256SUMS)和[交付检查](evidence/delivery-check.json)。
