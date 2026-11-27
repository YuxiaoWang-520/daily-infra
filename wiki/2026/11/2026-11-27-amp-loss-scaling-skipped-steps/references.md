# 参考资料与证据

主题：混合精度、loss scaling、溢出检测与更新进度。源码获取时间：2026-11-27T11:10:11+08:00。网页检索于同日执行，服务缓存日期另列于 [提取元数据](evidence/web-research.json)。

## 关键论断与来源

| 编号 | 论断 | 一手来源与定位 | 本地归档 | 范围与限制 |
| --- | --- | --- | --- | --- |
| S1 | 反向前缩放 loss，更新前反缩放 | [Mixed Precision Training v3 §3.2](https://arxiv.org/html/1710.03740v3#S3.SS2) | [论文笔记](evidence/paper-notes.md) | 数学机制；没有重跑论文实验 |
| S2 | 主权重与算术精度分别选择 | 原论文 §3.1、§3.3；[2.15 AMP 文档](https://docs.pytorch.org/docs/2.15/amp.html) | [官方笔记](evidence/official-notes.md) | 不假设所有 AMP 都额外复制一整份 FP32 权重 |
| S3 | 裁剪前反缩放，累计期间保持 scale | [2.15 AMP examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html) 的 clipping／accumulation | [官方笔记](evidence/official-notes.md)、[真实结果](evidence/runs/cpu/results.json) | 数学模型和 CPU 真实协议分别验证 |
| S4 | 每优化器状态、缓存标记、普通 step 条件 | 固定 main grad_scaler.py L43—55、342—378、414—427、477—487 | [源码导读](evidence/source-notes.md)、[完整清单](evidence/source-manifest.json) | 仅普通优化器路径；不泛化 fused／custom AMP |
| S5 | 检查与反缩放、动态 scale 更新 | 固定 main CPU AmpGradScalerKernels.cpp L97—121、168—190 | [检查片段](evidence/upstream/pytorch/main-AmpGradScalerKernels.cpp-L97-L121.txt)、[更新片段](evidence/upstream/pytorch/main-AmpGradScalerKernels.cpp-L168-L190.txt) | CPU 内核；不以源码代替 GPU 测量 |
| S6 | 多优化器各自跳步，最后统一 update | [官方多优化器示例](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers)、Python update | [更新片段](evidence/upstream/pytorch/main-grad_scaler.py-L516-L564.txt)、[真实结果](evidence/runs/cpu/results.json) | 两个独立普通 SGD、单进程 CPU |
| S7 | scaler 保存五个字段，update 后保存 | 固定 main grad_scaler.py L632—684 | [state_dict 片段](evidence/upstream/pytorch/main-grad_scaler.py-L632-L684.txt) | 内存 roundtrip，不证明磁盘崩溃恢复 |
| S8 | 原生裁剪含 epsilon，教学公式省略 | 固定 main clip_grad.py L170—188 | [裁剪片段](evidence/upstream/pytorch/main-clip_grad.py-L170-L188.txt) | 原生 FP32 对照使用 1e-6 绝对容差 |

## 源码版本与实际获取

官方仓库 [pytorch/pytorch](https://github.com/pytorch/pytorch)。默认分支 `main` 已实际 fetch，完整 SHA 为 `26bbfa344f2f4df9b06fcc980dba95fee81d5a19`。最新稳定 release `v2.15.0` 于 2026-10-28T17:09:49Z 发布，其 tag commit 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。

[source-update.json](evidence/source-update.json) 保存实际获取命令、退出码与结果摘要，原有缓存 checkout 未切换；[source-manifest.json](evidence/source-manifest.json) 保存所有片段的固定 permalink、来源路径、行号和 SHA256。13 个源码片段在 evidence/upstream/pytorch/，附 [官方 LICENSE](evidence/upstream/pytorch/LICENSE)。CPU 内核在 main／稳定 tag 全文件相同，GradScaler Python 全文件不同。

main 的 device 默认值已改为 None，稳定 tag 与本机签名默认 cuda；所有实验显式传 cpu。本机 `torch 2.15.0` 的构建 SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`，是运行包自己的版本证据。

## 原始论文与官方网页

原论文为 Micikevicius 等的 Mixed Precision Training，arXiv:1710.03740，初次 2017-10-10，本次阅读 v3（2018-02-15），实际章节为 §3、§3.1、§3.2、§3.3。[论文笔记](evidence/paper-notes.md) 记录原文对应和当前框架策略差异，没有复制全文或未读章节来充当证据。

官方网页固定到 PyTorch 2.15 的 AMP package／AMP examples，保存 [自行撰写的阅读笔记](evidence/official-notes.md)，不镜像整页。论文 HTML 的服务 acquired_at 为 2026-11-13，文档为 2026-11-26；这与今天发起检索是两个日期。搜索只用于定位一手来源，不以聚合器摘要代替论文与代码。

## 可复现证据与图源

两道练习入口、完整答案和合同在 [code/README.md](code/README.md)。[运行清单](evidence/runs/manifest.json) 保存命令、环境变量、时间、退出码及最终 Python 文件摘要；[CPU 原始输出](evidence/runs/cli.txt) 与 [严格 JSON](evidence/runs/cpu/results.json) 记录实际数据。测试和代码审查见 [review.md](review.md)。

[Mermaid 图源](figures/scaling-flow.mmd) 与正文保持同一流程，作者自行编写，没有引用外部图像。所有输入是公开机制的合成数值，不包含内部源码、标识或训练数据。最终文件完整性见 [SHA256SUMS](evidence/SHA256SUMS)，交付核验见 [delivery-check.json](evidence/delivery-check.json)。
