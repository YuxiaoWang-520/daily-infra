# 参考资料与证据

主题为 torch.compile 的计算捕获、graph break、guards 与重新编译成本。访问日期为 2026-11-13，时区 Asia/Shanghai。运行范围为 CPU Dynamo／FX 及自定义计数后端，优化代码生成与 CUDA 性能留待专项验证。

## 关键论断

| 编号 | 论断与定位 | 本地证据 | 范围 |
| --- | --- | --- | --- |
| S1 | torch.compile 参数经 Dynamo 到用户后端 | [源码路径笔记](evidence/source-notes.md)、[后端交付](evidence/upstream/main/output-graph.lines.txt) | 当日 main 与稳定 tag |
| S2 | guards 绑定缓存执行结果，匹配成功可复用 | [GuardedCode](evidence/upstream/main/convert-frame.lines.txt)、[C++ 匹配](evidence/upstream/main/extra-state.lines.txt) | 记录的实现路径 |
| S3 | Python scalar 的值条件与 Tensor 数据接口影响图交付 | [原始日志](evidence/runs/03-torch-logs.txt)、[完整场景 JSON](evidence/runs/02-compile-lab.json) | 当前实际构建、static 模式与给定输入 |
| S4 | graph break 分段，fullgraph 可拒绝分界 | [工作负载](code/workloads.py)、[测试](code/test_answers.py) | 显式分界与数据相关 Python 分支 |
| S5 | dynamic 三种模式、0／1 特化具有不同适用条件 | [文档片段](evidence/upstream/main/zero-one-doc.lines.txt)、[实际实验](evidence/runs/02-compile-lab.json) | 简单二维算术与指定输入序列 |
| S6 | where 改写保留本例整体求和的选择语义 | [答案](code/solutions.py)、[验收测试](code/test_answers.py) | 两侧合法、有限值、相同输出形状 |
| S7 | 图分段和重新编译分别改变边界与编译事件成本 | [官方资料笔记](evidence/document-notes.md) | 正文毫秒表由明确教学假设计算 |

## 官方文档

阅读 PyTorch 2.15 的 torch.compile API、Troubleshooting、Dealing with Recompilations、Custom Backends，以及 0／1 特化文档。使用版本化入口，具体章节与本地笔记对应保存。[官方阅读笔记](evidence/document-notes.md)、[检索与访问元数据](evidence/research-log.json)

自定义后端旧地址返回 404 后，通过实际搜索定位到 torch.compiler_custom_backends.html，已成功读取。原失败与成功 URL 均保留在检索记录中。

## 源码版本与归档

官方远端为 https://github.com/pytorch/pytorch.git，缓存放在 .cache/upstream/pytorch/。当天实际 fetch 的 main 为 a0d96468b7f2089907f400411eda3b09feda8f76。正式 release 查询为 v2.15.0，对应 tag commit 为 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457。[版本记录](evidence/versions.json)、[更新输出](evidence/runs/00-upstream-fetch.txt)、[正式 release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)

归档 6 个实现文件与 2 个文档文件的关键片段，两版本合计 16 份。清单保存完整 blob SHA-256、原始行区间、片段 SHA-256 与版本间差异行数。上游源码保留原始行号和许可证，缓存 checkout 保持原状，阅读通过固定 Git 对象完成。[源码笔记](evidence/source-notes.md)、[清单](evidence/upstream/manifest.json)、[许可证](evidence/upstream/LICENSE)

本机 torch 2.15.0 报告构建 SHA 为 d0f2af8b66b20da4baa50644c0cd3f887a2eae2e。运行结果按该构建归档，stable tag 与 main 的源码分析分别标注。[环境及执行代码哈希](evidence/runs/02-compile-lab.json)

## 论文对应与教学模型

本文研究编译前端 API、缓存条件和图分段机制，直接证据来自官方语义文档与实现。成本求和是本篇明确假设下的教学模型，算术和单位经过复核。本文没有采用特定论文算法。

## 可复现验证

Python 使用 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python。基础答案先在待实现状态运行测试，保留真实红灯，再完成实现。当前环境 Python 3.12.16、PyTorch 2.15.0、macOS arm64，保持已有依赖。[初始红灯](evidence/runs/01-red-tests.txt)、[通过测试](evidence/runs/04-tests.txt)、[命令与代码哈希](evidence/runs/08-verification.json)

完整实验保存每次输入 shape、输出、后端累计计数、完整 FX 代码及预期拒绝异常。TORCH_LOGS 原始记录可与场景 JSON 互相核对。每个成功调用均执行数值对照。[实验 JSON](evidence/runs/02-compile-lab.json)、[原始 stdout](evidence/runs/02-compile-lab.stdout.txt)、[原始 guards／graph break 日志](evidence/runs/03-torch-logs.txt)

图源、两轮审查、文件校验及交付检查均在本篇目录。[图源](figures/compile-cache.mmd)、[审查](review.md)、[SHA-256 清单](SHA256SUMS)、[交付检查](evidence/delivery-check.json)
