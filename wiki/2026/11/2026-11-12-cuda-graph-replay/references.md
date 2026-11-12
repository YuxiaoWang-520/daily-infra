# 参考资料与证据

主题为 CUDA Graph 的重复提交成本与捕获约束。检索日期为 2026-11-12，使用 Asia/Shanghai。正文按普通手动前向捕获展开，CPU 实验、教学模型、CUDA API 和未执行的 CUDA 验证分别记录适用范围。

## 关键论断与证据

| 编号 | 论断 | 一手定位 | 本地证据 | 范围 |
| --- | --- | --- | --- | --- |
| S1 | 图记录节点与依赖，实例化后重放 | NVIDIA Guide §4.2.1、§4.2.2 | [官方文档笔记](evidence/document-notes.md) | CUDA API 机制 |
| S2 | 输入输出地址及布局在普通 replay 中固定 | PyTorch 2.15 notes/cuda 的 API／Constraints | [两个版本约束源码](evidence/upstream/main/cuda.md.lines.txt) | 手动 capture／replay |
| S3 | replay 最终调用 cudaGraphLaunch | main CUDAGraph.cpp 的 replay | [C++ 源码](evidence/upstream/main/CUDAGraph.cpp.lines.txt) | 固定 main SHA |
| S4 | native allocator 用私有池保护捕获地址 | CUDACachingAllocator 的 Interaction 注释 | [allocator 源码](evidence/upstream/main/CUDACachingAllocator.cpp.lines.txt) | 指定 allocator 路径 |
| S5 | 条件节点和受限制的图更新可表达额外动态性 | NVIDIA Guide §4.2.3／4.2.4，PyTorch Data Dependent Control Flow | [阅读笔记](evidence/document-notes.md) | 本篇 CUDA 扩展实机验证待执行 |
| S6 | 两种队列完成时刻与 501 次严格获益 | 本篇递推和逐操作模拟 | [模型](code/launch_model.py)、[测试](code/test_answers.py) | 明确假设下的模型结果 |
| S7 | copy_ 更新数据，输出借用引用随复用变化 | 实际 CPU torch 运算 | [原始实验](evidence/runs/02-cpu-lab.json) | CPU storage 与别名 |

## 上游版本与来源

实际 fetch 的官方远端为 `https://github.com/pytorch/pytorch.git`，完整默认分支 SHA 为 `18f2eae1b30a6057d56ad2585a89fb3b965c3f43`。GitHub release API 返回 v2.15.0，tag 对应完整 SHA 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。[版本 JSON](evidence/versions.json)、[fetch 与查询输出](evidence/runs/00-upstream-fetch.txt)、[正式 release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)

核验文件包括 `torch/cuda/graphs.py`、`aten/src/ATen/cuda/CUDAGraph.cpp`、`c10/cuda/CUDACachingAllocator.cpp` 和 `docs/source/notes/cuda.md`。阅读与源码提取使用固定对象，归档保留原始行号和许可证。main 与稳定版均已保存关键片段，两个版本存在实现差异。[执行路径和版本差异笔记](evidence/source-notes.md)、[片段／blob 校验清单](evidence/upstream/manifest.json)、[上游许可证](evidence/upstream/LICENSE)

本机实验使用已有 Python 3.12.16、torch 2.15.0，包构建 SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。macOS arm64 上 `torch.cuda.is_available()` 返回 false。没有安装或升级依赖，基础验证使用 CPU。[环境与代码哈希](evidence/runs/02-cpu-lab.json)

## 官方资料与论文范围

阅读 PyTorch 2.15 CUDA semantics、CUDAGraph API 和 NVIDIA CUDA Programming Guide §4.2。NVIDIA 页面使用滚动地址，记录实际抓取的发布日期元数据 2026-10-31，不推断额外 Toolkit 版本。网页采用中文阅读笔记和章节定位归档，PyTorch 相关文档源码随固定 Git 对象保存。[检索元数据](evidence/research-log.json)、[官方文档笔记](evidence/document-notes.md)

本文讨论 API 执行与框架生命周期机制，主证据来自官方文档和实现。成本递推是本篇带显式假设的教学模型，逐操作模拟与边界测试已验证其实现。本文没有采用特定论文算法，论文阅读字段由该范围说明替代。

## 实验与完整性

初始答案占位实现先运行验收测试，记录真实失败，再完成实现并运行通过测试。原始日志、CPU 实验、CUDA 可用性探测和运行代码 SHA-256 均保存在本篇目录。代码审查和交付检查列于审查记录。[初始红灯](evidence/runs/01-red-tests.txt)、[通过测试](evidence/runs/04-tests.txt)、[CPU 实验](evidence/runs/02-cpu-lab.json)、[CUDA 探测](evidence/runs/03-cuda-probe.json)、[审查记录](review.md)

文件完整性使用本地 SHA-256 清单核验，交付检查同时验证相对链接、索引、图源、源码重建与执行代码哈希。[校验清单](SHA256SUMS)、[交付核验](evidence/delivery-check.json)
