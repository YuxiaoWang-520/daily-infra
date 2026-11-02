# 参考资料与证据

主题：多个请求什么时候能安全复用同一段前缀？访问日期为 2026-11-02，Asia/Shanghai；当前源码与版本采集起始时间为 `11:15:49+08:00`。

## 核心论断与一手来源

| 编号 | 需要证实的论断 | 一手来源与位置 | 本地归档 | 证据边界 |
| --- | --- | --- | --- | --- |
| P1 | 共享物理 KV 前缀、引用计数及修改保护的早期设计 | [PagedAttention v1](https://arxiv.org/html/2309.06180v1) §4.2、§4.4，Figure 8／10 文字，§6.4 | [论文笔记](evidence/paper-reading.md)、[读取记录](evidence/paper-acquisition.json) | 2023 年算法／实验，与当前自动哈希实现分开；不外推性能倍率 |
| D1 | APC 减少重复前缀输入处理，仍需计算后缀与输出 | [当前功能文档](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/docs/features/automatic_prefix_caching.md) | [固定原文](evidence/upstream/vllm-project/vllm/docs/features/automatic_prefix_caching.md) | 官方工作负载概述，不把“通常有益”写成无条件性能保证 |
| S1 | 指纹应覆盖父前缀、本块和额外有效输入身份 | [hash_block_tokens](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_utils.py#L592-L741) | [原始工具实现](evidence/upstream/vllm-project/vllm/vllm/v1/core/kv_cache_utils.py) | 不是每块 hash 全部权重／配置；模型池、输入身份和失效管理仍须正确 |
| S2 | 普通本地生成需保留最后位置，并受对齐与请求条件影响 | [Request 上限](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/request.py#L319-L344)、[KV 管理器](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_manager.py#L266-L307) | [Request 原文](evidence/upstream/vllm-project/vllm/vllm/v1/request.py)、[Manager 原文](evidence/upstream/vllm-project/vllm/vllm/v1/core/kv_cache_manager.py) | 公式限定单组普通生成、完整块；prompt scoring 等可有更紧上限 |
| S3 | 查找先要匹配连续前缀与缓存组 | [单组 coordinator](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_coordinator.py#L567-L599)、[完整块查找](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/single_type_kv_cache_manager.py#L789-L800) | [coordinator 原文](evidence/upstream/vllm-project/vllm/vllm/v1/core/kv_cache_coordinator.py)、[分组管理器原文](evidence/upstream/vllm-project/vllm/vllm/v1/core/single_type_kv_cache_manager.py) | 不把首个 miss 停止误写成所有后缀键均不存在 |
| S4 | touch 保护命中；零引用缓存可保留到实际驱逐 | [BlockPool touch／free](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/block_pool.py#L792-L847)、[分配顺序](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_manager.py#L569-L608) | [块池原文](evidence/upstream/vllm-project/vllm/vllm/v1/core/block_pool.py) | 释放引用与覆盖数据不同；无 GPU 并发／异步验证 |
| S5 | 当前存在受条件限制的部分块命中 | [partial_hash_hits_enabled](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/kv_cache_utils.py#L764-L803)、[第二阶段](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/v1/core/single_type_kv_cache_manager.py#L802-L823) | [源码笔记](evidence/source-reading.md)、前述完整源码 | 不以基础整块模型代替混合模型条件；未运行此路径 |
| S6 | SHA 算法之外还要约定编码与隔离范围 | [hashing.py](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/vllm/utils/hashing.py#L26-L60)、[设计文档](https://github.com/vllm-project/vllm/blob/e8ec12c5edc3e14598bb122237d2798a3d4f3fa0/docs/design/prefix_caching.md) | [编码原文](evidence/upstream/vllm-project/vllm/vllm/utils/hashing.py)、[设计原文](evidence/upstream/vllm-project/vllm/docs/design/prefix_caching.md) | salt 不等于认证／加密；教学 JSON 不冒充 Pickle／CBOR 协议 |
| E1 | 哈希身份、命中边界与状态生命周期按契约工作 | 本篇合成输入和独立计数断言 | [答案](code/answers.py)、[测试](code/test_answers.py)、[原始结果](evidence/runs/tests.txt)、[状态轨迹](evidence/runs/lab.json) | CPU 标准库元数据验证，不是实际模型 KV 或性能实验 |

## 官方源码的版本、获取与许可

- 官方远端：`https://github.com/vllm-project/vllm.git`；核验默认分支 main。
- 本次完整 SHA：`e8ec12c5edc3e14598bb122237d2798a3d4f3fa0`，提交时间 `2026-11-02T03:00:36Z`。
- 实际更新：检查缓存干净后 shallow fetch、detached checkout，命令和成功退出码在 [versions.json](evidence/versions.json)。源码缓存位于忽略目录 `.cache/upstream/vllm/`。
- 稳定发行元数据：[v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，`2026-10-18T06:50:48Z` 发布；commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。本文不单独分析稳定实现，不能把 main 分支行为称为已发行默认行为。
- 原始文件 13 份、669,945 字节：LICENSE、APC 设计与功能文档、CacheConfig、hashing、Request、KV 工具／manager／coordinator／各组 manager／block pool、Scheduler、EngineCore。逐文件读取命令和 SHA-256 见 [source-manifest.json](evidence/source-manifest.json)。
- [Apache-2.0 LICENSE](evidence/upstream/vllm-project/vllm/LICENSE) 与版权头保留；上游原始字节不做格式化。源码条件、入口、状态与未展开分支的分析在 [source-reading.md](evidence/source-reading.md)。

原论文的完整题名、作者、版本、提交日期、DOI、实际阅读章节、假设与当前实现差异在 [paper-reading.md](evidence/paper-reading.md)。本次实际访问官方完整 HTML，未声称下载或提取本地 PDF。论文全文不另行再分发，保留自己的笔记、精确定位与 [访问记录](evidence/paper-acquisition.json)。

正文两幅图均为原创，图源为 [共享映射](assets/shared-prefix.mmd)、[缓存生命周期](assets/cache-lifecycle.mmd)，不是实际 trace 或论文截图。容量数字由指定模型几何自行推导，不引用历史性能数据。

## 执行、审查和归档

- [两题的输入输出与执行说明](code/README.md)、[起始代码](code/exercises.py)、[完整答案](code/answers.py)。
- [运行摘要](evidence/runs/summary.json)、[29 项测试](evidence/runs/tests.txt)、[lab JSON](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt)、[格式检查](evidence/runs/format.txt)。
- [技术及教学两轮自审](review.md)、[窄范围独立 Python 审查与 MEDIUM 修复](evidence/python-review.md)。
- [核心校验和](evidence/SHA256SUMS)、[归档检查](evidence/runs/archive-check.txt)、[校验和复核](evidence/runs/checksums.txt)。
- [最终交付检查](evidence/runs/delivery-check.txt)：日期、索引、图源、最终路径运行及发布文件范围。

当前没有引擎安装、整模型数值比较、GPU 性能或真实多租户隔离实验；没有将这些未执行项目记为通过。GitHub 提交链接在完成通知中给出，避免文档自引用待生成的提交。
