# 资料、版本与证据

主题：KV Cache 为什么需要分页，逻辑 token 怎样找到物理块？检索日期：2026-10-28，Asia/Shanghai。逐文件实际时间、命令和 SHA-256 见 [源码清单](evidence/source-manifest.json)。

## 核心论断与一手来源

| 论断 | 来源与定位 | 本地证据及范围 |
| --- | --- | --- |
| 分块允许请求的逻辑历史对应不相邻的池内存储；按需增长仍有页尾余量 | [Kwon 等，2023，原始论文 v1，第 4.1—4.3 节](https://arxiv.org/abs/2309.06180v1)，Figure 5／6；结合当前管理器的向上取整代码。 | [论文笔记](evidence/paper-reading.md)、[源码笔记](evidence/source-reading.md)；示例数字为自选，不照搬论文性能结果。 |
| 页需求须按每请求向上取整再相加，内容字节与页容量不同 | 存储轴与独占页条件的推导；当前 `SingleTypeKVCacheManager.allocate_new_blocks`。 | [实际文件](evidence/upstream/vllm-project/vllm/vllm/v1/core/single_type_kv_cache_manager.py)、[账本输出](evidence/runs/lab.json)；不含共享、lookahead、异构组与元数据。 |
| 调度器把本轮槽需求交给缓存管理器；容量不足仍可能不被调度 | [Scheduler](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/core/sched/scheduler.py#L1424-L1443)、[allocate_slots 的容量检查](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/core/kv_cache_manager.py#L578-L607)。 | 原始文件归档；只追踪普通增长，不把所有预留分支化简成已缓存长度。 |
| 空闲块由队列管理，分配／归还涉及引用与缓存状态 | [BlockPool 获取](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/core/block_pool.py#L706-L740)、[归还](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/core/block_pool.py#L816-L849)。 | [原文件](evidence/upstream/vllm-project/vllm/vllm/v1/core/block_pool.py)；教学 FIFO 和块 `0` 可用规则均与引擎有意不同。 |
| 无 CP 的位置到槽映射是“除页长、查表、乘页长、加偏移” | [worker 映射](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/worker/block_table.py#L351-L376)、[V2 的 CP_SIZE=1 分支](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/worker/gpu/block_table.py#L382-L414)。 | [CPU 地址测试](code/test_answers.py)；不声称执行这些 Triton kernel。 |
| 槽号仍需结合实际布局 stride 才是元素地址 | [CUDA cache writer](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/csrc/cache_kernels.cu#L335-L366)，由 FlashAttention → fa_utils → custom op 追踪。 | [源码路径笔记](evidence/source-reading.md)；CUDA 平台条件明确，没有 GPU 指令／HBM 流量测量。 |
| 读取接口携带缓存、块表与长度／因果信息 | [当前 dense FlashAttention 调用](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/attention/backends/flash_attn.py#L1404-L1424)。 | 本篇验证其接口和数据流，未审核每种 Attention 后端内核，也不要求先 gather 连续 KV。 |
| 管理页与 kernel 页可能采用不同粒度 | [元数据块表转换](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/worker/utils.py#L368-L395)。 | 手算选同粒度，实际遇到拆分时需使用对应层的 ID、页长与布局。 |
| 底层缓冲初始化与请求的池内取还块是不同动作 | [allocate_kv_cache](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/vllm/v1/worker/utils.py#L564-L624)。 | 从配置大小创建缓冲及视图；不把池内逻辑回收等同 GPU 分配器释放。 |
| 旧设计文档不能替代当前实现分析 | [仓库历史声明](https://github.com/vllm-project/vllm/blob/84e455367336b62547d1f1273e9b50bc7f8edb47/docs/design/paged_attention.md#L1-L5)、[官方文档](https://docs.vllm.ai/en/latest/design/paged_attention/)。 | 文档源已固定版本归档，解释它的历史用途，不沿用其中的实现描述断言当前默认路径。 |
| SDPA 的 dropout 与矩形因果对齐语义 | [PyTorch 2.9 官方接口](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)。 | [API 阅读说明](evidence/source-reading.md)；CPU 对照只给末尾一行 Query，给定 KV 全部可见，显式 `dropout_p=0.0`。 |

## 源码版本与获取

- 官方缓存远端：`https://github.com/vllm-project/vllm.git`；默认分支 `main`。
- 检查缓存没有修改后，实际 `git fetch --depth=1 origin main` 并 `git switch --detach origin/main`，均退出 0；没有 reset、clean 或覆盖他人内容。
- 固定完整 SHA：`84e455367336b62547d1f1273e9b50bc7f8edb47`。相关文件均使用 `git show SHA:path` 导出；[16 项清单](evidence/source-manifest.json) 保留命令、字节数、哈希与时间。
- 当日查询 GitHub latest release 为 `v0.32.0`，2026-10-18 发布，tag commit 为 `5df63a7203cdb9328d824a27b682e77c7b9928be`。仅记录发布元数据，没有逐行比较稳定版，也未宣称它包含本篇默认分支的全部行为。[versions.json](evidence/versions.json)
- 本次仅静态阅读，无 vLLM 构建、设备执行或端到端生成。原文件遵循 [Apache License 2.0](evidence/upstream/vllm-project/vllm/LICENSE)，保留原版权与许可头。

## 论文、图源与获取限制

原论文的作者、版本、发布日期、DOI、阅读章节与公式对应见 [paper-reading.md](evidence/paper-reading.md)。实际通过官方 arXiv HTML 与 web PDF 访问阅读 v1，PDF 共 16 页。

本机 `curl` 的完整下载及两次续传超时，**不完整 PDF 没有用作论文证据，也没有进入文章归档**。失败记录、当时部分文件哈希和成功 web 阅读方式均如实保留在 [paper-acquisition.json](evidence/paper-acquisition.json)。不把失败下载标为成功，不将 HTML、PDF 或当前代码版本混为一谈。

本篇归档自己的论文与源码笔记、完整相关公开源文件、运行证据。论文全文未重新分发；正文 Mermaid 是自绘，源代码留在正文，不复制论文图。所有请求 ID、Tensor 与状态场景均为合成教学数据。

## 实验与复现

[环境及逐条命令](evidence/runs/summary.json)：Python 3.10.13、PyTorch 2.9.1、macOS arm64、CPU，CUDA 不可用。使用已有 Conda 环境，没有安装依赖。

- [练习及命令](code/README.md)、[答案](code/answers.py)、[测试](code/test_answers.py)、[诊断](code/run_lab.py)。
- [测试结果](evidence/runs/tests.txt)、[实际账本](evidence/runs/lab.json)、[Ruff check](evidence/runs/lint.txt)、[Ruff format](evidence/runs/format.txt)。
- [首次 RED](evidence/runs/initial-red-tests.txt) 是测试先于答案建立时的真实导入失败，非最终状态。
- 运行只验证 CPU 地址／状态语义及指定输入的 SDPA 对照。理论容量不等于实际 GPU 显存，未测吞吐、延迟、多卡、内存峰值或缓存安全隔离。
- [核心文件校验和](evidence/SHA256SUMS)、[归档链接及源码锚点检查](evidence/runs/archive-check.txt)、[技术与教学审查](review.md)、[独立代码审查](evidence/python-review.md)。
