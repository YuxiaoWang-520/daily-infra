# 参考资料与证据

主题：普通因果 Transformer 的逐步生成、Prefill／Decode 与逐层 KV Cache。访问日期：2026-10-26（Asia/Shanghai）。源码和论文的实际获取时间、命令、退出码与 SHA-256 见随文 JSON；性能未实测。

## 关键论断怎样核验

| 编号 | 论断 | 一手证据 | 本地证据及范围 |
| --- | --- | --- | --- |
| S1 | 因果约束让旧位置看不到追加的未来；逐位置部件不改变此前缀依赖 | Transformer 第 3.1、3.2.3、3.3 节 | 下方论文笔记；完整重算前缀／长序列的 logits 和各层 K/V 对照测试。缓存等价性是基于这些前提的逐层推导。 |
| S2 | Prefill 建立 Prompt K/V 并预测首个新 token，后续复用旧状态 | PagedAttention 第 2.1—2.2 节 | 下方笔记；[lab](evidence/runs/lab.json) 的输入长度、生成 ID 和缓存进度。 |
| S3 | 当前 Llama 路径生成 QKV，并按实际 positions 处理 Q/K | [llama.py 224—234](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/model_executor/models/llama.py#L224-L234) | [文件](evidence/upstream/vllm-project/vllm/vllm/model_executor/models/llama.py)。未运行 Llama。 |
| S4 | 缓存更新与 Attention 之间有显式依赖，缓存和槽位按层取得 | [attention.py 544—583](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/model_executor/layers/attention/attention.py#L544-L583)、[719—748](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/model_executor/layers/attention/attention.py#L719-L748)、[772—799](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/model_executor/layers/attention/attention.py#L772-L799) | [文件](evidence/upstream/vllm-project/vllm/vllm/model_executor/layers/attention/attention.py)。受 backend 是否内含更新、跨层共享、K/V、slot mapping 等条件约束。 |
| S5 | 所读后端散写新 K/V，并让 Query 使用缓存与序列元信息 | [flash_attn.py 1432—1466](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/v1/attention/backends/flash_attn.py#L1432-L1466)、[1404—1428](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/v1/attention/backends/flash_attn.py#L1404-L1428) | [文件](evidence/upstream/vllm-project/vllm/vllm/v1/attention/backends/flash_attn.py)。普通 decoder、非 DCP 分支；没有 GPU 执行、kernel 内部或所有后端证明。 |
| S6 | 当前调度器用统一 token 进度推进请求 | [scheduler.py 677—689](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/v1/core/sched/scheduler.py#L677-L689) | [文件](evidence/upstream/vllm-project/vllm/vllm/v1/core/sched/scheduler.py)。未据此声称已审查所有调度特性。 |
| S7 | 请求释放可能等待连接器或在途写入 | [scheduler.py 2910—2944](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/v1/core/sched/scheduler.py#L2910-L2944)、[2955—2971](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/v1/core/sched/scheduler.py#L2955-L2971) | 同上；[manager.free 627—636](https://github.com/vllm-project/vllm/blob/7128f568ac036f7b7f21860d1928747ac6f78213/vllm/v1/core/kv_cache_manager.py#L627-L636) 委托 coordinator。块池与引用计数内部未在本篇追踪。 |
| S8 | 非方阵 SDPA 的 is_causal 不自动表达历史位置偏移 | [PyTorch 2.14 SDPA 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | 文档定义 upper-left 对齐，布尔 True=参与。Q 长 1、K 长 3 时仅第一列可见；[测试](code/test_answers.py) 在本机 2.9.1 CPU 实证输出 10，对照显式历史 mask 输出 20。未把文档版本当成本机版本。 |

## 源码版本与读取边界

官方仓库：[vllm-project/vllm](https://github.com/vllm-project/vllm)。实际获取默认分支 `main`，固定完整 commit `7128f568ac036f7b7f21860d1928747ac6f78213`，提交时间 2026-10-26 10:40:54（UTC+08:00）。获取、稀疏检出与修正路径的记录见 [upstream-acquisition.json](evidence/upstream-acquisition.json)。没有比较稳定 release，不从 main 推导已发布版本或线上默认行为。

缓存放在 `.cache/upstream/vllm/`，本篇归档 7 个相关文件，包含 [Apache-2.0 LICENSE](evidence/upstream/vllm-project/vllm/LICENSE)。[source-manifest.json](evidence/source-manifest.json) 保留每个文件的路径、固定 URL、完整 SHA、字节数、SHA-256 和核验时间；工作区文件逐份与 `git show <commit>:<path>` 比较。

实际沿 LlamaAttention.forward → Attention.forward → 按层取上下文 → KV 更新 → 后端 Attention 调用边界阅读。FlashAttentionBackend 的相关 forward、缓存更新、普通 decoder 与 encoder 分支均对照；调度器阅读 schedule 的统一进度说明，以及请求完成到缓存管理器的释放调用。没有沿所有 backend selector、权重加载、编译、GPU runner、coordinator 块池和底层 kernel 追踪，不宣称端到端引擎运行证据。

当前路径包含多种 FA、量化、DCP、滑动窗口、跨层 KV 共享等配置；正文选定的一条路径用于解释缓存职责。后端源码里的物理缓存布局不能视为所有引擎的统一 API。

## 原始论文阅读笔记

### P1：Attention Is All You Need

Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Łukasz Kaiser、Illia Polosukhin。首次提交 2017-06-12；本次读取 [arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762v7)，修订日期 2023-08-02；[DOI](https://doi.org/10.48550/arXiv.1706.03762)。

实际阅读第 3.1 节 decoder 约束、3.2.1 节及公式（1）、3.2.3 节的 causal mask、3.3 节及公式（2）、3.4 节词表预测和 3.5 节位置表示。本文所用推理是：旧行的可见集合不随未来追加改变，且其他部件逐位置作用，所以固定前缀的历史状态可逐层复用。

原文为 encoder-decoder、post-norm 结构，包含 cross-attention，最终采用正弦位置编码；本文是 decoder-only、pre-norm、单 head、无 cross-attention，使用固定随机的可学习绝对位置表。只保留因果性与逐位置计算相关的不变量，不称为逐行复刻原论文。当前 Llama 的 RoPE／RMSNorm 等另由实际源码识别，练习没有实现其算法。

### P2：Efficient Memory Management for Large Language Model Serving with PagedAttention

Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica。读取 [arXiv:2309.06180v1](https://arxiv.org/abs/2309.06180v1)，2023-09-12，SOSP 2023；论文 DOI [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165)。

实际阅读第 2.1 节公式（1）—（3）、2.2 节 Prompt／逐 token 生成及缓存说明。这里明确区分建立 Prompt 状态并预测首 token，与后续只新算当前 K/V；同一 ID 的缓存还依赖位置和此前缀。正文的跨层推理与这个过程相互核对。

本篇不讲论文的分页算法、共享与性能实验，也不引用论文加速比来代表当前 vLLM。2017 年因果结构、2023 年服务过程与 2026 年 main 实现是三份时间不同的证据。

两份 PDF 均实际下载完成并经 `pdftotext -layout` 提取。初次短时下载超时，随后断点续传成功；原始失败与成功记录、PDF 字节数和 SHA-256 见 [paper-acquisition.json](evidence/paper-acquisition.json)。原文只留本地缓存，不在本私密教材仓库复制；归档自己的阅读笔记、精确版本与定位，方便读者从原站获取。

## 练习和验收口径

[answers.py](code/answers.py) 使用 CPU、推理模式，`cached_attention` 保持 FP32／FP64 输入类型，TinyDecoder 固定 FP64。它是按 token 顺序追加的功能参考；`torch.cat` 复制全部历史且分配新存储，不能当作 GPU 缓存实现或内存性能基线。

测试直接调用独立 PyTorch SDPA：新块输出对照完整因果矩阵的对应行，模型 full 路径不调用 `cached_attention`。逐步比较 logits、每层完整 K/V 和历史前缀不变性。单步和多 token 都验收，避免只比较最终 ID。

工作量 18→6 指逐位置部件处理的位置数；52→21 指每层、每 head 有效的因果配对数。1536 字节仅为指定 Toy 配置的 K/V 有效载荷。三者均由正文条件推导，不是实际 FLOPs、峰值或加速比。

运行证据见 [summary.json](evidence/runs/summary.json)、[tests.txt](evidence/runs/tests.txt)、[lab.json](evidence/runs/lab.json)。完整性检查见 [archive-check.txt](evidence/runs/archive-check.txt) 和 [SHA256SUMS](evidence/SHA256SUMS)。
