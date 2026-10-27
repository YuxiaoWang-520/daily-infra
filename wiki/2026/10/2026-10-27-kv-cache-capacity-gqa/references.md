# 资料、版本与证据范围

正文：[KV Cache 容量与 MHA／GQA／MQA](README.md)。实际检索日期：2026-10-27，Asia/Shanghai；逐文件时间见 [源码清单](evidence/source-manifest.json)。

## 核心论断怎样核验

| 论断 | 一手来源与定位 | 本地笔记及验证 |
| --- | --- | --- |
| MQA 保留 Query 的 head 维，K／V 共享单组 head | [Shazeer，2019，v1，第 2.4 与第 3 节](https://arxiv.org/abs/1911.02150v1) | [论文阅读](evidence/paper-reading.md)，核对伪代码中保留和移除的 `h` 轴；本篇使用独立编写的 PyTorch 教学实现。 |
| GQA 的每组 Query 共用一组 KV，MHA／MQA 是两个端点 | [Ainslie 等，EMNLP 2023，第 2.2 节、Figure 2](https://aclanthology.org/2023.emnlp-main.298.pdf) | [论文阅读与对应关系](evidence/paper-reading.md)，实际查看 PDF 第 2 页的分组图；正文自绘 4Q／2KV 图。 |
| 结构转换不是推理配置的无损切换 | 同论文第 2.1—2.2 节、3.1—3.3 节 | 投影矩阵分组平均后继续训练；本次未执行权重转换、训练或质量评测，不套用其模型实验结论。 |
| 模型分别维护本地 Query 和 KV head，少量 KV head 在 TP 中复制 | [当前 vLLM LlamaAttention](https://github.com/vllm-project/vllm/blob/1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7/vllm/model_executor/models/llama.py#L143-L170) | [源码阅读](evidence/source-reading.md)、[容量与 TP 账本](evidence/runs/lab.json)。 |
| KV 复制有投影加载路径支持，不只是一个估算公式 | [QKVParallelLinear](https://github.com/vllm-project/vllm/blob/1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7/vllm/model_executor/layers/linear.py#L1064-L1077)、[shard 选择](https://github.com/vllm-project/vllm/blob/1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7/vllm/model_executor/layers/linear.py#L1312-L1320) | [原文件](evidence/upstream/vllm-project/vllm/vllm/model_executor/layers/linear.py)；不宣称真实多 GPU 已运行。 |
| 普通 KV 页内容按 head、状态数、K／V 长度与 dtype 计数 | [AttentionSpec](https://github.com/vllm-project/vllm/blob/1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7/vllm/v1/kv_cache_interface.py#L556-L601) | [实际文件](evidence/upstream/vllm-project/vllm/vllm/v1/kv_cache_interface.py)；说明 layout override、padding、非同长 K／V 的边界。 |
| 缓存 dtype 与模型权重 dtype 要分别核对 | [Attention 的 dtype 与尺寸](https://github.com/vllm-project/vllm/blob/1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7/vllm/model_executor/layers/attention/attention.py#L321-L339) | [原文件](evidence/upstream/vllm-project/vllm/vllm/model_executor/layers/attention/attention.py)，并跟踪 FullAttentionSpec 的构造。 |
| 后端保留每个 KV 对应的 Query 数 | [FlashAttention 实现字段](https://github.com/vllm-project/vllm/blob/1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7/vllm/v1/attention/backends/flash_attn.py#L1078-L1095) | [原文件](evidence/upstream/vllm-project/vllm/vllm/v1/attention/backends/flash_attn.py)，只确认接口和语义，不推断实际 HBM 流量。 |
| SDPA 布尔 mask 的 True 表示可见，矩形因果 mask 有对齐约定 | [PyTorch 2.9 官方文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) | 对照显式构造带历史偏移的布尔 mask，`dropout_p=0.0`；重复 K／V 后使用普通 CPU SDPA，不启用其 GQA 开关。 |

容量公式从存储轴逐项推导，数值由 [参考答案](code/answers.py) 与 [测试](code/test_answers.py) 复算；不把论文或其他引擎的容量表直接换成本文结果。

## 源码获取与版本边界

- 官方远端：`https://github.com/vllm-project/vllm.git`，默认分支 `main`。
- 检查上游缓存无修改后，实际执行 `git fetch --depth=1 origin main`，再切到已获取的 `origin/main` 的 detached checkout；没有 reset、clean 或覆盖他人文件。
- 本次固定 commit：`1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7`。只读取并归档相关文件，没有编译或执行这个版本。
- GitHub 当日 latest release 查询得到 `v0.32.0`，2026-10-18 发布，commit 为 `5df63a7203cdb9328d824a27b682e77c7b9928be`。只记录发布元数据，未声称它与上述默认分支逐行相同。[版本记录](evidence/versions.json)。
- 本地公开源码缓存：`.cache/upstream/vllm/`。7 个归档文件的命令、精确路径、完整 SHA、字节数和 SHA-256 见 [manifest](evidence/source-manifest.json)；许可见 [Apache License 2.0](evidence/upstream/vllm-project/vllm/LICENSE)。

## 论文、图源与再分发

[论文阅读笔记](evidence/paper-reading.md) 记录题名、作者、版本、原始日期、实际阅读章节与实现对应。MQA 下载的是 arXiv v1；GQA 的 arXiv v3 PDF 获取超时后，改从官方 ACL Anthology 下载会议版并阅读，**没有把会议版冒称为 v3 PDF**。[获取命令、失败与成功记录及哈希](evidence/paper-acquisition.json)。

原 PDF 留在本地研究缓存，仓库归档自己的阅读笔记、来源定位和获取校验记录，不重新分发全文及原文提取文本。GQA 图已在本机渲染检查；正文 Mermaid 为自行绘制，图源直接保留在正文，箭头表示读取关系，不是实测 trace。

## 实验边界与复现

实际环境见 [summary.json](evidence/runs/summary.json)：Python 3.10.13、PyTorch 2.9.1、macOS arm64、CPU，无 CUDA。没有安装或升级全局依赖。

- [执行入口与契约](code/README.md)、[最终测试](evidence/runs/tests.txt)、[诊断账本](evidence/runs/lab.json)、[Ruff lint](evidence/runs/lint.txt)、[format](evidence/runs/format.txt)。
- [首次 RED](evidence/runs/initial-red-tests.txt) 是参考实现尚未建立时的真实导入失败；[默认设备修复前](evidence/runs/default-device-before-fix.txt) 是独立审查建议新增回归后复现的真实错误。两者不是最终测试状态。
- 字节数为逻辑内容计算。没有真实 GPU 内存分配、kernel 性能、多卡运行、上游构建、模型训练或质量实验。
- 代码与公开源码的核心文件校验和：[SHA256SUMS](evidence/SHA256SUMS)；归档、相对链接与源码锚点核验见 [archive-check.txt](evidence/runs/archive-check.txt)。
