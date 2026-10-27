# 审查记录

主题：KV Cache 怎么算大小，MHA、GQA、MQA 各改变了什么？

文章状态：published。审查时间：2026-10-27T11:25:41+08:00；最终归档目录中的实际复验时间见运行汇总。

技术审查与教学审查：主代理分别执行两轮自审。Python 代码另经独立代理审查及修正复核，范围和实际执行情况见 [独立审查记录](evidence/python-review.md)；这不代表全文经过独立技术或教学审查。

## 第一轮：技术事实与证据

- [x] 用长上下文及并发请求的容量问题引出账本，明确已缓存 token 的统计口径。
- [x] 实际更新官方 vLLM 默认分支并固定完整 commit；7 个公开文件归档，保留原许可、获取命令与哈希。
- [x] 核对 LlamaAttention、QKVParallelLinear 及其加载分片、Attention、FullAttentionSpec、FlashAttention 后端的 head 数路径。
- [x] 实际阅读 MQA 原始 v1 与 GQA 的 ACL 会议原文；GQA arXiv PDF 超时与会议版替代如实记录，未混称版本。
- [x] 从元素维度推导容量，复算 BF16 与 GiB、MHA／GQA／MQA、每 rank 与 TP 副本的不同口径。
- [x] 两道代码练习有完整答案、验收与边界；容量整数计算及分组语义有可运行测试。
- [x] 容量估算、CPU 语义、静态源码、论文实验分开陈述，无 GPU 性能或质量复现主张。

### 实际检查、发现与修正

| 严重程度 | 观察与证据 | 处理 | 复验 |
| --- | --- | --- | --- |
| LOW，已修正 | 独立审查指出 mask 位置张量继承默认设备。新增回归后真实失败，[日志](evidence/runs/default-device-before-fix.txt)。 | 对齐 `q.device`，测试与 lab 明确 CPU。 | 主代理及独立代理均实跑最终 22 tests 通过。 |
| LOW，已补齐 | 非有限输入原先仅测 Q 的 NaN，未覆盖全部拒绝契约。 | 为 Q／K／V 分别加入 NaN、正 Inf、负 Inf 子测试。 | 最终测试通过，独立代理确认覆盖。 |
| 核验，无遗留发现 | MQA 在 TP=4 时逻辑内容为 128 MiB，不能据此推断每卡 32 MiB。 | 同时追踪模型整除约束、KV 复制份数与权重 shard 选择；正文列单 rank 与四 rank 账本。 | 整数诊断为单 rank 134217728、集群 536870912 byte；源码定位吻合。 |
| 核验，无遗留发现 | 当前缓存规格支持不同 K／V 长度、状态打包和页 padding。 | 给简化容量公式列出完整假设，单独解释普通分支的化简。 | 从本地 `AttentionSpec` 原文件复核字段，未扩大结论范围。 |
| 核验，无遗留发现 | 矩形 SDPA 的因果开关不自动表达本例的历史偏移。 | 显式按绝对位置构造布尔 mask，`dropout_p=0.0`；重复 KV 后用普通 CPU SDPA 对照。 | 非完整 Prefill、单 token Decode 与分块 Query 测试通过；核对 PyTorch 2.9 文档。 |

## 第二轮：教学逻辑与表达

- [x] 先解释 head、向量与字节，从一个 token 的 32 byte 小账本推进到层数、历史长度，再引入公式。
- [x] 分开说明 K／V 两份与 BF16 两字节这两个 `2`，定义 KiB／GiB，避免只给最终数字。
- [x] 自绘 4Q／2KV Mermaid 显示读取关系；配合不同 Query 读同一 KV 却输出 5 与约 8.808 的数字例子。
- [x] 先解释分组，再引入 `Hq/Hkv` 映射和源码；TP 一节先定义 rank，再说明分片／复制。
- [x] 工程取舍说明适用条件、质量代价与容量不等于加速比；区分逻辑内容、页分配及整个进程显存。
- [x] 题目在折叠答案前给出输入与验收，详解包含数数过程、einsum 轴、独立对照与教学实现的局限。
- [x] 面试追问覆盖概念、字段、方案选择和 OOM 排查；前置链接指向已发布文章。

### 教学复核中的补充

| 严重程度 | 观察 | 修改与复验 |
| --- | --- | --- |
| LOW，已补齐 | 仅写“因果 mask”不足以让读者理解两行 Query 的历史偏移。 | 正文列出位置 3、4 各自可见的 Key 集合，说明 bool mask 和关闭 dropout；与代码和官方文档逐项对应。 |
| 核验，无遗留发现 | 以 dtype 容量表解释 FP64 实验可能造成单位混淆。 | 已明确区分正文 BF16 与练习 FP64，并展开 `480 byte` 的全部因子；两边与实际输出一致。 |

## 运行与验证记录

环境：Python 3.10.13、PyTorch 2.9.1、macOS arm64、CPU，CUDA 不可用；没有安装或升级全局依赖。命令使用 `/Users/wangyuxiao/conda3/bin/python3`，真实时间、cwd 与退出码见 [summary.json](evidence/runs/summary.json)。

| 对象 | 命令 | 实际结果 | 原始记录 |
| --- | --- | --- | --- |
| 基础答案与边界 | `python3 -m unittest discover -s code -p 'test_*.py' -v` | 22 tests，退出码 0。 | [tests.txt](evidence/runs/tests.txt) |
| 容量账本与独立 SDPA | `python3 code/run_lab.py` | 退出码 0；FP64 最大绝对误差 `2.220446049250313e-16`，满足 `atol=rtol=1e-12`；紧凑／展开 KV 为 480／960 byte。 | [lab.json](evidence/runs/lab.json) |
| Python 静态检查 | `python3 -m ruff check --select E,F,I,UP --ignore E501 code` | 退出码 0。 | [lint.txt](evidence/runs/lint.txt) |
| Python 格式检查 | `python3 -m ruff format --check code` | 退出码 0。 | [format.txt](evidence/runs/format.txt) |

首次建立答案前的导入失败保存于 [initial-red-tests.txt](evidence/runs/initial-red-tests.txt)，是测试先行的历史记录，不表示最终实现失败。

已在最终归档目录重新运行上述四项检查，均退出 0，并复核 59 个本篇相对 Markdown 链接、17 个固定源码锚点、7 个源码哈希。`shasum -a 256 -c evidence/SHA256SUMS` 验证 11 个核心文件均为 OK（代码、练习说明、公开上游文件，不含忽略的缓存）。对应 [源码清单](evidence/source-manifest.json)、[SHA256SUMS](evidence/SHA256SUMS) 和 [archive-check.txt](evidence/runs/archive-check.txt) 均已归档。

## 发布结论

- 内容审查无未解决的 CRITICAL／HIGH／MEDIUM 问题；独立代码审查 LOW 已关闭。
- 未执行：vLLM 构建、真实 GPU／多卡运行、显存峰值与带宽／延迟测量、checkpoint 转换训练与质量实验。本文未据此作实证承诺。
- 技术证据、教学表达、基础答案和最终归档检查均通过，正文与发布索引同步更新。GitHub 同步必须另经 Git 流程确认，本文审查不替代推送结果。

## 勘误与更新

暂无发布后勘误。
