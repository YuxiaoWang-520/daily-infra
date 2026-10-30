# 源码与 API 阅读笔记

检索日期为 2026-10-30（Asia/Shanghai），版本获取截止 `11:15:25+08:00`。后续写作不把分支名当成不可变证据；以下所有行号对应固定 commit。引擎未安装、构建或执行。

## 1. 获取方式、文件与许可证

vLLM 官方远端为 `https://github.com/vllm-project/vllm.git`，默认分支 main；先检查缓存工作区，再实际执行 `git fetch --depth=1 origin main` 和 `git switch --detach origin/main`，退出码均为 0。固定 SHA：`e49ea84e929e3d48a2bc500454dfcd4377f9a722`。

Sarathi-Serve 官方远端为 `https://github.com/microsoft/sarathi-serve.git`，默认分支 main；当天实际 shallow／sparse clone，退出码 0。固定 SHA：`96f9911790ecc00af12ee9fae47cb8fa9ba0d199`。提交时间为 `2026-01-08T10:40:57+05:30`，不是 2024 年论文的历史环境。

[versions.json](versions.json) 记录实际命令、时间、输出、分支、完整 SHA 和稳定版本元数据。vLLM `v0.32.0` 的发行与 tag commit 只作元数据核验，没有开展稳定版源码对比。浅 fetch 日志中的上游跟踪引用更新，不是对本 Wiki 远端强制推送。

每个发布源码文件均由 `git show <SHA>:<path>` 提取原始字节。13 个文件共 916,864 字节，含两个项目各自的 Apache-2.0 LICENSE；逐文件版本、命令、路径、字节数、SHA-256 与永久链接在 [source-manifest.json](source-manifest.json)。原字节不做重排或删除尾空格，因此文章下 `.gitattributes` 仅对 `evidence/upstream/**` 关闭 whitespace 检查；本篇原创文件仍正常检查。

本地源码根目录：

- [vLLM](upstream/vllm-project/vllm/)。
- [Sarathi-Serve](upstream/microsoft/sarathi-serve/)。

## 2. 先把调度入口和执行边界连起来

本篇沿当前 V1 的普通 `EngineCore.step` 入口阅读，不声称覆盖 async 的另一条完整调用链：

1. `EngineCore.step` 调用 `scheduler.schedule(...)`，将 `SchedulerOutput` 交给 executor，等待结果或采样，再调用 `update_from_output`。见 [core.py 674—704](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/engine/core.py#L674-L704)。
2. `Scheduler.schedule` 决定每请求新增计算位置数，并维护进度、预算及 KV 分配。输出结构定义在 [output.py 本地原文](upstream/vllm-project/vllm/vllm/v1/core/sched/output.py)。
3. worker 的 `execute_model` 将输出交给所选 runner；[gpu_worker.py 1337—1365](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/worker/gpu_worker.py#L1337-L1365)。executor 的进程／RPC 实现不是本篇展开范围。
4. runner 选择受 `use_v2_model_runner` 控制：[577—593](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/worker/gpu_worker.py#L577-L593)。本文后续位置构造的读取限定该标志为 False 时的 MRV1 `gpu_model_runner.py`；不推断全部署默认值。

它们共同限定“调度给出本轮计划，模型执行后再处理有效输出”的边界。CPU 模拟器把全部请求的有效输出统一记在轮末，是教学模型的明确设定，不是所有分布式执行阶段的 trace。

## 3. vLLM：核对三个容易混淆的量

| 所问的问题 | 当前代码定位 | 已确认的行为 | 没有据此推出的结论 |
| --- | --- | --- | --- |
| 本轮有多少新位置要算？ | [scheduler.py 838—855](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L838-L855) | 需求来自 token 数、占位与已有进度差；按整轮预算及输入预算裁剪 | “一个请求等于一个 token”或“预算等于总上下文长度” |
| 单请求额外阈值什么时候有效？ | [753—768](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L753-L768) | 多请求时可用正阈值，单请求不施加；adaptive 可提升到公平份额下限 | 正阈值是全配置固定 chunk；0 关闭整个能力 |
| 等待队列中装不下整段怎么办？ | [1331—1346](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L1331-L1346) | 关闭 chunking 时有不装入条件；允许时以请求剩余预算裁剪 | 完全复制了作者论文 Algo 3 的所有优先级 |
| 计数何时推进？ | [1844—1867](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/sched/scheduler.py#L1844-L1867) | 安排之后即推进 computed 与 in-flight，标记仍在 Prefill 的请求 | “computed”数值在任意时刻都证明 GPU 已执行完成 |

`schedule` 先处理 running 集合，其中可以有 partial Prefill；还存在节流、对齐、模型长度、MM 和异步处理。因此“已运行工作先于新准入”不自动等于“所有 Decode 全局严格优先于任何 Prefill”。本篇不沿固定优先级复刻当前完整调度器。

配置解释见 [config/scheduler.py](upstream/vllm-project/vllm/vllm/config/scheduler.py)：48—60 为两级 token 上限；79—90 为长 Prefill cap；134—140 为能力开关。类字段含测试便利默认值，实际 EngineArgs 会生成有效配置，不在文章中宣称某个固定默认预算对所有启动方式成立。

## 4. 历史位置、KV 可见范围与有效输出

MRV1 `_prepare_inputs` 的关键对应关系：

- 1946—1960：按各请求本轮数量重复请求索引，以 prior computed count 加 local query offset 得到逻辑位置。
- 2039—2049：Query 边界是本轮分段位置；optimistic sequence length 包含此前进度和本轮新增量，源码标注其对投机路径的乐观假设。
- 2060—2065：用尚未覆盖所有已知 token 的条件构造 discard mask。
- 2135—2145：GPU 侧序列长度／位置构造；2148 起处理 block table slot mapping。分页写入细节已在前篇讨论，本篇只沿读侧接口核验，不声称再做完整缓存内核审计。

见 [位置构造](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/worker/gpu_model_runner.py#L1946-L1960) 和 [长度、丢弃条件](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/worker/gpu_model_runner.py#L2039-L2065)。

注意力后端选择也有条件。本篇选 `flash_attn.py` 的密集接口读法：1274—1286 从缓存和元数据提取 KV、Q 分段、KV 长度与块表；1404—1428 把当前 Q、cache K/V、causal 和 block table 等传入密集实现。实际包还支持不同 FA 版本、窗口、特殊模型等，不能把单一条件下接口读取当作所有 GPU 路径的执行证据。

与练习的对应是：Query 行数取决于当前块，KV 的可见长度包含历史。真实物理 KV 采用分页缓存及元数据，练习用 `torch.cat` 只表达内容集合。没有 GPU 实验，不能由接口传参推出实测速度或所有后端 mask 均已验证。

`Scheduler._update_request_with_output` 在 2741—2743 注释明确预期正在 Prefill 的请求返回空 token IDs。runner 的 discard 逻辑与之对应。“不交付有效新 token”不应扩写为“所有路径都完全跳过采样 kernel”。

KV 容量另外受 [kv_cache_manager.py 517—547](https://github.com/vllm-project/vllm/blob/e49ea84e929e3d48a2bc500454dfcd4377f9a722/vllm/v1/core/kv_cache_manager.py#L517-L547) 约束：完整当前 sequence fit 检查可能先于本轮分配失败。chunk 不是总 KV 容量的豁免，也不能把当前输入准入检查叫作全部未来输出的最大长度预留。

## 5. 作者实现与论文如何对应？

Sarathi-Serve `SarathiScheduler._schedule`：124—160 先选 Prefill 已完成、当前可调度的旧序列；每个 Decode 累计一个 token，并有内存槽检查与抢占。162—187 再选部分 Prefill；196—235 再考虑等待请求。

`_get_seq_next_num_prefill_tokens` 74—96：静态分支中的 `min(remaining_prompt, chunk_size-num_batched_tokens)` 正是“整轮已有工作先扣预算”。动态分块时 `chunk_size` 来自另一条 schedule 分支，本文不为其外推固定公式。配置 `get_max_num_batched_tokens` 266—271 返回相应 cap，再次说明不要从变量名读成每条 Prompt 无条件固定长度。

论文 Algo 3 讨论总体调度机制，作者当前代码处理执行状态和内存抢占。我们只核对机制对应，不复现论文设置、不声称源码年代相同，更不把当前 vLLM 原样叫作论文实现。

## 6. PyTorch API 与 CPU 验证

实际阅读 [PyTorch 2.9 SDPA 官方文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) 的 causal 和 dropout 语义，本地实际运行版本为 2.9.1。官方说明非方阵 `is_causal=True` 采用左上角对齐。本篇独立对照是完整方阵输入；错误示例才故意把后半段 Query 和整段 KV 直接交给矩形 API。

参考实现手工构造全局位置 mask。Q/K 全 0 时位置 3 的输出，应从允许的 Value 0、1、2、3 平均得到 1.5；错三角为 0，错丢前缀为 3。这些数值已由实际测试核验，归档在 [tests.txt](runs/tests.txt) 与 [lab.json](runs/lab.json)。

未执行：vLLM／Sarathi-Serve 安装启动，任意 GPU Attention kernel，整模型生成、分页内核性能、真实 TTFT／ITL 统计。已执行仅为本篇 CPU 教学实现、其测试、静态检查与归档核验。
