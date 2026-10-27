# 当前 vLLM 源码阅读路径

固定版本：`1e10a92d1da2da4e2b70fd66d7d31b1672ba6da7`。这是 2026-10-27 更新官方 `main` 后实际读取的版本；仅静态阅读，没有运行 vLLM。

## 从模型到缓存规格

1. `vllm/model_executor/models/llama.py`：`LlamaAttention.__init__`，143—170 行分别计算本地 Query／KV head 和投影宽度；212—219 行将两种 head 数交给 Attention。288 行附近从配置取 Query 与 KV head。不同 head 数反映模型的投影结构，不是缓存管理器随意删除已有 head。
2. `vllm/model_executor/layers/linear.py`：`QKVParallelLinear.__init__`，1064—1077 行计算本地 head 与 KV 复制份数。1312—1320 行区分 Q 的 shard rank 和 K／V 的 shard rank，后者按复制份数分组选择。核对了公式背后的权重路径。
3. `vllm/model_executor/layers/attention/attention.py`：321—339 行把实际 KV dtype、Query head 与 KV head 分别保留；674—683 行将本地 KV 布局交给 `FullAttentionSpec`。532—543 行的 Q／K／V 视图也区分 Query 与 KV 的 head 轴。
4. `vllm/v1/kv_cache_interface.py`：`AttentionSpec`，556—601 行提供 head 槽位、状态大小、页内容与可选 padding。`KVCacheSpec.num_states` 在 269 行附近通过 token／state 关系求状态数。普通 dense KV 的默认 `tokens_per_state=1`，无 override 时可化简为 `P×Hkv×(Dk+Dv)×b`；本文再取 `Dk=Dv`。不是抹掉其他分支后的通用物理内存公式。
5. 同文件 `FullAttentionSpec.max_memory_usage_bytes`，642—648 行结合最大长度、块长与上下文并行计算页数量，说明整数内容账本之外仍有分配粒度和配置条件。
6. `vllm/v1/attention/backends/flash_attn.py`：1078—1095 行维护 Query／KV 比例；1195—1203 行说明紧凑 K／V 输入。相关 forward 路径将 Query 与 K／V cache 交给后端；本篇不据此承诺某一 GPU 指令、流量或速度。

## TP 复制的口径

本文只模拟同一组普通 Llama Attention 层的 TP，未加入 PP／DCP／其他 KV 分片策略。KV head 大于等于 TP 时要求整除并分片；KV head 小于 TP 时要求 TP 可被 KV head 数整除，每 rank 至少有一个 KV head。Query head 同样要能被 TP 整除。

`ModelConfig.get_num_kv_heads`（`vllm/config/model.py`，1663 行附近）也有 `max(1, total_num_kv_heads // tensor_parallel_size)`，但它另有 MLA 分支。正文通过 Llama 的尺寸约束与权重加载路径交叉核对，不把单个配置方法推广为所有架构。

## 与教学实现的对应与差异

`grouped_attention` 的连续组映射对应 Query／KV 的分组语义；显式 einsum 让读者看见共享轴。它不实现引擎缓存写入、页表、量化、调度或 fused kernel，会保留完整分数与概率。CPU SDPA 对照验证的是这个数学映射，不是主分支的设备执行或模型输出质量。
