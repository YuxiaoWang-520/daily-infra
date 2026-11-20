# 官方实现阅读笔记

访问日期：`2026-11-20`。所有实际源码定位在 [source-manifest.json](source-manifest.json)，页面 metadata 在 [web-retrieval.json](web-retrieval.json)。

## 默认配置必须绑定版本

同日查 GitHub latest release 得到 Megatron `core_v0.20.1`，而读取当日默认分支时发现 `moe_token_dispatcher_type="alltoall"`。稳定 tag 的同一配置仍为 `"allgather"`。官方 latest 文档以其版本语境提供描述，不能用它替代另一个固定 SHA 的默认配置。正文因此要求显式选择 alltoall，再追该路线。

稳定版的普通 Top-k helper 同样保留 pre／post Softmax 的分支，但没有据此声称稳定 tag 与 main 所有功能一致。当前 main 的 score function、分组路由、bias、hash、融合等选项远多于本篇的最小语义；这些分支不在 CPU 教学模型中模拟。

## 路由、容量与 aux 的三个不同对象

1. Task gate 决定主任务如何加权专家输出。当前普通非融合 Softmax 中间计算使用 FP32；教学 float64 验证计算公式，不提供 bit-exact 结论。
2. Capacity mask 在当前调用收到的路由矩阵上计算，按 `T*k` 获得平均分配数，筛除后没有重新归一化 gate。padding 与非 padding 分支的 routing map 不同；本文只分析不填充的保留分配。
3. Aux 会单独从 logits 构造对应路由／分数，统计范围也影响分母。本文 local-batch 辅助项使用未丢弃计数；并未通过 CPU task loss 的分布式通过来宣称完整 aux 训练正确。

## AlltoAll dispatcher 的实际阶段

官方 [API 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.token_dispatcher.html)列出 preprocess、permutation、EP AllToAll、可选 TP AllGather／本地 sort、逆 sort／TP ReduceScatter、返回 EP AllToAll、unpermute。顺着固定源码核对：

- MoELayer 选择 dispatcher；TopKRouter 调用打分 helper 及可选容量限制。
- dispatcher 将专家计数按目标 owner 求和得到 input splits；获取跨 rank 计数后形成 output splits，不能假设与自身 input splits 相同。
- permute 同时处理 hidden state 与概率，缓存逆映射。
- token 与概率各有 AllToAll，local expert 调用接收 token、每专家行数和 gate。
- 返回通信对调 splits，unpermute 恢复原 token 顺序；更多 TP 与融合路径另有条件。

教学两进程程序只发送 token 数据和专家 ID，gate 留在源端，数学上在专家输出返回后加权。其实际消息数和上游不相同，因此正文通信预算显式只算假设的远端 hidden-state 负载。

## 反向与空 owner

Megatron `_AllToAll.backward` 对调 input／output splits 后再次交换，体现数据重排的逆传递。CPU 程序采用同样的第一阶结构，但在 Gloo 上运行，自定义包装器没有建立高阶 collective 梯度支持。

所有源 token 对某个专家的梯度在专家 owner 计算时汇集；返回到 token 的多分支输入梯度再由索引反向相加。局部 loss 使用统一全局分母 8，没有再次按 EP 大小平均。空接收 rank 保留零 autograd 依赖并参加通信，实际样例已通过，不只停留在单进程推断。

本机为建立独立 dense 对照额外保存了完整合成张量，不适合测 EP 内存收益。没有 CUDA／NCCL、TP、fused GEMM、量化、shared expert 或重叠性能证据。
