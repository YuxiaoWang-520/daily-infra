# 官方实现阅读笔记

当天实际 fetch 的 PyTorch main 为 `205e1e1fca6c6634bff5b06691d6648a4f72fe9a`，稳定 tag 为 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。路径、行号、原样片段、许可证与摘要见 [source-manifest.json](source-manifest.json)。

## Attention 的两种表示

1. `torch.nn.functional.scaled_dot_product_attention` 绑定 native 算子。接口说明 bool True 为参与，并提醒 dropout 始终由参数决定；参考实验显式使用 0.0。
2. native `convert_boolean_attn_mask` 将 True 变成 0、False 变成负无穷，供分数加法使用。它与 MultiheadAttention 的 padding mask 含义不同。
3. 本文指定 math backend 后，dispatch 进入 `_scaled_dot_product_attention_math`；QK 矩阵乘发生在加入 mask 前，随后 safe softmax 和 AV。显式稀疏关系不能被说成这条 dense 路径已经省略对应乘法。
4. `torch.nn.attention.varlen.varlen_attn` 接收 `[T,H,D]` 一类布局、累计边界和最大长度。`window_size=(-1,0)` 被转成 causal 标志；backend 先按条件选择，再把元数据传给 `torch_attn._varlen_attn`。
5. 官方例子使用 CUDA int32 累计位置。本文生成 `[0,3,5]` 作 CPU metadata 核对，未把 CPU dense QKV 的 `[1,1,T,D]` 原样传给另一个接口，也未执行该 GPU 路径。
6. 稳定版 varlen 公共参数与 main 匹配，完整文件不同；main 包括有条件的 Flash／cuDNN 选择。没有据此对本机安装包承诺相同后端或数值。

## CE 与监督位置

1. `F.cross_entropy` 将 logits 和现有 target 交给 `_C._nn.cross_entropy_loss`，没有 LM 标签移位步骤。题目在预测位置上生成 labels，因此不会再 shift。
2. 当前 native CE 对整数 target、label_smoothing=0 进入 log_softmax 加 NLL。NLL 跳过 ignore_index；无类别权重时，总权重是非忽略 target 数，mean 再除以该数。
3. 本文 loss helper 只接受有限 CPU logits、int64 类别标签，返回 sum 与有效 count。全忽略 sum 可微为零，但不能用零分母归一化；实际 mean NaN 单独记录为布尔判据，严格 JSON 中没有非标准 NaN 数值。
4. 普通 DDP 平均除以 rank 数是正文明确的推导假设，并链接已有通信篇。本文使用单进程 autograd 验证尺度关系，未运行通信；梯度累积、特殊归约和不同目标权重需要另核对。

## 网页与版本边界

当天检索通过官方 PyTorch 文档和原始 arXiv 页面。SDPA 文档的缓存 acquired_at 是 2026-11-23，论文两次读取均为 2026-10-07；实际 Git fetch 日期为 2026-11-26。搜索也返回 NVIDIA Megatron 资料，但本文没有把它作为实现结论，不需要为未讨论的引擎增加源码缓存或引用。

成本表使用合成长度做整数算术：输入槽与平方面积分开，允许的因果位置对也单列。没有把代理面积当成实测 FLOPs、显存或时间。图源与 CPU 数值用于语义验证，没有 GPU 图像或时间 trace。
