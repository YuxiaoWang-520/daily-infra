# 运行练习

在本篇目录中，使用已有 Python 3.10+、PyTorch 环境运行：

```bash
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/run_lab.py
```

先做正文题目，再看 `answers.py`。`cached_attention` 是单序列、单 head 的 CPU FP32／FP64 连续追加参考；True=keep 的绝对位置 mask 在内部构造，空新块返回空输出。它不修改调用者输入或历史缓存，使用 `torch.cat` 分配新缓存，**会复制历史**，不能作为生产缓存性能方案。

`TinyDecoder` 是随机固定权重的 CPU FP64 模型，默认两层。包含绝对位置嵌入、逐 token LayerNorm、单头因果 Attention、残差、ReLU FFN 和词表投影；没有 tokenizer、预训练能力、batch、padding、RoPE、GQA、dropout、训练、EOS、投机生成或分页。`full` 用独立的 CPU SDPA 对照，不调用增量 Attention。

`greedy_generate` 固定生成给定数量的 token。最后一次采样的 token 尚未再次送入模型，因此缓存长度是 `P+G-1`；`max_length` 限制已送入模型的位置，返回序列可以多出最后一个尚未送入的 token。零生成次数不做 forward。

缓存必须来自相同且未修改的模型、相同前缀和位置约定；shape 检查不能证明这个语义前提。参考对象的字段不可重绑定，Tensor 内容仍可被调用者改写，不提供防篡改保证。保存旧缓存和生成分支时应尊重这个所有权约定。

测试对照所有新位置的 logits 和每层完整 K／V，覆盖单 token、多 token 块、位置偏移、历史不变性、空块、输入非连续、输入所有权、上下文边界及非法参数。容差见测试；没有执行 vLLM 或 GPU。`lab.json` 中的 18→6 和 52→21 是明确定义的工作量账本，不能读成加速比。
