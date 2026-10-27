# 运行容量与 GQA 练习

在本篇目录中使用已有 Python 3.10+、PyTorch 环境：

```bash
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/run_lab.py
```

本次使用 `/Users/wangyuxiao/conda3/bin/python3`，Python 3.10.13、PyTorch 2.9.1 CPU；环境和真实命令见 [运行汇总](../evidence/runs/summary.json)。没有新增依赖。

先做正文第 8 节，再读 [answers.py](answers.py)。

- `kv_cache_bytes`：按各请求已经缓存的 token 数求和，返回 K 与 V 的整数内容字节数。假设各层相同、K／V 同维、相同缓存 dtype，无共享、分页余量或元数据。
- `kv_heads_per_rank`：只模拟已核对的 vLLM Llama head 分片／复制约束。调用者另外检查 Query head 可被 TP 整除；不覆盖其他并行方式或缓存架构。
- `grouped_attention`：规则 batch 的 CPU FP32／FP64 参考，输入必须有限，尺寸为正。Query 头按相邻分组共享 KV，`query_start` 指明绝对位置，KV 位置从零开始。输入不被修改，非连续输入可用；不支持 padding、ragged batch、dropout 或半精度。

诊断中的单请求容量应为：MHA `4294967296`、GQA `1073741824`、MQA `134217728` byte。TP=4 的 MQA 每 rank 仍为 `134217728` byte，四个 rank 合计为 `536870912` byte。

小尺寸 FP64 诊断中，紧凑 K／V 内容为 `480` byte，重复展开为 `960` byte；分组参考与独立 SDPA 对照在测试容差内一致。[实际输出](../evidence/runs/lab.json)。

GQA 代码会物化分数矩阵与概率，Q reshape 可能复制；独立对照故意使用重复 K／V。没有测 GPU 或进程内存，也没有证明 checkpoint 转换无损；这些实现不作为高性能缓存管理方案。
