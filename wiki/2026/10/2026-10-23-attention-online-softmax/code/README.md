# 在线 softmax 与 Attention 练习

先读正文第 12 节题目，再看 [answers.py](answers.py)。已有 Python 3.10+、PyTorch 即可运行：

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 run_lab.py
```

预期 15 项通过。小例子分数 `[1,2,3]`、Value `[10,20,30]`，输出约 25.7521。lab 还输出 m／l／u 换刻度前后状态；8192 长度的容量数只是整数计算，没有分配 8 GiB 张量。

`online_softmax` 两遍，第一遍在线最大值／分母，第二遍写概率。`tiled_attention` 按 Q／K 块维护 m／l／u，不另持久存全量 S／P。单个 tile 可以覆盖整个问题，分数上界是 `min(block_q,Nq)*min(block_k,Nk)`；多个常数份临时量、输入转换、完整输出和可选完整 mask 仍需容量。

范围：CPU FP32／64 输入、内部与输出 FP64，二维单 head、推理前向，无 dropout／反向／缓存。True=keep，causal 使用本地 `j<=i`，显式 mask 与 causal 取交集。空或全屏蔽行零输出是约定；拒绝 NaN／Inf、分数与未归一化分子溢出，即使某些极端输入的数学归一化结果仍有限也可能报错。

测试独立比较 CPU PyTorch softmax／SDPA，固定小 FP64 案例容差 `1e-12`。不代表 FA2／FA4／GPU 低精度路径的逐 bit、数值、性能或内存验收。[实际日志](../evidence/runs/)。
