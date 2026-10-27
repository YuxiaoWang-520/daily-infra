# 原始论文阅读与实现核对

阅读日期：2026-10-27，Asia/Shanghai。原文下载、提取命令、哈希与一次超时记录见 [获取清单](paper-acquisition.json)。全文保留在本地研究缓存，仓库只归档自写笔记与准确定位。

## MQA：共享什么？

Noam Shazeer，*Fast Transformer Decoding: One Write-Head is All You Need*，首次提交于 2019-11-06，实际读取 arXiv `1911.02150v1`，[原文](https://arxiv.org/abs/1911.02150v1)，DOI：`10.48550/arXiv.1911.02150`。

阅读第 2.2—2.4 节与第 3 节，重点核对增量 Attention 的状态和 MQA 伪代码：Query 保留 head 轴，K／V 及其投影去掉多份 head。论文的动机是增量解码的反复读取；它的速度、质量结果有自己的实验条件，不是本文 CPU 结论。

对应本文：容量按 K 和 V 的历史轴数数；参考函数的 MQA 是 `Hkv=1`、多个 Query 共享同一组输入。原文伪代码说明缩放可折入投影，本篇显式除以 `sqrt(D)`，与独立 SDPA 的默认缩放一致。未把原文 TensorFlow 伪代码当作已执行程序。

## GQA：在两个端点之间怎样分组？

Joshua Ainslie、James Lee-Thorp、Michiel de Jong、Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai，*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*，2023，EMNLP，4895—4901 页，DOI：`10.18653/v1/2023.emnlp-main.298`，[会议原文](https://aclanthology.org/2023.emnlp-main.298/)。arXiv 首次提交为 2023-05-22，当前页面另列 2023-12-23 的 v3；本次实际读的是 ACL 会议 PDF，未混同其二进制版本。

阅读第 2.1—2.2 节、Figure 1／2 及第 3.1—3.3 节。PDF 第 2 页的 Figure 2 已渲染检查：一组 Query 对应一组 Key 和 Value，分组数量为 KV head 数；论文的 `G` 是组数，本文写作 `Hkv`，组内 Query 数为 `Hq/Hkv`。端点分别是 MHA 与 MQA。

转换涉及同组 K／V 投影矩阵平均及继续训练，不能解释为推理缓存上的无损操作。原实验采用指定 T5 设置，本文没有复现训练或质量评测。正文没有引用论文速度表作为当前 GPU 的性能承诺。

对应实现：vLLM Llama 与 QKV 投影分别维护 Query／KV head；Attention 规格按 KV 计数，后端保留 Query／KV 的比例。自己的 einsum 参考和展开 KV 后的 SDPA 对照检验分组语义，不复现论文的 checkpoint 转换训练。

## 图源与实验范围

正文的 4Q／2KV Mermaid 为自绘关系图，不复制论文图像；所有容量数字由本文选择的教学配置推导。理论容量、CPU 语义实验、源码静态分析与论文实证分别记录，不互相替代。
