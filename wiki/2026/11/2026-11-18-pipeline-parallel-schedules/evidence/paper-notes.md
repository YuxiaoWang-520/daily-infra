# 原始论文阅读笔记

访问日期为 `2026-11-18`，时区为 `Asia/Shanghai`。实际阅读原始 arXiv HTML，并核对摘要页的版本历史。两份原文均标注 arXiv perpetual non-exclusive license；本篇归档自己的笔记与推导定位，原文通过一手来源获取，未将全文再分发到 Git。

## GPipe

arXiv ID 为 `1811.06965`，首次提交于 `2018-11-16`，本次阅读 `v5`，版本日期为 `2019-07-25`。摘要页题名为 *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*，实际 HTML 版内题名为 *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism*。两处题名分别记录，以 ID 和版本定位。[摘要与版本历史](https://arxiv.org/abs/1811.06965)、[实际阅读原文](https://arxiv.org/html/1811.06965v5)

作者为 Yanping Huang、Youlong Cheng、Ankur Bapna、Orhan Firat、Mia Xu Chen、Dehao Chen、HyoukJoong Lee、Jiquan Ngiam、Quoc V. Le、Yonghui Wu、Zhifeng Chen。

阅读 §2.1—2.3 与图 2，检查按连续层划分、microbatch 梯度累积、一次更新和重算。本文自行排出三 stage、四 microbatch 的单位时间表，并把“保存份数”与“每份保存字节数”拆开计算。本文的 FIFO 反向依据当前 PyTorch 具体循环，容量表采用自编线性假设，未复述论文的实验性能。

## 同步 1F1B 与交错流水线

题名为 *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*，arXiv ID 为 `2104.04473`。首次提交于 `2021-04-09`，本次阅读 `v5`，版本日期为 `2021-08-23`。[摘要与版本历史](https://arxiv.org/abs/2104.04473)、[实际阅读原文](https://arxiv.org/html/2104.04473v5)

作者为 Deepak Narayanan、Mohammad Shoeybi、Jared Casper、Patrick LeGresley、Mostofa Patwary、Vijay Anand Korthikanti、Dmitri Vainbrand、Prethvi Kashinkunti、Julie Bernauer、Bryan Catanzaro、Amar Phanishayee、Matei Zaharia。

阅读 §2.2.1—2.2.2、图 3—4，以及 §3.1、3.2、3.4。核对同步 flush 的参数版本语义、预热／交替／排空和保存窗口。论文的气泡比例以理想计算时间为分母；本文另推导总时间分母，并用 `48+18=66 ms` 的假设例子区分二者。

当前 Megatron 非交错路径的队列与普通 1F1B 对应，PyTorch 将稳态第一份前向归入预热，局部计算次序仍可一致。交错多 chunk 和无 flush 调度只作为范围边界，本篇没有实现或验证它们。固定 batch 的三个 `b` 值及耗时表是本篇自己的预算，用于解释需要同时测量 microbatch 的实际计算成本。

返回 [正文](../README.md) 或 [参考资料](../references.md)。
