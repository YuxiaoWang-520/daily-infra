# 原论文阅读笔记

论文：**PyTorch Distributed: Experiences on Accelerating Data Parallel Training**。

作者：Shen Li、Yanli Zhao、Rohan Varma、Omkar Salpekar、Pieter Noordhuis、Teng Li、Adam Paszke、Jeff Smith、Brian Vaughan、Pritam Damania、Soumith Chintala。

固定版本：[arXiv:2006.15704v1](https://arxiv.org/abs/2006.15704v1)，首次提交 `2020-06-28T20:39:45Z`；[arXiv DOI](https://doi.org/10.48550/arXiv.2006.15704)。VLDB 2020 发表版本 DOI：[10.14778/3415478.3415530](https://doi.org/10.14778/3415478.3415530)。本篇固定阅读 arXiv v1，不混用两个版式的页码。

## 实际读取

2026-11-03 访问官方 [完整 HTML](https://arxiv.org/html/2006.15704v1)，同时下载固定 PDF、用 `pdftotext -layout` 提取，并渲染检查 PDF 第 5 页的 Figure 3 与 Algorithm 1。获取命令、退出码、原始字节和 SHA-256 在 [paper-acquisition.json](paper-acquisition.json)。

重点阅读 §2.2、§2.3、§3.2.2—§3.2.4、Algorithm 1 与 Figure 3；§3.2.4 在 PDF 第 6 页，Figure 3 与 Algorithm 1 在第 5 页。HTML 章节号与 PDF 一致。

## 和本篇的对应关系

- §2.2／§2.3：梯度同步先于本地更新；AllReduce 是各方参与的归约，不是 rank 0 独占结果。正文先由单参数目标推导全局梯度，不把所有 loss 分母假定为相同。
- §3.2.2／§3.2.3：桶把小梯度合并；就绪后可与剩余反向计算重叠。Figure 3 强调顺序失配和缺失梯度会破坏同步。本文的单通道时间递推是自行明确假设的教学模型，不是论文中的测量曲线。
- Algorithm 1：注册 hook、填桶、按序提交、最终等待，与当前 reducer 主线对应；当前还包含重建桶、配置分支及手动收尾等变化。不能照搬 2020 年“初始化后固定桶”的概括。
- §3.2.4：`no_sync` 为累积提供同步边界；正文另外推导 equal micro-batch 的 `/K` 和一般有效分母的区别。

论文研究对象是 PyTorch v1.5，硬件、库版本和工作负载均是当年的环境。本篇没有引用其吞吐或倍率作为当前性能结论，也没有将本机 CPU Gloo 与论文 GPU 实验比较。

## 原文归档与图源

HTML 标示 CC BY-NC-SA 4.0，发表版本还有自身许可说明。为避免将不同再分发／改编条件混在本仓库通用许可下，本篇仅归档原创阅读笔记、准确定位和获取 hash；PDF、提取全文及渲染图保存在忽略缓存，没有作为交付正文再分发。正文图示均为按自有数字例子绘制，非论文图的复制。
