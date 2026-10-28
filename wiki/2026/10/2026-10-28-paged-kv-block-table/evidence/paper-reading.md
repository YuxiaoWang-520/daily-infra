# 原始论文阅读笔记

Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica：*Efficient Memory Management for Large Language Model Serving with PagedAttention*。arXiv `2309.06180v1`，2023-09-12；SOSP 2023，DOI `10.1145/3600006.3613165`。[arXiv v1](https://arxiv.org/abs/2309.06180v1)、[官方 HTML](https://arxiv.org/html/2309.06180v1)、[PDF](https://arxiv.org/pdf/2309.06180v1)。访问日期：2026-10-28。

实际读取第 3.1、4.1—4.3、5.1、7.1—7.2 节。web PDF 为 16 页：第 5—6 页定位 Figure 5／6 与式（4）；第 12 页定位 Figure 18。注意 PDF 页序与会议版连续页码不同。

- 论文以固定 token 数组成 KV 块，用每请求块表连接逻辑顺序与不连续存储；按需增长有尾页余量。原文 `B` 是页长，本篇为避免与 batch 混淆写为 `P`。
- 式（4）仍跨可见 Key 归一化，不是各页输出的任意平均；正文用自选两位置例子检查这一点，稳定分块算法链接已发布第 10 篇。
- Figure 6 的早期表带 filled 信息；不能据此断言当前 `BlockTable` 的每个 int32 项也存填充计数。当前实现将块号、长度、槽映射等分别传递。
- 原设计与当前代码都有分块读写主线；CPU gather、FIFO、从 `0` 起的可用块，以及不含共享／CP 的页池是本篇简化。
- 页长收益与成本须结合工作负载。没有把第 7 节的历史最优页长、kernel 延迟或吞吐数字用作当前默认值或本机实测。

获取说明：[记录](paper-acquisition.json) 如实保留三次不完整下载；实际阅读使用官方 HTML 和成功的 web PDF 访问。不完整本地 PDF 未作证据或提交；仓库保存自己的笔记与代码证据，不重发论文全文。Mermaid 为正文内自绘图。
