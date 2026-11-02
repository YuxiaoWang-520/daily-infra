# 原始论文阅读笔记

访问日期：2026-11-02（Asia/Shanghai）。阅读的是原始论文固定 v1，不把 2023 年论文中的实现当成当前默认分支。

## 版本与来源

- 题名：*Efficient Memory Management for Large Language Model Serving with PagedAttention*。
- 作者：Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica。
- arXiv：`2309.06180v1`，提交于 `2023-09-12T12:50:04Z`。
- [arXiv 官方元数据](https://arxiv.org/abs/2309.06180v1)、[固定版本全文 HTML](https://arxiv.org/html/2309.06180v1)、[arXiv DOI](https://doi.org/10.48550/arXiv.2309.06180)。
- 论文标注 SOSP 2023（2023-10-23 至 10-26），会议 DOI 为 [10.1145/3600006.3613165](https://doi.org/10.1145/3600006.3613165)。
- 实际阅读：§4.2、§4.4 的 Parallel sampling／Shared prefix，Figure 8、Figure 10 对应文字，§6.4 的实验设置与边界。

## 与本篇主问题直接相关的机制

§4.2 用逻辑块到物理块的映射管理 KV。§4.4 说明多个序列可引用相同物理前缀，引用计数记录使用者；修改共享尾部时需要保护其他使用者。Shared prefix 段讨论预先保存共同输入状态、让请求接着处理自己的任务输入。

这支持“共享的是计算状态和物理数据，而非仅仅记住一段文本”的解释。Figure 8 的写时复制属于论文中的共享修改场景；本篇练习只管理不可改写的完整块，没有实现该算法。

§6.4 使用特定模型和翻译任务检验共享前缀。我们只核对其场景与机制，不转载性能表，也不把其倍率作为当前 vLLM 或本机数据。

## 与当前代码的对应及区别

| 论文中的概念 | 当前代码对应 | 不能混同之处 |
| --- | --- | --- |
| 多请求共享物理前缀 | 块池、引用计数、请求逻辑映射 | 当前包含自动 hash 查找，不只是论文中的预定义前缀 |
| 引用释放与共享写入保护 | touch、free、可写判断与相应私有化路径 | 本篇没有执行 GPU 复制或实现完整写时复制 |
| 固定块便于管理 | 基础单组对齐路径 | 当前还有受条件约束的部分块与混合状态路径 |

本篇因果依赖推导、4-token 数字例子、容量手算、合成标签和两幅 Mermaid 都是自己重新组织的教学材料，不复制论文原图。

## 原文保存方式

通过网页工具实际读取官方完整 HTML 和元数据，未下载或声称处理了本地完整 PDF。仓库保存本阅读笔记、版本／章节定位与 [获取记录](paper-acquisition.json)，不额外复制整篇论文。HTML 标注 CC BY 4.0；本篇需要的实现证据另以带许可证的官方源码归档。
