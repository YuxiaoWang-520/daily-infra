# Orca 原论文阅读记录

- **题名**：Orca: A Distributed Serving System for Transformer-Based Generative Models。
- **作者**：Gyeong-In Yu、Joo Seong Jeong、Geon-Woo Kim、Soojeong Kim、Byung-Gon Chun。
- **出版版本**：USENIX OSDI 2022，2022 年 7 月，会议论文集 521—538 页；使用出版方正式 PDF，未把后续实现当作论文原版。
- **定位**：[官方论文页](https://www.usenix.org/conference/osdi22/presentation/yu)、[正式 PDF](https://www.usenix.org/system/files/osdi22-yu.pdf)。
- **访问**：2026-10-29，实际完整下载并用 `pdftotext` 提取；文件 3,906,936 字节，含封面共 19 页。命令、退出码、哈希见 [paper-acquisition.json](paper-acquisition.json)。本地研究缓存 `.cache/research14/` 不进入 Git；归档自己的阅读笔记，不重新分发论文全文或图。

## 实际阅读的机制与定位

第 3 节、Figure 3／4（会议页 524—525）对比整批执行与逐轮交互。正文的四请求例子自行选数，只借此解释换入边界，没有复现论文实验。

Figure 5 与第 3 节后半（526 页）讨论 selective batching：逐 token 运算可以组合，Attention 需要保持请求边界。本文不实现论文的 Attention 执行器。

第 4.2 节、Algorithm 1（528 页）在选择数量之外检查 K/V 容量；论文的 `max_tokens` 表示最终总长度上限。当前 vLLM 的整段当前输入容量检查与分页分配不是这条未来最大长度预留规则，也不自动继承论文的公平性保证。

## 与实现及练习的对应边界

对应关系是“每轮选择 → 模型执行 → 更新请求”，当前路径见 [源码笔记](source-reading.md)。练习一固定每轮成本，练习二固定完整 Prefill 和严格队首阻塞；二者都比当前引擎简化得多。没有采用论文的性能倍率，没有复现其硬件、分布式流水或负载。
