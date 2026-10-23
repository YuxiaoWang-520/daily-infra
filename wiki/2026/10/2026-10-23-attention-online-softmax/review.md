# Attention 篇审查

2026-10-23。主代理完成技术与教学两轮自审；单一独立代码代理只读静态审查本篇 3 个 Python 文件，没有执行或 GPU 验证。

## 第一轮：论文、当前源码与实现

实际读取 Online normalizer v2 的第 2—3 节、Algorithm 3／Theorem 1；FlashAttention v2 的第 2.2／3.1／3.2 节、Algorithm 0／1、Appendix B.3／Algorithm 2。对照当前固定仓库的 C++ 行块入口、局部 QK／mask／重缩放／PV／归一化，以及 CuTeDSL Softmax 的在线状态和 rescale 方法。

保留实际差异：原论文循环层次与参考不同；参考保留未归一化 u、最后除 l；GPU 源码使用 exp2、缩放参数、低精度局部 P 与架构布局。当前 README 已有 FA4 CuTeDSL，未把 C++ 路径说成全部当前默认路径，也没有追称所有新路径 dispatch 都已分析。

实际发现与修正：

| 级别与来源 | 发现 | 修正与复验 |
| --- | --- | --- |
| MEDIUM，自审 | online_softmax 第二遍先 exp 再屏蔽，极大无效分数可内部溢出，最终填零会隐藏它。 | 增加真实失败回归，保存 [masked-exp-before-fix.txt](evidence/runs/masked-exp-before-fix.txt)，改为指数前屏蔽，最终 15 项通过。 |
| MEDIUM，独立代码审查 | 文案称内部永不形成全量得分形状，但大 tile 能覆盖整个小问题，现有大 tile 测试正好如此。 | docstring、正文、资料与练习说明统一给出 `min(Bq,Nq)*min(Bk,Nk)` 上界，明确没有独立持久 S／P，不对所有块配置保证收益。数值实现无需改变。 |
| MEDIUM，自审 | 任意全屏蔽、非方形 causal 不能自动对应所有 GPU API。 | 明确 True=keep、局部 top-left `j<=i`、全屏蔽零输出是本篇约定；源代码模板条件与任意稠密 mask 支持分开。 |

独立审查确认 m／l／u 更新、旧状态重缩放、safe_max、零分母、掩码交集与独立 PyTorch 对照一致；无 CRITICAL／HIGH 或其他必须修正项。最后文案修正由主代理核验，未称独立运行。

## 第二轮：教学表达

从三份信息的打分与混合开始，首次解释 Q／K／V、点积和 softmax，再引入二维形状、缩放、容量。先用 `[0,0,0]` 反例解释块内归一化不能直接拼，再用单项指数等式推导修正因子，最后增加 V 的加权分子。

`[1,2,3]` 的具体状态与 lab 对齐；Mermaid 以 Q 块、Key 循环和 m／l／u 更新表达完整回路。mask 与极端值放在基本递推之后，当前 FA4 作为理解同一不变量的进阶入口，没有用新架构术语替代基础推导。

两题分别训练归一化状态与 Attention 输出，题目、完整答案、命令、容差、未验证部分和内存假设齐全。大容量例子只计算整数，不实际分配巨大张量。

## 验证与结论

[15 项测试](evidence/runs/tests.txt) 覆盖独立 CPU softmax／SDPA、最大值增加、前块全屏蔽、整行屏蔽、causal 与显式交集、不同 tile／尾部、非连续、零维度、推理契约与中间溢出拒绝。[lab](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt)、[格式](evidence/runs/format.txt) 通过；实际命令与环境见 [summary.json](evidence/runs/summary.json)，完整性见 [archive-check.txt](evidence/runs/archive-check.txt) 与 [校验和](evidence/SHA256SUMS)。

初始 RED 与 masked-exp 修复前日志是历史失败，发布结论以最终完整测试为准。通过，无未解决 CRITICAL／HIGH。没有真实 FlashAttention／CUDA、训练反向、低精度硬件路径、HBM／allocator 峰值或性能结果。
