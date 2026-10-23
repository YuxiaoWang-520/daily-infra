# GEMM 分块审查

2026-10-23。主代理完成技术与教学两轮自审；独立代理静态审查本篇 3 个 Python 文件，所有执行由主代理负责。

## 第一轮：技术

实际阅读 Goto 论文第 1—4 节相关模型、Figure 5、4.2.1—4.2.3，核对作者稿与正式 DOI 的版本差异。分析刷新后的 Triton 固定源码，从 program 映射、stride 地址到 K 累加与最终 store，未只引用教程首页。

实际自审修正：MEDIUM，若把有效元素计数直接对应源码实际加载，会漏掉 M／N 取模产生的冗余槽位；正文、参考笔记与练习范围已明确。MEDIUM，FP32 accumulator 不等于 FP32 输出，正文补上源码最后转 FP16，CPU 参考也声明不同类型范围。LOW，作者稿仍含 20YY 占位，参考页按实际稿件与正式出版元数据分别记录，没有杜撰 PDF 版本日期。

CPU 数值验证覆盖手算、FP64 奇数边界、FP32 非连续输入、空 M／N／K 与推理契约。模型测试核对非方阵有效读取、BK 变化、零输出块与只写零结果。

## 第二轮：教学

先计算一个输出，再看 2 段 K 的 2×2 输出，之后数输入读次数，再引入 BM／BN／BK、层级与源码。保留“BK 为何在比值里消掉”的自然追问，避免把所有 tile 维度扩大都说成复用更好。图源保存在正文 Mermaid，数值与 lab 一致。

代价通过累加器从 BM×BN 增长来解释，再讲并行度和边缘；没有用性能术语堆砌代替推导。两题有完整答案、边界和常见错误，CPU 参考没有被包装成高性能 GPU 实现。

## 运行与结论

[11 项测试](evidence/runs/tests.txt)、[诊断](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt)、[格式](evidence/runs/format.txt) 通过。实际环境与命令见 [summary.json](evidence/runs/summary.json)，资料完整性见 [archive-check.txt](evidence/runs/archive-check.txt) 与 [SHA256SUMS](evidence/SHA256SUMS)。初始 RED 是实现前真实失败。

独立代码审查未发现 CRITICAL／HIGH／MEDIUM，提出两项 LOW：公开 docstring 未直接写 inference-only，以及零输出块时 k_steps 仍非零。前者已补充；后者增加真实失败回归后改为零块返回零步，[修复前输出](evidence/runs/empty-steps-before-fix.txt) 保留，完整 11 项测试修复后通过。最后两处局部改动由主代理复核，不称为独立运行。

通过，无未解决 CRITICAL／HIGH。无 GPU 执行、shared memory／寄存器测量、实际指令或速度结论；CPU 验收不代替它们。
