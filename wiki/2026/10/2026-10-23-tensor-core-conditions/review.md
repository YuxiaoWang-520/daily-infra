# Tensor Core 篇审查

2026-10-23。主代理技术／教学两轮自审，独立代理只读静态检查本篇 3 个 Python 文件，不包含 GPU 或独立执行。

## 第一轮：事实

实际读取 PTX ISA 9.4 的 warp-level mma 形状、类型顺序、片段分布、参与与舍入要求，以及 sm80 对指定 FP16 形式的条件。实际获取并分析刷新后的 Triton 加速重写和 MMAv2 类型选择／lowering，不停留在 tl.dot 名字或输入 half。

自审修正：MEDIUM，指令片段约束不能写成所有全局 shape 都必须整除；正文把人为全矩阵 padding 与库内部尾部处理分开。MEDIUM，源码存在某分支不能证明特定调用命中；把源码、编译内容、二进制关联和运行指标证据分层表达。LOW，`.aligned` 不能望文生义为指针对齐证明，补充实际 warp 参与语义。

独立代码审查未发现 CRITICAL／HIGH／MEDIUM／LOW 或需修正项。补齐、裁剪、舍入和异常范围与声明一致；它明确未进行硬件验证。

## 第二轮：教学

先从上篇的局部块乘接到硬件完成方式，再解释一条具体指令；随后区分三种类型、数一次 padding 代价，再走源码与验证方法。没有开篇罗列全部矩阵指令家族或拿新架构缩写代替解释。图解是接口到硬件的层次关系，指令 FLOPs 没有再乘线程数。

两题分别处理几何成本和输入舍入，完整可运行；把硬件验证放在明确的扩展范围。保持“用了”与“用满”的证据区别。

## 验证与发布

[9 项测试](evidence/runs/tests.txt)、[诊断断言](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt)、[格式](evidence/runs/format.txt) 通过。实际命令和环境见 [summary.json](evidence/runs/summary.json)，资料完整性见 [archive-check.txt](evidence/runs/archive-check.txt) 与 [校验和](evidence/SHA256SUMS)。初始 RED 是答案模块实现前的历史失败。

通过，无未解决 CRITICAL／HIGH。没有目标 GPU、实际 MMA／SASS 执行、硬件利用率或性能结果；CPU 舍入／padding 验证不代表 Tensor Core 数值复现。
