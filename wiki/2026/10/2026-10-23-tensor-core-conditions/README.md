---
title: "Tensor Core 怎样使用，什么条件决定它是否有效？"
date: "2026-10-23"
timezone: "Asia/Shanghai"
status: published
topics: [Tensor Core, MMA, dtype, padding, 指令证据]
prerequisites: [浮点格式, GPU 线程, GEMM 分块]
source_checked_at: "2026-10-23，固定 Triton 源码与 PTX ISA 9.4"
---

# Tensor Core 怎样使用，什么条件决定它是否有效？

> Tensor Core 是专门服务矩阵等运算的硬件执行资源；程序通过相应指令和库／编译器路径使用它。输入改成 FP16 只是改变了一个条件。实际路径还取决于目标硬件、计算类型、矩阵片段、布局和实现选择，收益则要进一步看整个算子的成本。

## 1. 上一篇已经分块了，还缺什么？

上一篇把矩阵乘法拆成：拿 A 的一小块与 B 的一小块，更新 C 的一小块。我们仍没有规定“这块乘法怎样由硬件完成”。

最直接的想法是，用很多标量乘加逐个计算。另一类硬件可以专门完成一个矩阵片段的乘加，让线程协作提交成组的运算。Tensor Core 就属于这样的矩阵计算执行资源。

先把三个层次分开：

```mermaid
flowchart LR
    OP[框架调用：A @ B] --> IMPL[库或编译器选择实现]
    IMPL --> INS[生成矩阵指令或普通算术指令]
    INS --> HW[相应硬件单元执行]
```

`torch.matmul` 是操作接口，MMA 是矩阵乘加的指令／操作概念，Tensor Core 是硬件资源。它们不是同一个名字的三种写法。

因此，“用了 PyTorch 矩阵乘法”“输入是 half”“确实执行 Tensor Core 指令”“达到了很高的 Tensor Core 利用率”，是不同程度的结论，需要不同证据。

## 2. 一条矩阵指令在算多大的事情？

先看官方 PTX ISA 中的一种具体示例：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

不用一次记住所有单词。把它拆开读：

- `mma`：矩阵乘加，数学操作是 `D=A*B+C`。
- `m16n8k16`：这一操作的 A 为 `16×16`，B 为 `16×8`，C／D 为 `16×8`。
- `row.col`：该形式的 A、B 使用指定的矩阵布局约定。
- 末尾类型顺序是 **D、A、B、C**：两个乘法输入为 FP16，累加状态和目标为 FP32。

按照一次乘加 2 FLOPs 的计数，这份 `16×8×16` 矩阵乘加是 `4096 FLOPs`。它是协作操作的整体计数，不能再因为有 32 个线程就乘 32。

对这类 warp 级 MMA，矩阵片段分散在 warp 的线程寄存器里。不是某个线程独自拿着完整的三个矩阵、其他线程各再复制算一遍。[PTX ISA 9.4，warp-level mma 语义与示例](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

这个特定 FP16 `m16n8k16` 形式要求 `sm_80` 或更高的适用目标。其他类型、形状和架构有自己的条件，不能把它当作所有 Tensor Core 的唯一形状。

## 3. 三种 dtype，为什么必须分别问？

回想第 03 篇：把 `1.001` 存为 FP16，读回已经变成 `1.0009765625`。之后用 FP32 算，也不能恢复那一小段丢掉的信息。

实际矩阵路径至少要分清：

| 问题 | 一个可能的配置 |
| --- | --- |
| A／B 在 Tensor 中怎样存 | FP16 |
| 矩阵指令接受什么乘法输入，怎样保留部分和 | FP16 输入、FP32 累加状态 |
| 结果怎样写回 Tensor | FP16 或 FP32，取决于实现 |

例如，第 07 篇的 Triton 教程先保留 FP32 accumulator，最后显式转为 FP16。看到最后的输出 dtype，不能倒推出中间所有运算都用了 FP16；看到 FP32 accumulator，也不能声称输入保留了原来的 FP32 精度。

再举一个常见疑问：FP32 的 Tensor 就一定只做普通 FP32 乘法吗？也不一定。框架可以在允许的条件下选择较低内部精度的快速路径，但输出仍为 FP32。PyTorch 的 [set_float32_matmul_precision 文档](https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html) 明确区分内部计算精度与输出 dtype。

TF32 就是需要辨别的内部计算格式之一，不能只按 Tensor 的 `dtype` 字段判断是否使用。具体后端、配置和目标设备是否采用它，要看实际生成或调用路径；不要把某个版本的配置默认值当成所有环境的默认值。

还有一层数值边界：PTX 对上述某些矩阵形式没有规定全部累加顺序、舍入与非正规数处理细节。**“FP16 输入、FP32 累加”并不足以规定逐 bit 的结果。** 本篇 CPU 练习因此不自称 Tensor Core 数值模拟器。[PTX mma 的 Precision and rounding](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

## 4. 形状不整齐，是不是就完全不能用？

假设我们选择按 `16×8×16` 的几何片段来组织一份教学实现，而任务是：

```text
A: 17×17，B: 17×9，C: 17×9
```

可以把 M／N／K 分别向上补到 32／16／32，补零后计算，再裁剪回 `17×9`。

为什么补零不改变有效结果？新增的 K 位置有零乘积；新增的行列对应额外输出，最后丢弃。这里说的是实数代数与有限输入的补零思路；不同计算分组下仍不保证浮点 bitwise 相同。

代价也能算：

\[
\text{工作量比例}=\frac{M' N' K'}{MNK}
=\frac{32\times16\times32}{17\times9\times17}\approx6.299
\]

原本按约定计 `5202 FLOPs`，整块补齐账本变为 `32768 FLOPs`，还需要额外容量、填零和裁剪。即使补齐后单次矩阵片段算得快，整体也不一定划算。

这是**人为整矩阵补齐的比较模型**，不是说所有库都要求用户把全局维度补成这些倍数。库可以内部处理尾部、选其他 tile 或混合路径；具体实现可能只在边缘承担一部分额外工作。

同样，形状整齐也不充分。存储布局、leading dimension、指针／加载对齐、shared memory 描述符和片段分发都有各自约束。PTX 指令名里的 `.aligned` 还有 warp 一致参与语义，不能望文生义地把它当成“输入指针已满足某个字节对齐”的证明。

## 5. 当前编译器在哪里检查这些条件？

我们实际拉取了 Triton 的矩阵教程、加速重写规则和 NVIDIA MMA lowering，固定 `main` 为 `1ec1eee840b8a35b15101e5ab9a4388e905291f3`。下面是一条可定位的实现分析，不代表已经编译或运行一个 GPU kernel。

### 5.1 从 tl.dot 到候选矩阵路径

教程里的 `tl.dot` 是较高层操作。编译器还要根据设备与操作条件改写。

[AccelerateMatmul.cpp 第 44—67 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L44-L67) 的 `getMMAVersionSafe` 按 compute capability 排候选，再调用 `supportMMA` 检查操作是否支持。

[第 501—538 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L501-L538) 的一个 `BlockedToMMA` 重写还检查目标、编码与类型，并存在拒绝路径。比如该规则对较旧目标注明退回 FMA，也为 FP64 选择额外限定。

这说明“硬件存在某能力”和“这版编译器、这份操作采用它”要分别检查。这个规则不是所有后端和所有新架构路径的完整列表，不能把一个 `return failure()` 泛化为整个库完全不支持某类计算。

### 5.2 已选中的类型，如何走向具体指令形式？

[MMAv2.cpp 第 291—337 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L291-L337) 的 `getMmaTypeDot` 分别读取 A、B、D 的元素类型；FP32 输入还检查 `InputPrecision::TF32`。这就是前面“不能只看输出 dtype”的代码证据。

[第 593—619 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L593-L619) 继续组织 NVVM 类型与矩阵形状，并在适用分支使用相应指令字符串。MMA operand 的布局与寄存器打包也由更前面的转换代码处理，不能把任意线性数组直接解释成每个 lane 的片段。

源码分析证明的是“这些条件与构造确实存在”。它还没有证明你的特定调用命中了其中哪条分支。

## 6. 怎样证明一次真实调用用了它？

先固定输入 shape、dtype、stride、设备、精度配置与库／编译器版本，再确认 profiler 捕获的是这次调用的具体 kernel。

接着检查该 kernel 对应的中间代码和二进制。PTX 是虚拟指令，SASS 是目标机器指令；PTX 中出现矩阵指令说明编译内容包含相应操作，不能仅凭一份不确定是否执行的缓存文件就断言线上路径已使用。目标二进制、调用关联及必要的硬件指标要彼此对应。

对自己的 CUDA 二进制，可用官方 `cuobjdump --dump-sass` 检查目标代码；对 Triton／库生成的 kernel，需要先取得实际命中内核的二进制，而不是反汇编另一个示例。[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

当前 PTX 还定义 `wgmma`、`tcgen05` 等不同矩阵指令家族，其协作范围、存储和同步方式并不都等同旧的 warp 级 MMA。本文用旧形式建立直觉，并通过当前编译规则说明选择仍在演进；排障时必须按目标家族继续读对应 ISA。[PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)

即使确认用了矩阵硬件，也还要回到 Roofline：数据供给够不够？任务是否太小？有没有格式转换、padding、同步和输出处理？“用了”不等于“用满”，更不等于端到端一定快。

## 7. 动手练习

### 练习一：补齐的成本要先数清楚

实现 `padding_plan(m,n,k,tile)`。输入非负维度和正整数三元 tile，输出补齐维度、原始／补齐 FLOPs 与比例。它不判断真实硬件支持。

验收：`(17,9,17)`、tile `(16,8,16)` 得到 `(32,16,32)`、5202、32768；已对齐形状比例为 1；原任务零工作时比例返回 `None`，不除以零。

<details>
<summary>详解与答案</summary>

每维用 `ceil(d/t)*t`，整数写法是 `((d+t-1)//t)*t`。把三维相乘再乘 2，得到约定 FLOP 数。

完整 [answers.py](code/answers.py) 拒绝布尔、负维度和非正 tile。`work_ratio` 只比较规定的补齐方案，不计算转换成本、访存或执行时长；不能当自动调优的性能评分。

</details>

### 练习二：FP32 计算能恢复输入精度吗？

实现 `quantized_padded_matmul`：接受 CPU FP32／64 输入，先转 FP16 或 BF16 保存再提升为 FP32，补零计算、裁剪返回 FP32。小规模、推理用途，不修改输入；拒绝非有限输入和存储转换溢出。

用 `[[1.001]] @ [[1.0]]` 观察两种存储结果，再用非方阵与“量化后的输入直接做 CPU FP32 @”比较。不能拿原始输入参考与量化输入参考混成同一种误差。

<details>
<summary>详解与答案</summary>

先存储舍入，再 FP32 运算，1.001 分别得到 `1.0009765625` 和 `1.0`。这说明精度损失可以发生在计算之前。完整实现 [answers.py](code/answers.py) 使用 CPU `@`，**不模拟 Tensor Core 内部乘积、累加顺序或舍入**；FP32 累加本身也可能溢出。

最终裁剪后克隆为连续输出，避免只返回保留整个补齐存储的视图。练习额外分配了补齐矩阵，是看清代价的参考实现，不推荐用它替代成熟 GEMM 库。

```bash
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/run_lab.py
```

已有 Python 3.10+、PyTorch 即可。实际 [9 项测试](evidence/runs/tests.txt) 通过，涵盖奇数、非连续、空维度、无效输入和存储溢出；[输出](evidence/runs/lab.json) 只含 CPU 数值和计数。没有 Tensor Core 实机指令或速度结果。

</details>

## 8. 面试官继续追问

| 层次 | 问题 | 回答要点 |
| --- | --- | --- |
| 基础 | FP16 Tensor 一定使用 Tensor Core 吗？ | 只是存储类型条件，仍需硬件、操作、布局与实现路径。 |
| 数值 | FP32 输出能证明全程完整 FP32 输入精度吗？ | 不能，内部路径和输入舍入需要分别说明。 |
| 指令 | m16n8k16 的 FLOPs 是否乘 warp 线程数？ | 4096 是该协作矩阵操作整体计数，不再乘 32。 |
| 源码 | getMmaTypeDot 为什么同时读 A、B、D？ | 乘法输入与累加／输出相关类型共同决定指令形式。 |
| 方案 | shape 不整齐应否总由用户全矩阵补齐？ | 先看库处理能力与代价，整矩阵补齐可能浪费大量计算和容量。 |
| 验证 | 有矩阵指令就能说达到硬件峰值吗？ | 还需实际命中、利用率和时间证据，并分析全算子及端到端开销。 |

本篇是 ISA／编译路径和补零基础，不附会特定新论文算法。数值背景见第 03 篇及其混合精度论文笔记；真实矩阵指令正确性和性能留待目标 GPU 验证。下一篇研究另一种优化：把多个步骤合进一个 kernel，什么时候反而更慢？

资料：[references.md](references.md)。审查：[review.md](review.md)。前置：[GEMM 分块](../2026-10-23-gemm-tiling/README.md)。返回 [Wiki 索引](../../../index.md)。
