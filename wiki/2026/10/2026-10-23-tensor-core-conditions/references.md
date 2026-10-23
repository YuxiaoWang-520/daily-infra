# Tensor Core 资料与实现笔记

2026-10-23，Asia/Shanghai。基础以当前官方 ISA 和实际固定编译器源码为证据；本机没有 CUDA，不提供执行或反汇编结果。

## 官方语义

- [PTX ISA 9.4，warp-level mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)：读了 D=A*B+C、片段跨 warp 分布、类型顺序 D/A/B/C、舍入说明、sync/aligned 的参与语义、m16n8k16 FP16 的 sm80 条件与示例。4096 FLOPs 是按 2mnk 自行计数，不是物理每周期吞吐声明。
- [PyTorch 2.14 set_float32_matmul_precision](https://docs.pytorch.org/docs/2.14/generated/torch.set_float32_matmul_precision.html)：内部精度与输出 dtype 不同。本文没有复述其复合 BF16 算法或声称本机采用，不据此引入未读论文。
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)：真实二进制检查入口。没有运行 cuobjdump，也没有虚构 cubin／SASS。
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 中还存在 wgmma 与 tcgen05 家族。本文只说明不能把 warp 级教学约定推广为所有家族，未声称完整分析最新家族实现。

## 源码链路

官方 `triton-lang/triton` 的 `main` 刷新后固定 `1ec1eee840b8a35b15101e5ab9a4388e905291f3`。实际按文件获取，未完整 clone 或构建。

| 问题 | 源码位置 | 本篇结论 |
| --- | --- | --- |
| tl.dot 之后还会选择吗 | [AccelerateMatmul，44—67 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L44-L67) | 目标候选与 supportMMA 检查均存在；未继续穷举 supportMMA 所有类型规则。 |
| 何时重写拒绝 | [501—538 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L501-L538) | 规则受目标、编码、类型影响；单规则不等于整个编译器能力表。 |
| 类型与内部精度怎样参与 | [MMAv2，291—337 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L291-L337) | A／B／D 类型与 TF32 input precision 分别检查。 |
| 接下来怎样构造 lowering | [593—619 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp#L593-L619) | NVVM 类型、形状与适用分支指令字符串；不是本次 kernel 已命中的证明。 |

归档：[矩阵教程](evidence/upstream/triton-lang/triton/python/tutorials/03-matrix-multiplication.py)、[完整加速规则](evidence/upstream/triton-lang/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)、[完整 MMAv2](evidence/upstream/triton-lang/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp)、[许可证](evidence/upstream/triton-lang/triton/LICENSE)。时间、命令、摘要见 [manifest](evidence/source-manifest.json)。

## 教学简化与验收

padding_plan 是选定三维 tile 的几何成本模型，不是硬件支持判定器。量化补齐练习是 CPU storage 转换后 FP32 @，仅验证输入信息损失和补零代数，未模拟 MMA 精度细节。零填充保持实数代数，不保证不同实现逐 bit 相同。

没有特定论文算法需要额外引用；格式／混合精度背景已在 [第 03 篇论文笔记](../2026-10-22-floating-point-formats/references.md) 归档。当前编译器源码与官方 ISA 是本篇实现事实的直接来源。

[运行环境](evidence/runs/summary.json)、[测试](evidence/runs/tests.txt)、[诊断](evidence/runs/lab.json)、[校验和](evidence/SHA256SUMS)、[链接与行号核验](evidence/runs/archive-check.txt)。
