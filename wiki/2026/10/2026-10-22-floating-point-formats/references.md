# 浮点篇资料、实现与实验边界

正文：[FP32、BF16 与 FP16](README.md)。源码获取时间、精确命令和哈希见 [source-manifest.json](evidence/source-manifest.json)。

## 1. 核心论断与一手证据

| 需要核验的内容 | 精确来源 | 本地证据与观察 |
| --- | --- | --- |
| FP16 的 11 位有效精度、最大 65504、正规／次正规值与 epsilon | [Half.h，`numeric_limits<c10::Half>`](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/torch/headeronly/util/Half.h#L730-L788) | [原文件](evidence/upstream/torch/headeronly/util/Half.h)，结合本机 `torch.finfo` 输出与输入转换测试核验。 |
| BF16 的 1／8／7 位分配、有效精度 8、编码范围与舍入 | [BFloat16.h](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/torch/headeronly/util/BFloat16.h) | [原文件](evidence/upstream/torch/headeronly/util/BFloat16.h)：文件开头格式说明、100 行附近的 nearest-even、后段 numeric_limits。 |
| 当前路径迁到 headeronly，不能只读旧 wrapper | [c10/util/Half.h](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/c10/util/Half.h)、[BFloat16.h](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/c10/util/BFloat16.h) | [旧 Half 入口](evidence/upstream/c10/util/Half.h)、[旧 BF16 入口](evidence/upstream/c10/util/BFloat16.h) 只是包含新路径及兼容内容。 |
| 缩放、反缩放、非有限检测与尺度更新的机制 | [torch/amp/grad_scaler.py](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/torch/amp/grad_scaler.py) | [原文件](evidence/upstream/torch/amp/grad_scaler.py)，阅读 `scale`、`_unscale_grads_`、`unscale_`、`_maybe_opt_step`、`update` 及状态检查。没有在本机运行 CUDA AMP。 |
| 数学等价不保证逐位相同，累加精度／计算路径会改变误差 | [numerical_accuracy.md](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/docs/source/notes/numerical_accuracy.md) | [原文件](evidence/upstream/docs/source/notes/numerical_accuracy.md)，结合自己的顺序与累加实验；不照搬有歧义的切片坐标示例。 |
| 混合精度训练三项策略的原始论证 | [Mixed Precision Training，v3](https://arxiv.org/abs/1710.03740v3) | [逐节论文笔记](evidence/paper-reading.md)、[获取记录与哈希](evidence/paper-manifest.json)，阅读 3.1—3.3 与 Figure 1。 |

上游许可：[LICENSE](evidence/upstream/LICENSE)。所有主分支代码固定为 `e149fba40596a8181f44e1f318da5fa10d609118`；逐文件获取，不是整仓 clone，也没有编译这个版本。

## 2. 官方文档

- [NVIDIA Floating Point and IEEE 754，当前页面标为 13.4](https://docs.nvidia.com/cuda/floating-point/index.html)：读取格式、运算与 FMA、点积顺序等部分，用于确认表示与舍入的基本模型。文章未宣称所有 BF16 路径都属于 IEEE 754 标准基本格式，也未把 C++ 通用转换分支当作实际 GPU 指令。
- [PyTorch 2.14 Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)：不同归约顺序、backend 和精度配置可能改变结果；dtype 与内部算术需分别确认。当前快照某段切片示例写成 `A.sum(-1)[0]` 与 `A[:,0].sum()`，分别是行与列，本篇未采用其为数学等价示例；自己的标量实验逐项核验。
- [PyTorch 2.14 AMP](https://docs.pytorch.org/docs/2.14/amp.html)：autocast 与 GradScaler 是不同职责；具体算子策略与设备有关，不等于把全模型直接 `.half()`。本篇解释机制，不给未经执行的 CUDA 训练代码贴上通过标记。

文档在 2026-10-22 实际访问。HTML 原文没有重新分发；保存自己的笔记及固定源码中的文档源文件，不声称主分支 Markdown 与 2.14 HTML 完全相同。

2026-10-26 补充核对：[RISC-V BF16 格式规范，固定 commit `67c1319d31e07fbb48ca7a6c0fc7448f4d3090ba`](https://github.com/riscv/riscv-bfloat16/blob/67c1319d31e07fbb48ca7a6c0fc7448f4d3090ba/doc/riscv-bfloat16-format.adoc) 与 [AMD XAPP1406，Revision 1.0、2026-01-28](https://docs.amd.com/r/en-US/xapp1406-aie-ml-fp-computation/Floating-Point-Numerical-Formats)，用于补充指数偏置 127、编码与真正指数的区别、完整位串解码。自己的阅读笔记和复算命令见 [偏置核对](evidence/bf16-exponent-bias.md)；新验证见 [CPU 位编码记录](evidence/runs/bf16-exponent-bias-2026-10-26.json)。本次没有刷新原文的 PyTorch 源码快照。

## 3. 可复算的实验结论

- `1.001` 经 FP16 存储为 `1.0009765625`，经 BF16 存储为 `1`；再转为 float64 只能显示这个已保存的值。
- `70000` 经 FP16 转为 inf；BF16 对应本机值为 `70144`。这是 CPU 转换实测，不证明所有设备的次正规数或转换行为。
- 1000 个存储为 FP16 的 `0.1`，高精度参考和为 `99.9755859375`，并非理想十进制 `100`。显式 FP16 逐步累加为 `105.1875`；本例 FP32 逐步累加等于量化输入参考和。
- 本篇没有用某个随机误差阈值证明所有算子或模型“兼容”。练习阈值由题目明确传入，比较的是具体存储值。

验收器也要避免自己制造误差。修复前的归一化比较曾对极端值错误通过，相对范数也丢失过小误差；最终小规模参考使用 Fraction 精确判定，以及独立 80 位 Decimal 上下文计算描述性相对范数。它不以速度为目标，不等价于当前 GPU 验证工具或 PyTorch `allclose` 的浮点边界实现。

## 4. 验证记录

环境：Python 3.10.13、PyTorch 2.9.1（`5811a8d7da873dd699ff6687092c225caffcf1bb`）、macOS arm64、CPU。没有 GPU、主分支构建或模型训练实验。

- [运行与环境汇总](evidence/runs/summary.json)、[原始输出目录](evidence/runs/)。
- [完整代码与运行命令](code/README.md)、[审查与修正](review.md)。
- [归档文件校验和](evidence/SHA256SUMS)。
