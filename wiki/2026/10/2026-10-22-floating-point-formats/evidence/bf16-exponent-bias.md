# BF16 指数偏置补充核对

核对日期：2026-10-26，Asia/Shanghai。范围为正文第 2 节的格式解释；本次没有更新原文冻结的 PyTorch 源码版本，也没有修改练习实现。

## 一手来源与阅读结论

1. [RISC-V BF16 格式说明](https://github.com/riscv/riscv-bfloat16/blob/67c1319d31e07fbb48ca7a6c0fc7448f4d3090ba/doc/riscv-bfloat16-format.adoc)，固定 commit 为 `67c1319d31e07fbb48ca7a6c0fc7448f4d3090ba`。本次读取完整格式文件，重点核对开头位图和格式参数表：BF16 分配 1 位符号、8 位指数和 7 位小数；表中偏置为 127，最小正规指数为 -126；FP32 使用相同的指数参数。该文件是格式规范资料，不是本机执行代码。
2. [AMD XAPP1406：Floating-Point Numerical Formats](https://docs.amd.com/r/en-US/xapp1406-aie-ml-fp-computation/Floating-Point-Numerical-Formats)，页面标注 Revision 1.0、2026-01-28。读取 FP32 和 BF16 两节：指数编码是无符号整数，正规数按编码减去 127 解读；BF16 保留 FP32 的指数编码，缩短小数部分。其分类表用于核对指数全零／全一的特殊含义。AMD 设备的次正规数处理属于硬件行为，不推广成所有 BF16 设备的行为。

以上是自己的阅读笔记，没有重新分发上游文档全文。原有 PyTorch 类型定义继续见 [BFloat16.h](upstream/torch/headeronly/util/BFloat16.h)。

## 如何解释 127，而不把约定说成必然？

- 先用 `0.5 = 1 × 2^-1` 说明需要负指数，再区分存储编码 `E` 与真正指数 `e`。
- 在这一偏置约定下，指数位数为 `k` 时取 `2^(k-1)-1`；BF16 的 `k=8`，因此取 127。这是编码约定，不是 8 位存储的唯一可能设计，更不是由 7 个小数位决定。
- 对正规数执行 `E=e+127`／`e=E-127`。保留编码 `0` 和 `255` 后，正规指数为 `-126～127`，不声称范围完全对称。
- 负指数缩小尺度；符号位决定数值正负。把两者分开，才能读懂正数 `0.5` 和负数 `-6.5`。

## 可复算的 CPU 验证

在仓库根目录使用已有环境运行；不需要安装依赖：

```bash
/Users/wangyuxiao/conda3/bin/python3 - <<'PY'
import torch

for value in [0.25, 0.5, 1., 1.0078125, 1.5, 1.9921875, 2., 3., 4., 6.5, 256., -6.5]:
    t = torch.tensor(value, dtype=torch.bfloat16)
    bits = t.view(torch.int16).item() & 0xffff
    s, E, f = bits >> 15, (bits >> 7) & 0xff, bits & 0x7f
    decoded = (-1) ** s * (1 + f / 128) * 2 ** (E - 127)
    assert 1 <= E <= 254
    assert decoded == value == t.item()
    print(f'{s} | {E:08b} | {f:07b} -> {decoded}')
PY
```

实际验证还包括最小／最大正正规值、最小正次正规值、零、无穷大，以及 `[2,4)` 内的向上步长。结果、环境和断言数见 [JSON 记录](runs/bf16-exponent-bias-2026-10-26.json)。这里只报告 CPU 读写与解码，没有测量 GPU、吞吐或硬件次正规数运算。
