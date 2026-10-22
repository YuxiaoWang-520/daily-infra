# 参考资料、源码快照与证据对应

本篇研究的是普通 strided Tensor 的语义与实现，没有引入独立的论文算法。主要依据为 PyTorch 官方源码、API 文档与实际 CPU 验证；没有用不相关论文补充引用数量，也没有把 Attention 的维度重排描述成完整 Attention 算法。

## 1. 三份版本证据分别回答什么？

| 证据 | 具体版本 | 可支持的结论 |
| --- | --- | --- |
| 本次实际获取的官方源码 | `main`，`c9d49411dc9c251a6f9812799112e92532748eec` | 该快照下的元信息、连续性、转置、view／reshape／contiguous 实现 |
| 本次浏览的官方 API 文档 | `stable` 当时重定向至 `2.14`；正文使用固定 `2.14` URL | API 的公开语义；不作为本机安装版本证明 |
| 本机运行环境 | PyTorch `2.9.1`，runtime git `5811a8d7da873dd699ff6687092c225caffcf1bb` | 本文例子、练习和测试在该 CPU 环境下的实际结果 |

官方仓库：[pytorch/pytorch](https://github.com/pytorch/pytorch)。默认分支 `main` 通过仓库 API 核验，分支头通过 [branch API](https://api.github.com/repos/pytorch/pytorch/branches/main) 获取。固定快照的 commit 时间为 `2026-10-22T04:36:24Z`；归档核验时间为 `2026-10-22T07:09:04.266Z`，即北京时间 `15:09:04.266`。

CLI clone／ls-remote 最初未完成，已停止；本篇改用已连接的 GitHub 官方接口，按完整 SHA 实际获取源文件。这里保存的是真实文件快照，**没有声称已完成整仓 Git clone、编译主分支或运行最新版本**。获取方式、SHA 和文件清单见 [source-manifest.json](evidence/source-manifest.json)。

## 2. 核心论断对应哪些源码？

| 编号 | 核心论断 | 官方固定版本定位 | 本地归档 |
| --- | --- | --- | --- |
| S1 | Storage 与 sizes／strides、offset 分开保存；首元素指针考虑元素大小与 offset | [`TensorImpl.h`：成员 2995—3037，指针 1819—1831](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/c10/core/TensorImpl.h#L2995-L3037) | [TensorImpl.h](evidence/upstream/c10/core/TensorImpl.h) |
| S2 | 普通 transpose 交换 size 与 stride，并建立视图 | [`TensorShape.cpp::transpose`，3739—3770](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/aten/src/ATen/native/TensorShape.cpp#L3739-L3770) | [TensorShape.cpp](evidence/upstream/aten/src/ATen/native/TensorShape.cpp) |
| S3 | 默认连续性逆序检查；空 Tensor 为 True，单例轴忽略 | [`Contiguity.h::_compute_contiguous`，13—33](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/c10/core/Contiguity.h#L13-L33) | [Contiguity.h](evidence/upstream/c10/core/Contiguity.h) |
| S4 | 常见具体尺寸路径查询维护的连续性标志，不是每次重新遍历 | [`TensorImpl.h::is_contiguous_default_impl`，878—921](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/c10/core/TensorImpl.h#L878-L921) | [TensorImpl.h](evidence/upstream/c10/core/TensorImpl.h) |
| S5 | view 要计算兼容 stride，失败报错，成功复用 storage | [`view_impl`／`view_impl_symint`，4007—4030；alias helper，1931—1958](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/aten/src/ATen/native/TensorShape.cpp#L4007-L4030) | [TensorShape.cpp](evidence/upstream/aten/src/ATen/native/TensorShape.cpp) |
| S6 | computeStride 按可合并区段匹配新 shape；保留空张量、单例轴及符号关系处理 | [`TensorUtils.cpp::computeStride_impl`，327—407](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/aten/src/ATen/TensorUtils.cpp#L327-L407) | [TensorUtils.cpp](evidence/upstream/aten/src/ATen/TensorUtils.cpp) |
| S7 | reshape 对不兼容布局执行连续格式 clone 后再构造 view | [`reshape_symint`，1963—2013；`reshape`，2034—2070](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/aten/src/ATen/native/TensorShape.cpp#L1963-L2013) | [TensorShape.cpp](evidence/upstream/aten/src/ATen/native/TensorShape.cpp) |
| S8 | contiguous 已满足格式时返回自身，否则 clone 指定格式 | [`TensorProperties.cpp::contiguous`，137—147](https://github.com/pytorch/pytorch/blob/c9d49411dc9c251a6f9812799112e92532748eec/aten/src/ATen/native/TensorProperties.cpp#L137-L147) | [TensorProperties.cpp](evidence/upstream/aten/src/ATen/native/TensorProperties.cpp) |

正文只摘录必要步骤，完整原始文件归档在上述目录。上游许可证保存在 [LICENSE](evidence/upstream/LICENSE)，相关原始版权信息随文件保留。文件校验和见 [SHA256SUMS](evidence/SHA256SUMS)。

## 3. 实际阅读的官方文档与笔记

访问日期均为 2026-10-22。

| 文档 | 核验位置 | 本篇阅读笔记 |
| --- | --- | --- |
| [Tensor Views（main）](https://docs.pytorch.org/docs/main/tensor_view.html) | view 与 base 共享数据；transpose 示例；reshape 的共享限制 | 创建视图不搬运元素。修改视图影响共享数据；reshape 的调用者不能假定总共享或总独立。对应第 2、5 节。 |
| [`Tensor.view`（2.14）](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html) | shape／stride 兼容条件与示例 | 元素数相同是必要条件；跨相邻轴合并还有步长关系。不能简化成“只有连续 Tensor 可以 view”。对应第 4 节。 |
| [`Tensor.contiguous`（2.14）](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.contiguous.html) | 已满足 memory format 返回自身 | 连续化不保证独立副本。对应连续切片与内存持有分析。 |
| [`torch.clone`（2.14）](https://docs.pytorch.org/docs/2.14/generated/torch.clone.html) | preserve format 与稠密非重叠输入 | 稠密转置的 clone 可以保留原 stride；独立副本与默认连续是两件事。 |
| [Channels Last 官方教程](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) | Memory Format API 与 Format checks | 保持 NCHW 逻辑 shape，可以通过 stride 表达不同物理排列。仅借此说明语义，不引用教程里的硬件性能比例。 |

这些是本地阅读笔记，不是完整网页镜像。原页面保持外部精确链接；随文原始实现快照和实验记录确保核心结论不只依赖网页链接。

## 4. 本机版本与当前源码的比较

按运行环境公开的 git SHA，另外获取并归档了：

- [运行版本 Contiguity.h](evidence/runtime-source/c10/core/Contiguity.h)。
- [运行版本 TensorUtils.cpp](evidence/runtime-source/aten/src/ATen/TensorUtils.cpp)。

具体尺寸的 `_compute_contiguous` 主体与当前快照一致，均先处理 numel 为 0、忽略 size 为 1 的轴，再逆序比较期望 stride。`computeStride` 的核心区段匹配仍能解释本文具体整数例子。

两份源码并不完全相同：当前快照对符号维度的判断使用 `TORCH_GUARD_OR_TRUE`／`TORCH_GUARD_OR_FALSE` 等保守处理，Contiguity 中也有符号判断调整。本文没有验证这些符号执行路径；不能从 CPU 2.9.1 的测试通过推出当前主分支全路径一致。

## 5. 教学代码与实验如何对应？

| 内容 | 推导依据 | 实际验证 |
| --- | --- | --- |
| 地址模型 | storage 起点 + offset + 索引与 stride 的点积 | 转置、步进切片、裁剪和非零 offset 与 PyTorch 的逻辑值对照 |
| 默认连续性模型 | `_compute_contiguous`，限具体非负尺寸与 stride | 标量、空 Tensor、单例轴、零 stride 与普通布局对照 |
| pack_logical | 行优先逻辑枚举 + 地址模型 | 与 PyTorch reshape 展平值对照，另验证重复读取及存储越界 |
| 拆头／合头 | 明确坐标等式与连续区段关系 | 逐坐标验证、错误 reshape 反例，以及两种 BHSD 布局的共享／复制差异 |
| 32 MiB／64 MiB | 元素数 × 元素字节数；一读一写的理想模型 | 仅整数算术估算，没有 GPU 性能或真实流量测量 |

`pack_logical` 是 Python 语义模型，不是实际 copy kernel；它的循环、内存开销和错误处理不能代表 ATen 实现。CPU Tensor 的 storage 指针比较只用于本实验的已知来源，不能作通用重叠判断。

实际命令、版本和退出码见 [运行汇总](evidence/runs/summary.json)，输出见 [lab.json](evidence/runs/lab.json)，审查见 [review.md](review.md)。
