# 广播篇资料、源码与阅读笔记

正文：[广播与物化成本](README.md)。本篇只分析普通 eager strided Tensor 的具体尺寸；不分析稀疏、符号 shape、特殊 Tensor subclass 的完整路径。

## 1. 来源与获取方式

实际获取 PyTorch 官方 `main` 的逐文件源码快照，完整 SHA 为 `e149fba40596a8181f44e1f318da5fa10d609118`。通过已认证的 `gh api` 请求固定 ref，具体命令、UTC 获取时间、每个文件的 SHA256 和字节数保存在 [source-manifest.json](evidence/source-manifest.json)。这是逐文件获取，不是整仓 clone，也没有构建或运行该主分支。

| 关键论断 | 固定版本定位 | 本地归档与阅读观察 |
| --- | --- | --- |
| 广播尺寸从末尾对齐，1 沿用另一方，0 不应被简单 max 覆盖 | [ExpandUtils.cpp，`infer_size_impl`](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/aten/src/ATen/ExpandUtils.cpp#L17-L45) | [原文件](evidence/upstream/aten/src/ATen/ExpandUtils.cpp)：比较相等／单例，再选择尺寸；注释明确另一方可为 0。 |
| 扩展单例轴可以用零 stride 表达 | [同文件，`inferExpandGeometryImpl`](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/aten/src/ATen/ExpandUtils.cpp#L65-L116) | 对尺寸不等的轴先检查源尺寸为 1，然后设置目标尺寸与 stride=0。 |
| `expand` 返回视图，`repeat` 创建重复副本 | [TensorShape.cpp，`expand`／`repeat`](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/aten/src/ATen/native/TensorShape.cpp) | [原文件](evidence/upstream/aten/src/ATen/native/TensorShape.cpp)：`expand` 调 `as_strided`；`repeat` 创建目标再执行展开／复制相关步骤。 |
| 常见逐元素算子可直接在迭代器中表示广播 | [TensorIterator.cpp，`compute_shape`／`compute_strides`／`allocate_or_resize_outputs`](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/aten/src/ATen/TensorIterator.cpp) | [原文件](evidence/upstream/aten/src/ATen/TensorIterator.cpp)：输入零字节步长与输出配置分别处理；不应把显式 `expand` 例子当成所有算子的统一调用栈。 |
| 恢复广播前尺寸需要归约复用轴 | [ExpandUtils.h，`_sum_to`](https://github.com/pytorch/pytorch/blob/e149fba40596a8181f44e1f318da5fa10d609118/aten/src/ATen/ExpandUtils.h#L462-L503) | [原文件](evidence/upstream/aten/src/ATen/ExpandUtils.h)：归约多出来的前导轴及目标为 1 的轴，并恢复 shape；这里只核验工具函数，未追踪完整 autograd 生成链。 |

上游 [LICENSE](evidence/upstream/LICENSE) 一并保存。文件未改写，不把教学摘录伪装成完整的实现。

## 2. 官方文档阅读笔记

2026-10-22 实际打开的 `stable` 链接转到 PyTorch 2.14 页面，因此引用固定版本 URL；本机运行版本是 2.9.1，两者分开记录。

- [Broadcasting semantics，2.14](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html)：支持的运算按末尾尺寸对齐。文档的正常尺寸示例用于建立规则，空尺寸的精确选择另由固定源码和本机测试核验。也保存了本次主分支的 [broadcasting.md](evidence/upstream/docs/source/notes/broadcasting.md)，不声称它与 2.14 HTML 逐字一致。
- [Tensor.expand，2.14](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.expand.html)：扩展视图不分配重复数据，扩展轴可使用零 stride；有别名的批量写入可能不正确，写入前需考虑独立副本。正文只断言已验收的 `add_` 报错，不扩大为所有写操作。
- [Tensor.repeat，2.14](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.repeat.html)：明确复制数据。本机验收让 bias、repeat、expand 同时存活，再比较各自 storage 与值，避免空指针或分配地址复用的混淆。

文档的完整 HTML 未重新分发；保留自己的阅读笔记、准确 URL 和可再分发的源码／文档源文件。核心机制由固定源码支撑，不能只靠浮动网页。

## 3. 本篇推导与边界

### 平方距离为什么是 `[N,M,D]`？

第一个输入的坐标是 `(i,1,d)`，第二个是 `(1,j,d)`。广播后的每个 `(i,j,d)` 都需生成一个差值；沿 D 求和才得到 `(i,j)`。这直接从坐标对应推导，并不需要假设先复制输入。

`4096²×128×4=8 GiB` 是单块三维 float32 结果的载荷，不是进程峰值实测；最终 `4096²×4=64 MiB` 二维输出无法因分块而消失。块差值 `64²×128×4=2 MiB`，还需考虑归约小结果和实际运行时行为。

完整答案用 `inference_mode` 排除梯度图，及时释放每一轮 diff 的 Python 引用。它只限定本函数中间 Tensor 的逻辑形状，不承诺分配器保留量、RSS 或真实 GPU 峰值等于单个 tile。

### 有论文算法需要核验吗？

本篇主体是广播语义；平方距离由逐项差值定义展开，分块也仅按输出矩形拆解。没有引入需要单独归因的研究算法，因此不添加无关论文。后续涉及 Roofline、online softmax 与 FlashAttention 时会单独阅读原始论文并核对代码。

### 与实际 runtime 的关系

运行环境为 Python 3.10.13、PyTorch 2.9.1（`5811a8d7da873dd699ff6687092c225caffcf1bb`）、macOS arm64、CPU。这里的 2.9.1 实验核验 shape、零 stride、输出值、已知来源下的存储复用和梯度行为；没有证据说明它在所有 backend、编译模式或主分支上走完全相同的 kernel。

## 4. 复现入口

- [代码、题目与运行命令](code/README.md)。
- [测试与诊断原始输出](evidence/runs/)。
- [实际环境、命令与退出码](evidence/runs/summary.json)。
- [审查与修正](review.md)。
- [文件校验和](evidence/SHA256SUMS)。
