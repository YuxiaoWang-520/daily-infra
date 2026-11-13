# 官方文档阅读笔记

实际检索与阅读日期为 2026-11-13，时区 Asia/Shanghai。通过 Parallel 查询并提取官方页面，记录标题、URL、检索／提取 ID 和返回日期元数据。[检索记录](research-log.json)

## torch.compile API

阅读参数 fullgraph、dynamic、backend，以及 code object 缓存、guards 和重编译限制说明。默认允许可捕获区域分段，严格模式要求相应区域满足单图捕获条件。dynamic=None 会在形状变化后尝试泛化，True 提前尽可能动态化，False 采用特化路径。具体算子和配置仍会影响结果。[PyTorch 2.15 API](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)

该版本文档还说明 fullgraph 会启用相关 unbacked 语义和标量捕获选项。正文使用明确的 graph_break 与数据相关 Python 分支作拒绝实验，避免把所有标量读取都概括为同一行为。

## Troubleshooting

阅读 Terminology 中的 graph break、Guards、Recompilation、Dynamic Shapes，Logging Tools 中的 TORCH_LOGS，以及 Data-dependent operations。核心关系为先检查已有结果的条件，全部无法匹配时才重新处理。graph break 可以结束前一段捕获、经过 Python 工作再恢复后续区域。[PyTorch 2.15 排障文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)

实际启用 graph_breaks、recompiles、guards、dynamic，保存完整 stderr。日志中出现 scale==2.0 的 EQUALS_MATCH 和 batch 维度 size mismatch，分别对应常量和静态形状实验。正文用累计图数与日志互相核对，记录器的输入还实际出现 SymInt。[原始日志](runs/03-torch-logs.txt)、[完整 FX 图与数值输出](runs/02-compile-lab.json)

## Dealing with Recompilations

阅读 Is Dynamic Shapes Enabled、Wrapping Constants with Tensors、Changing the Cache Size Limit 和 Graph Breaking to Reduce Recompilation Costs。Python 数值用于算术时可以尝试 Tensor 数据接口，仍要检查其 dtype、device 和其他用途。官方示例还说明，刻意分界有时可以缩小变化区域引发的重新处理范围，需要同时评估边界及优化成本。[重编译文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/programming_model.recompilation.html)

本篇固定 dynamic=False 比较 Python float 和零维 Tensor。这个配置使实测结论有明确范围，后续版本的 Python scalar 自动泛化需要按具体实现检查。

## Custom Backends

阅读 Overview 的 GraphModule／example_inputs 到执行 callable 的协议，以及返回 gm.forward 的官方示例。这个协议让观察器可以统计图交付并执行 FX 计算。优化代码生成、AOTAutograd 接入和不同后端的 kernel 行为具有各自的范围。[自定义后端文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_custom_backends.html)

初始旧路径 custom_backends.html 返回 404，搜索后使用上面的版本化有效地址，已成功获取内容。没有安装新的库或创建额外连接。

## Zero-One Specialization

阅读完整短文，记录尺寸 0、1 与广播、连续性及符号约束的关系。本篇 batch=0、1、2、4 的真实序列观察到 1、2、3、3，正文据此限定动态形状的复用范围。[0／1 特化文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.html)、[固定文档源码](upstream/main/zero-one-doc.lines.txt)

## 归档方式与范围

官方网页以自己的中文阅读笔记和章节定位归档，相关文档的固定 Git 源码也保留关键段落。没有保存整站 HTML。本文属于 API 与实现机制分析，官方文档、实际实现及 CPU 运行记录支撑核心论断，论文算法字段由范围说明替代。
