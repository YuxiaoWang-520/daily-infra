# 源码阅读笔记

核验日期：2026-11-05。官方默认分支 main：`4892050dadb28d1ca1065268e35084747967e34e`；稳定 tag v2.15.0：`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。本地获取方式和文件校验见 [versions.json](versions.json)、[source-manifest.json](source-manifest.json)。

## 问题一：初次 forward 在什么上下文中运行？

[main checkpoint.py](upstream/pytorch/main/torch/utils/checkpoint.py) 的 `checkpoint`（437 行起）转发到 `_checkpoint_impl`。非重入分支先推进 `_checkpoint_without_reentrant_generator_impl`，调用用户函数，再推进生成器完成后处理。没有把这条路径的 forward 描述成 `no_grad`；使用 `CheckpointFunction.apply` 的重入分支才是另一条实现。

## 问题二：边界与内部保存值分别放在哪里？

`_CheckpointFrame.save_inputs`（1016 行起）用 `_make_saved_tensor` 保存 Tensor 输入，`get_inputs` 负责取回。生成器在 `new_frame.save_inputs(*args)` 后进入 `_checkpoint_hook`。

`_checkpoint_hook.pack_hook`（1354 行起）产生 `_Holder`、保留其弱引用并记录元数据，不像普通保存那样把内部 Tensor 作为 pack 返回值。反向触发 unpack 时，根据 graph task／group 确认是否已经重算，取回输入并调用 `frame.recompute_fn`。

重算时 `_recomputation_hook` 把所需 Tensor 与 holder handle 关联；unpack 取出并使相应 handle 失效。这是按需恢复反向数据的路径，不能据此说图结构本身没有成本。

## 问题三：什么时候停止？

`_recomputation_hook`（1288 行起）递增已重建保存值的计数，达到需要的数量且启用 early stop 时抛内部 `_StopRecomputationError`，外层捕获它。于是进入 block 不保证执行到函数末尾。本篇入口序列只验证发生了重算，没有把它换算成精确算力开销。

## 问题四：RNG 与确定性检查保护什么？

生成器（1975 行起）记录 CPU 状态，并有受设备推断范围限制的非 CPU 状态处理。`recompute_fn`（2006 行起）在 `fork_rng` 中恢复状态并进入原计算相关上下文。本篇只验证 CPU eager，不把这段机制外推为任意自定义随机源都被保护。

`_default_meta_extractor`（1244 行起）只提取 shape、dtype、device；`check_recomputed_tensors_match`（1029 行起）检查保存数量及元数据，不比较张量元素值。因此正文用数值断言验证梯度与更新，而不只依赖默认 determinism check。

## 问题五：外层 hooks 能看见什么？

[main graph.py](upstream/pytorch/main/torch/autograd/graph.py) 的 `saved_tensors_hooks` 说明嵌套时只应用最内层 hooks，并要求 pack 返回值避免持有原输入 Tensor 引用形成环。答案使用 `detach()` 返回共享存储的张量，只在字典记录设备、地址与字节数。

本次统计上下文只围绕模型初次 forward，排除 loss；checkpoint 的内部保存、RNG 状态、图元数据和重算工作空间都不保证可见。记录的是该 fixture 的外层保存载荷，不能解释成全程实际峰值。

## 稳定版、实测包与 main 的边界

已归档 [v2.15.0 checkpoint.py](upstream/pytorch/v2.15.0/torch/utils/checkpoint.py)，并与本机同名文件逐字节比对一致。实测包报告的构建 SHA 独立保留，没有以 tag SHA 替换它。没有验证完整包的二进制来源，也没有在 main 下运行实验。

[实际 diff](upstream/main-v2.15.0.diff) 为 46 行增加、28 行删除。变化包括设备模块默认推断／错误处理，以及 SAC 在周围用户 hooks 存在且未显式选择 `respect_saved_tensors_hooks` 时，由稳定版警告变为当前 main 的报错。本文普通 checkpoint 实验未使用 SAC；不能据此宣称 SAC 统计正确或这些 main 变化已在本机执行。

所有上游文件按原字节归档，许可证位于 [LICENSE](upstream/pytorch/LICENSE)。文中固定链接的行号对应 main 归档；稳定版的行号不同。
