# 固定源码阅读笔记

当日 main 为 `cbee6335ae15a556794e2552c553f5befda84516`。摘录保留原始行号，完整 blob 和摘录哈希位于 [source-manifest.json](source-manifest.json)。

## 参数所有权入口

`_fully_shard.py::fully_shard` 的参数默认值包括 reshard_after_forward=None。第 260—325 行核验 mesh、获取设备、标记自动策略、创建参数组并把 FSDP 行为应用到模块。

`_fsdp_param.py::_init_sharded_param` 选择分片 placement，按参数 shape 切片，处理 padding，再创建分片 DTensor 并替换模块参数。第 355—408 行能看到补齐 storage 与逻辑 narrow 视图的区别。练习统一 flat 槽位的预算要与实际每 Tensor 的布局区分。

`to_sharded` 第 1029 行起将模块指回分片参数，并根据 keep_unsharded_storage 控制完整存储生命周期。`to_unsharded` 第 1122 行起指向物化参数，并清理相应后向重分片状态。

## 自动策略与执行层级

`_fsdp_state.py` 第 118—145 行注册前后 hooks，第 179—232 行延迟初始化识别真正的 root。自动策略下，root 的 post_forward_mesh_info 设为 None，以保留前向后的完整参数。稳定 tag 的入口与 root 片段也已单独归档，语义与本次结论一致。

`_fsdp_param_group.py` 的 pre_forward 在第 627 行附近调用 unshard 和 wait_for_unshard。post_forward 调用 reshard，pre_backward 再请求参数，post_backward 取得 autograd 梯度并按策略重分片后执行归约。

## 集合通信和平均语义

`_fsdp_collectives.py::DefaultAllGather` 调用 all_gather_single，DefaultReduceScatter 调用 reduce_scatter_single。foreach_all_gather 组合输入、分配输出和启动通信，参数组等待后 copy-out 并切换到完整参数。

foreach_reduce 解析归约因子，并在通信前后按需缩放。第 1000—1064 行的 _get_gradient_divide_factors 根据 dtype、分组、设备与 factor 选择 AVG 或 SUM 加缩放。本文 CPU 练习显式使用 SUM，并在局部 sum loss 中使用全局样本分母。

## 混合精度与优化器

`_fsdp_api.py::MixedPrecisionPolicy` 分别描述参数 dtype、reduce dtype 和输出 dtype。容量与流量均按实际参与的 dtype 核算。

Adam 摘录保留 m 的 lerp、v 的平方更新、偏差修正和 addcdiv 参数更新。CPU 程序使用实际 torch.optim.Adam，默认禁用 foreach，FP64、无 weight decay、无 AMSGrad。它把局部分片交给本地优化器，再与全集参考逐项比较。

## 证据边界

源码阅读支撑当前 FSDP2 的入口、状态机与 collective 选择。真实运行记录支撑 CPU／Gloo 教学程序的归约和局部 Adam。GPU 上的 hooks、DTensor、异步 stream、预取效果、显存峰值与吞吐需要相应硬件运行证据。
