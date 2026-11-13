# 源码路径与版本核对

官方 PyTorch main 已实际 fetch 并固定到 a0d96468b7f2089907f400411eda3b09feda8f76，稳定 tag v2.15.0 对应 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457。两版本片段使用 git show 的固定对象读取，保留上游许可证。[更新输出](runs/00-upstream-fetch.txt)、[版本记录](versions.json)

## 参数到捕获

torch/__init__.py 的 compile 将 fullgraph 映射为 nopython，将 dynamic 及 backend 等交给 Dynamo optimize。torch/_dynamo/eval_frame.py 的 _optimize 分别选择严格捕获或允许分段的转换路径。正文实验显式传入这三个设置，形成可复核的输入条件。[main 入口](upstream/main/torch-init.lines.txt)、[main eval_frame](upstream/main/eval-frame.lines.txt)

## 图交付

torch/_dynamo/output_graph.py 的 call_user_compiler 包装 _call_user_compiler，后者整理图元数据后调用 compiler_fn(gm, example_inputs)，并检查结果可调用。本篇 CountingBackend 在该协议入口记录图并返回 gm.forward。compile_subgraph 与 compile_and_call_fx_graph 的入口也已实际阅读，用于核对图边界到后端交付的衔接。[main 交付片段](upstream/main/output-graph.lines.txt)

记录器保存的 FX 代码由实际运行产生，包含生成的 forward 和计算节点。本篇算术没有调用优化后端，图交付统计的解释限定在当前实验配置。[真实生成图](runs/02-compile-lab.json)

## guards 与缓存

torch/_dynamo/convert_frame.py 在构建 guards 后，使用 GuardedCode 保存执行代码及 guard manager。torch/csrc/dynamo/extra_state.cpp 的 lookup_in_list 核对 backend 并运行 guard manager，遍历条目寻找匹配结果。这为“旧值再次匹配旧结果”提供直接实现证据。[GuardedCode](upstream/main/convert-frame.lines.txt)、[C++ 缓存匹配](upstream/main/extra-state.lines.txt)

## Python float 的版本边界

当天 main 提交只删除 builder.py 中 wrap_symfloat 的 unspecialize_float_killswitch 条件。相邻代码还检查 specialize_float、常量来源、NaN／Inf、CUDA Graph 配置以及静态假设／自动动态状态，满足特化条件时安装 CONSTANT_MATCH。稳定 tag 的对应分支也已读取归档。[main 浮点分支](upstream/main/float-builder.lines.txt)、[稳定版浮点分支](upstream/v2.15.0/float-builder.lines.txt)、[当天 main 提交](https://github.com/pytorch/pytorch/commit/a0d96468b7f2089907f400411eda3b09feda8f76)

因此正文把 Python scalar 的累计图数限定到实测构建与 dynamic=False。自动泛化的整体行为需要结合输入历史、配置和完整版本检查。

## 文档与可复核清单

两个文档文件分别保存动态形状／Tensor 常量接口及刻意分界的策略，以及 0／1 特化说明。它们在本次 stable tag 与 main 之间内容一致，6 个实现文件整体均有差异。清单保存各 blob 的完整 SHA-256、归档行区间和片段哈希。共 16 份片段、1018 行，可从两个固定对象重建。[清单](upstream/manifest.json)、[文档片段](upstream/main/recompilation-doc.lines.txt)、[许可证](upstream/LICENSE)
