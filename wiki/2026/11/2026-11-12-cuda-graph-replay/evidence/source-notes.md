# 源码路径与版本核对

上游为 `https://github.com/pytorch/pytorch.git`，本机缓存在 `.cache/upstream/pytorch/`。实际执行 `git fetch --depth 1 origin main`，读取更新后的默认分支对象，使用完整 SHA 的 `git show` 获取证据。保留缓存已有 checkout，源码阅读使用固定对象。[版本记录](versions.json)、[更新记录](runs/00-upstream-fetch.txt)

## 调用路径

1. `torch/cuda/graphs.py` 的 `graph.__init__` 选择 side stream，`__enter__` 完成捕获前同步、缓存处理和 stream 上下文设置，再调用 `CUDAGraph.capture_begin`。
2. `aten/src/ATen/cuda/CUDAGraph.cpp` 的 `capture_begin` 核对 stream、捕获状态与内存池，调用 `cudaStreamBeginCapture`。
3. Python `capture_end` 调用 C++ `capture_end_pre`，随后按 `keep_graph` 调用实例化，再执行 `capture_end_post`。C++ 结束捕获、结束池路由、处理错误并检查节点。
4. C++ `instantiate` 调用 `cudaGraphInstantiateWithFlags`。Python `replay` 在需要时先实例化，随后调用 C++ `replay`。后者处理随机数状态并使用当前 stream 调用 `cudaGraphLaunch`。
5. `make_graphed_callables` 的 forward 片段比较输入指针，必要时向 `static_input_surface` 复制，再调用 fwd_graph.replay。它展示框架包装如何管理静态输入，正文的 CPU 练习只覆盖缓冲区概念。
6. `c10/cuda/CUDACachingAllocator.cpp` 的 `Interaction with CUDA graph capture` 注释解释原生 allocator 私有池和高水位保留。

每个节点均可在本地带原始行号的归档中查找。归档包含两个版本的同名符号及文档约束。[main Python](upstream/main/graphs.py.lines.txt)、[main C++](upstream/main/CUDAGraph.cpp.lines.txt)、[main allocator](upstream/main/CUDACachingAllocator.cpp.lines.txt)、[main 文档源码](upstream/main/cuda.md.lines.txt)

## tag 与 main 的核对

| 范围 | 核对结果 |
| --- | --- |
| 基本生命周期 | v2.15.0 与 main 都有 capture、instantiate 和 cudaGraphLaunch 主路径 |
| Python replay | main 增加跨 stream 重放前等待捕获前准备工作的说明 |
| RNG 缓冲区 | main 的 C++ capture_begin 预备 RNG 状态缓冲，Python capture_end_pre 保留缓冲并处理释放前同步 |
| capture 错误路径 | main 提前取得成功返回的 graph 所有权，并增加条件捕获范围的分配释放检查 |
| allocator 与文档 | 全文件存在其他改动，本篇仅以归档的固定地址／私有池注释及约束段支撑相关论断 |

main 的 RNG 预分配和条件捕获状态变化说明版本漂移会影响底层生命周期。本文示例采用固定前向算术、持有图及全部 I/O Tensor，并显式建立跨 stream 依赖。CUDA 路径需要目标设备上的正确性验证。

稳定 tag 的关键路径也已归档。`manifest.json` 保存完整 blob 的 SHA-256、原始行区间、归档 SHA-256 和两版本的差异行数，原始文件内容可从固定 Git 对象重建。[v2.15.0 Python](upstream/v2.15.0/graphs.py.lines.txt)、[v2.15.0 C++](upstream/v2.15.0/CUDAGraph.cpp.lines.txt)、[源码清单](upstream/manifest.json)

## 固定版本入口

当天默认分支对象为 `18f2eae1b30a6057d56ad2585a89fb3b965c3f43`。稳定 tag 为 `v2.15.0`，对应 `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。[当天 main](https://github.com/pytorch/pytorch/commit/18f2eae1b30a6057d56ad2585a89fb3b965c3f43)、[稳定 tag 对象](https://github.com/pytorch/pytorch/commit/31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457)
