# GPU 并行映射资料

核验日期：2026-10-22。基础语义依据官方指南，不强行附会研究论文；未执行 GPU 代码。

| 主张 | 直接证据 | 边界 |
| --- | --- | --- |
| 线程、block、SM 与协作范围 | [CUDA 编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | 以 NVIDIA CUDA 为范围，不推广到所有厂商 warp／wave 大小。 |
| warp=32、块内分组与同步 | Writing SIMT Kernels 的线程索引、shared memory、warp 与驻留资源部分 | 一维练习省略多维线性化；没有用隐含锁步代替同步。 |
| 对齐与跨步访问的扇区解释 | [CUDA Best Practices 13.4，10.2.1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) | 原文示例限定 CC 6.0+ 和 4 字节访问。练习只是区间覆盖，不能证明实际 DRAM 事务。 |
| Triton 逻辑位置、mask、load/store | [固定源码 29—54 行](https://github.com/triton-lang/triton/blob/c1641abbb17a80b8c9cad1fa05b4c70d6ac260ab/python/tutorials/01-vector-add.py#L29-L54) | program 的元素数不能直接当 CUDA thread 数。 |
| grid 使用向上取整，调用返回可能异步 | [固定源码 62—78 行](https://github.com/triton-lang/triton/blob/c1641abbb17a80b8c9cad1fa05b4c70d6ac260ab/python/tutorials/01-vector-add.py#L62-L78) | 本篇只分析映射，异步执行在第 06 篇展开。 |

当前 CUDA 编程指南已采用按主题拆分的新路径；上面的 URL 是访问日实际打开的页面，不把旧文档路径重定向后的内容冒称历史版本。Best Practices 页标题显示 13.4，本文只引用明确的语义和访问条件，不移用旧设备速度。

实际源码来自 `triton-lang/triton` 默认分支 `main`，固定 `c1641abbb17a80b8c9cad1fa05b4c70d6ac260ab`。按文件获取，没有完整 clone 或构建。[命令、时间与哈希](evidence/source-manifest.json)，[完整教程](evidence/upstream/triton-lang/triton/python/tutorials/01-vector-add.py)，[许可证](evidence/upstream/triton-lang/triton/LICENSE)。

自己的对应笔记：CUDA 单线程写法按 `b*T+t` 选一个元素；Triton program 按 `pid*BLOCK_SIZE+arange` 选一段元素。两者都需要尾部保护，但编译抽象不同。访存层面再把选定元素坐标通过 stride 转成字节地址，才能分析同一 warp 某条加载的集中程度。没有用 CPU 枚举顺序推断 SM 调度顺序。

验证：[运行环境](evidence/runs/summary.json)、[9 项测试](evidence/runs/tests.txt)、[诊断](evidence/runs/lab.json)、[校验和](evidence/SHA256SUMS)、[归档检查](evidence/runs/archive-check.txt)。
