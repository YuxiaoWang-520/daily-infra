# 官方资料阅读笔记

检索与核对日期为 2026-11-12，时区 Asia/Shanghai。通过 Parallel 实际检索并读取官方文档，查询与 URL 清单归档在本地。[检索记录](research-log.json)

## PyTorch 2.15 CUDA semantics

阅读 CUDA Graphs 下的 Why CUDA Graphs、PyTorch API、Constraints、Partial-network capture、Graph memory management、Sharing memory across captures 和 Data Dependent Control Flow。静态地址、新输入 copy_、长寿命输入输出和 side stream warmup 共同解释普通捕获的使用方式。Python 的 CPU 逻辑在捕获时执行，replay 复用记录的 CUDA 节点。尺寸和内部布局需要符合捕获假设，CPU／GPU 同步会触发捕获限制。[PyTorch 2.15 CUDA semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html)

当前文档已介绍 `torch.cond` 及 `cudagraphs` backend 的设备数据相关控制流路径。正文据此区分普通 Python 分支与设备条件节点。图池共享还要求顺序和并发条件，输出生命周期同样需要检查。正文专门说明分桶、padding、数据复制和池保留的成本。[条件控制流](https://docs.pytorch.org/docs/2.15/notes/cuda.html#graph-data-dependent-control-flow)、[池共享](https://docs.pytorch.org/docs/2.15/notes/cuda.html#sharing-memory-across-captures)

`stable/notes/cuda.html` 检索返回版本入口，进一步实际访问明确的 `2.15/notes/cuda.html`。通用 `stable/generated/...` 的提取内容出现旧 source tag，版本化 `2.15/generated/torch.cuda.CUDAGraph.html` 已指向 v2.15.0。正文采用版本化文档，并与固定 tag 源码相互核对。

## PyTorch 2.15 CUDAGraph API

阅读 `keep_graph`、capture_begin／end、instantiate、replay、raw_cuda_graph 和 raw_cuda_graph_exec。默认路径在 capture_end 实例化，保留模板时可显式实例化，首次 replay 也能触发实例化。访问及修改原始图需要调用方维护模板、执行实例和相关对象生命周期。修改模板后必须根据 API 重新实例化以应用修改。[PyTorch 2.15 CUDAGraph](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.CUDAGraph.html)

## NVIDIA CUDA Programming Guide §4.2

阅读 Graph Structure、Building and Running Graphs、Stream Capture、Cross-stream Dependencies and Events、Prohibited and Unhandled Operations、Invalidation、Graph Instantiation／Execution、Updating Instantiated Graphs 和 Conditional Graph Nodes。一次执行图包含多个节点与依赖，capture、instantiate、launch 各有独立阶段。跨 stream 捕获需要形成完整依赖，非法操作可能使 capture 失效。节点更新按节点类型和更新种类受限制，条件节点支持设备控制的条件执行与循环。[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

该地址为滚动在线指南，记录抓取返回的发布日期 2026-10-31 和实际访问日期。正文没有给该网页套用未经核验的 Toolkit 版本号。CUDA 运行时及驱动在本机缺少实机证据。

## 本地归档与可复核范围

官方网页采用中文阅读笔记与精确章节定位保存。没有保存整站 HTML。PyTorch 相关约束的文档源码与关键实现已按固定 commit、原始行号归档，保留上游许可证。检索记录保存 URL、标题、检索／提取 ID 和返回日期元数据，正文结论由实际阅读及固定源码支撑。[源码清单](upstream/manifest.json)、[许可证](upstream/LICENSE)

## 论文对应关系

本文研究 CUDA Graph 的 API 生命周期、固定参数语义和框架实现，以官方语义文档及实现作为主证据。队列递推为本篇明确假设下的教学模型，已用逐操作模拟检验。本文没有引入需要另行解释的论文算法。
