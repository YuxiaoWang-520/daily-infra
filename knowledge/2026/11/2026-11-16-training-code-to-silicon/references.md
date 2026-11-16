# 参考资料与阅读笔记

## 1. 取样和证据范围

本专题以公开实现、官方规范及自编合成程序重建训练代码到芯片的路径。资料核验日期为 `2026-11-16`，时区为 `Asia/Shanghai`。架构划分和自研交付顺序是本文的教学组织与工程建议，具体函数、接口和运行结果分别落到以下证据。

这次追踪的是基础矩阵运算、自动求导、设备代码生成与提交接口，采用源码和规范说明机制。文中没有引入特定论文算法。

## 2. 实际拉取的官方源码

| 官方仓库 | 当日 `main` 的完整 commit | 核验时最新 release | release 发布时间 |
| --- | --- | --- | --- |
| `pytorch/pytorch` | `7d191ca7274de43d574c2dbe35365fefd008babf` | `v2.15.0` | `2026-10-28T17:09:49Z` |
| `triton-lang/triton` | `46eccb199bfd3f3559b512c4a15870bd45e6a74d` | `v3.9.0` | `2026-10-27T18:56:21Z` |
| `NVIDIA/open-gpu-kernel-modules` | `e6a2e01579fbbd1ca2ed43fef2ee2589e9bd8bd9` | `620.70.06` | `2026-11-05T00:19:23Z` |

PyTorch 使用已有缓存并 fetch 官方 `main`，保留缓存工作树原有状态。另两个仓库使用 `--filter=blob:none --no-checkout --depth=1` 拉取当日默认分支，随后用固定 commit 的 `git show` 阅读选定文件。三个缓存都位于仓库的 `.cache/upstream/` 下。固定片段、来源行号、完整源文件 SHA256 和片段 SHA256 已归档。[源码清单](evidence/source-manifest.json)、[release API 核验](evidence/release-snapshot.json)

正文分析对应上述默认分支快照。release 表用于说明当时稳定发布参照，各段实现分析按链接中的 commit 定位。实际执行的 CPU PyTorch build commit 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`，与源码阅读快照分别记录。三套上游版本的兼容安装组合需要相应组件的兼容文档和集成测试。

### 2.1 PyTorch 的算子和求导连接

| 阅读问题 | 精确位置 | 本地片段 | 得到的结论 |
| --- | --- | --- | --- |
| 二维 `@` 如何处理？ | `aten/src/ATen/native/LinearAlgebra.cpp:2022–2056` | [matmul-2d](evidence/upstream/pytorch/matmul-2d.txt) | `_matmul_impl` 在两输入均二维时调用 `mm` |
| `mm` 怎样分派设备？ | `aten/src/ATen/native/native_functions.yaml:4139–4156` | [mm-dispatch](evidence/upstream/pytorch/mm-dispatch.txt) | 结构化委托及 CPU／CUDA 实现名称可见 |
| 矩阵梯度如何连接？ | `tools/autograd/derivatives.yaml:1239–1243` | [autograd-mm](evidence/upstream/pytorch/autograd-mm.txt) | 导数规则连接到两个矩阵反向 helper |
| ReLU 和均值的反向规则？ | 同文件 `2150–2153`、`1159–1164` | [ReLU](evidence/upstream/pytorch/autograd-relu.txt)、[mean](evidence/upstream/pytorch/autograd-mean.txt) | 阈值梯度及按元素数缩放的语义与推导一致 |
| CUDA mm 的入口？ | `aten/src/ATen/native/cuda/Blas.cpp:913–927` | [cuda-mm-entry](evidence/upstream/pytorch/cuda-mm-entry.txt) | `mm_out_cuda` 使用 `beta=0、alpha=1` 进入 addmm 实现 |
| 怎样选择 Lt／GEMM 路径？ | 同文件 `556–612`、`676–749` | [selection](evidence/upstream/pytorch/cuda-addmm-selection.txt)、[launch](evidence/upstream/pytorch/cuda-addmm-launch.txt) | 存在输入与设备约束、Lt 回退以及 activation 处理 |
| FP32 GEMM 怎样进库？ | `aten/src/ATen/cuda/CUDABlas.cpp:1017–1050`、`1301–1320` | [float](evidence/upstream/pytorch/cublas-float.txt)、[preference](evidence/upstream/pytorch/cublas-preference.txt) | preferred backend 继续参与选择，cuBLAS 分支使用 `cublasGemmEx` |
| 行列主序如何适配？ | `aten/src/ATen/native/cuda/cuBlasCommonArgs.h:100–171` | [layout](evidence/upstream/pytorch/cublas-layout.txt) | 根据结果布局交换输入、设置转置、尺寸和 leading dimension |
| CUDA 标量如何回主机？ | `aten/src/ATen/native/cuda/CUDAScalar.cu:59–83` | [scalar-read](evidence/upstream/pytorch/cuda-scalar-read.txt) | NVIDIA CUDA 路径借助 pinned 主机存储及 `memcpy_and_sync` 取值 |

这里追到了公开框架代码中的库调用边界。cuBLAS 的内部算法选择、最终 GPU 指令及执行次数，要在选定设备和安装组合上进一步观察。某个框架操作与设备 kernel 的对应关系受分解、融合和具体算法影响。

### 2.2 Triton 的编译与加载连接

| 阅读问题 | 精确位置 | 本地片段 | 得到的结论 |
| --- | --- | --- | --- |
| NVIDIA 后端有哪些阶段？ | `third_party/nvidia/backend/compiler.py:661–671` | [stages](evidence/upstream/triton/nvidia-compiler-stages.txt) | Triton 语言路径经过 TTIR、TTGIR、LLIR、PTX、cubin |
| PTX 和 cubin 怎样生成？ | 同文件 `547–575`、`577–618` | [PTX](evidence/upstream/triton/nvidia-ptx.txt)、[ptxas](evidence/upstream/triton/nvidia-ptxas.txt) | LLVM 汇编输出及 `ptxas` 子进程命令可见 |
| 加载器怎样对接 CUDA？ | `third_party/nvidia/backend/driver.c:290–326` | [load-binary](evidence/upstream/triton/nvidia-load-binary.txt) | context、模块、函数及资源查询通过 Driver API 完成 |
| 怎样启动？ | 同文件 `1050–1127` | [launch](evidence/upstream/triton/nvidia-launch.txt) | 构造扩展启动配置并调用 `cuLaunchKernelEx` |
| Python 编译对象怎样接加载器？ | `python/triton/compiler/compiler.py:482–493` | [Python load](evidence/upstream/triton/python-load-binary.txt) | 编译对象使用设备镜像和元数据加载函数 |

本例的运行脚本执行 eager CPU 和 FX 前向。上述 Triton 路径来自源码阅读。PyTorch 编译路径和特定发行包搭配的 Triton 版本应按该发行包核对。

### 2.3 Linux 驱动与设备接入连接

| 阅读问题 | 精确位置 | 本地片段 | 得到的结论 |
| --- | --- | --- | --- |
| NVIDIA 开源 KMD 怎样配套？ | `README.md:1–23` | [matched-components](evidence/upstream/nvidia-open-gpu-kernel-modules/matched-components.txt) | 版本 `620.70.06` 要求匹配的 GSP 与用户态组件 |
| Linux 的应用入口在哪里？ | `kernel-open/nvidia/nv.c:252–273`、`3034–3041` | [file-operations](evidence/upstream/nvidia-open-gpu-kernel-modules/linux-file-operations.txt)、[ioctl](evidence/upstream/nvidia-open-gpu-kernel-modules/linux-ioctl-entry.txt) | 可见字符设备操作与 ioctl 连接 |
| PCI 设备怎样接 OS？ | `kernel-open/nvidia/nv-pci.c:1191–1205`、`2722–2734` | [probe](evidence/upstream/nvidia-open-gpu-kernel-modules/linux-pci-probe.txt)、[pci-driver](evidence/upstream/nvidia-open-gpu-kernel-modules/linux-pci-driver.txt) | 可见 PCI driver 登记与设备启用 |
| 新设备怎样接框架？ | `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/` 下相关文件 | [算子登记](evidence/upstream/pytorch/openreg-operator-registration.txt)、[allocator](evidence/upstream/pytorch/openreg-allocator.txt)、[guard](evidence/upstream/pytorch/openreg-guard.txt) | 操作、存储与设备 guard 是不同接入点 |

通用队列、doorbell、完成序号及自研固件的叙述是接口设计模型。本文保留公开实现边界，不据这些入口推断完整的 NVIDIA 内部提交协议。

## 3. 官方文档与自己的阅读笔记

以下均在资料核验日实际检索或打开。滚动页面按访问日期解释，表中的版本取自当时页面标识。归档采用自己的摘要和源码片段，官方文档正文通过来源页面阅读。

| 来源 | 版本／定位 | 阅读笔记和正文用途 |
| --- | --- | --- |
| [PyTorch Accelerator Integration](https://docs.pytorch.org/docs/2.15/accelerator/index.html) | `2.15`，Runtime／Operators／Frontend／High-level Modules | 设备支持覆盖多项子系统，本文再继续展开底层硬件职责 |
| [PyTorch Operator Registration](https://docs.pytorch.org/docs/2.15/accelerator/operators.html) | `2.15`，必要 Tensor 操作与 PrivateUse1 | 基础创建、视图、复制与后端登记共同形成接入面 |
| [PyTorch Device Management](https://docs.pytorch.org/docs/2.15/accelerator/device.html) | `2.15` | 设备资源接入及 runtime 抽象的参考 |
| [PyTorch Profiler Integration](https://docs.pytorch.org/docs/2.15/accelerator/profiler.html) | `2.15`，legacy stubs／Kineto | 框架的设备集成接口需要厂商 tracing 数据和真实活动记录 |
| [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) | `13.5`，CUDA Compilation Trajectory | host／device 分拆，设备镜像、fatbinary 与加载选择，本文选常见 PTX 路线说明 |
| [Runtime vs Driver API](https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html) | `13.5` | runtime 与 driver API 在 context 和模块管理上有不同封装，支持按规则互操作 |
| [Driver API Execution Control](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html) | `13.5`，`cuLaunchKernel`／`cuLaunchKernelEx` | function、grid、block、shared memory、stream、参数及 ABI 的连接 |
| [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) | `13.5`，Data Layout／cuBLASLt | 经典列主序约定，Lt 提供更多布局与执行描述能力 |
| [cuDNN Overview](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/overview.html) | 滚动文档，页面更新 `2026-11-06` | legacy API 与 Graph API／frontend 共同构成接口背景 |
| [NCCL Overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) | `2.32.3` | 集合通信与点对点原语，连接多设备训练及 stream |
| [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | 当日 CUDA Guide，Heterogeneous Systems／线程组织 | host／device、grid／block 和硬件映射的公开语义 |
| [LLVM 后端开发](https://llvm.org/docs/WritingAnLLVMBackend.html) | 滚动开发文档，页面更新 `2026-11-15` | 目标描述、寄存器、指令、lowering、选择与发码需要目标相关实现 |
| [CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/index.html) | `13.5` | 主机与设备调试，断点、单步及设备状态检查 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | 当日滚动文档，Introduction／Tools | 区分 memcheck、racecheck、initcheck 和 synccheck 的覆盖范围 |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | 当日 User Guide，应用标记与时间线 | 主机调用与设备活动的关联有助于观察空隙和重叠 |
| [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 当日 Profiling Guide，Metric Collection | kernel 指标、计数器、replay 与采集开销需要配套解释 |
| [CUPTI Usage](https://docs.nvidia.com/cupti/main/main.html) | 当日文档，Correlation ID Mechanism | API 与 GPU activity 通过 correlation ID 连接，记录还带 context／stream 等范围 |
| [OpenTitan DV](https://opentitan.org/book/doc/contributing/dv/methodology/index.html) | `master` 文档，方法与 testplan | 仿真、形式方法、测试和覆盖计划可以共同组织芯片验证 |
| [OpenROAD Flow Scripts](https://openroad-flow-scripts.readthedocs.io/en/latest/mainREADME.html) | 当日 `latest` 文档，RTL-to-GDSII | 综合、floorplan、placement、CTS、routing 和 finishing 的职责与产物 |

## 4. 自编模型、实测与推断的边界

- `code/reference.py` 使用标量 Python 独立实现固定样例的前向、解析梯度和 SGD。有限差分测试独立检查 `22` 个导数。
- `code/walkthrough.py` 使用实际 CPU PyTorch 执行 FP32／FP64 前向、反向和更新，保留 autograd 节点类型及 CPU 操作统计，另输出前向 FX 图。
- `144 FLOP` 和大矩阵示例的 `128 MiB`、`1024 FLOP/B` 来自正文给定假设与公式，可由整数运算复算。
- GPU 库选择、Triton 代码生成、CUDA API 和 KMD 入口来自源码／规范。实际 GPU kernel、SASS、性能、调试工具和芯片流程的执行状态未在此实验中测量。
- 9 层职责、通用提交队列及研发阶段表由本文组织，用于说明接口、依赖与验收，具体产品需要按架构重划边界。

全部运行命令、退出码和代码 SHA256 归档在验证记录中。日志写入前替换了本机绝对仓库／用户目录，CPU profiler 的本机进程与线程编号也作占位处理，数值、操作名和形状保持真实。[验证记录](evidence/runs/verification.json)

## 5. 许可与图源

公开源码片段保持原文，行号与摘要在清单中登记。PyTorch 的许可、Triton 的许可、NVIDIA 包级许可与所选源文件的版权头随片段归档。片段目录及原始 red 测试日志保留原有空白，使用本篇局部 `.gitattributes` 对这些路径关闭 Git 空白告警。[PyTorch LICENSE](evidence/upstream/pytorch/LICENSE)、[Triton LICENSE](evidence/upstream/triton/LICENSE)、[NVIDIA COPYING](evidence/upstream/nvidia-open-gpu-kernel-modules/COPYING)、[NVIDIA 文件版权头](evidence/upstream/nvidia-open-gpu-kernel-modules/file-notices.txt)

三幅 Mermaid 图由本文自编，分别表示职责依赖、通用执行顺序及软硬件研发关系，均非设备实测 trace。图源与正文代码块一致。[职责图](assets/stack.mmd)、[执行图](assets/execution.mmd)、[研发图](assets/development.mmd)

返回 [正文](README.md)。
