# 参考资料与证据

主题：参数之外，训练显存究竟花在哪里？检索日期为 2026-11-04（Asia/Shanghai），源码与字节核验时间 `2026-11-04T11:16:30+08:00`。参考资料优先使用官方代码、原论文和官方文档。

## 核心论断与来源

| 编号 | 论断 | 一手来源及定位 | 本地归档 | 证据范围 |
| --- | --- | --- | --- | --- |
| P1 | Adam 保留梯度与梯度平方的两个平滑历史，并做偏差修正 | [Adam v9](https://arxiv.org/abs/1412.6980v9)，Algorithm 1、§2—§3 | [论文笔记](evidence/paper-reading.md)、[获取与 hash](evidence/paper-acquisition.json) | 不规定状态 dtype；不引用历史实验作为今天性能证明 |
| S1 | 新建 Adam 在首次有梯度时延迟创建两个 zeros_like 状态和标量 step | [初始化路径](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/adam.py#L140-L212)、[step 调用](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/adam.py#L216-L271) | [adam.py 原文](evidence/upstream/pytorch/pytorch/torch/optim/adam.py)、[源码笔记](evidence/source-reading.md) | 默认新状态；不推断加载的 state 或其他 optimizer |
| S2 | 论文递推与实际 moment 更新一致，更新存在临时分母 | [moment 递推](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/adam.py#L455-L476)、[single-tensor 分母](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/adam.py#L529-L547)、[foreach 分母](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/adam.py#L779-L800) | 同上原文、[FP64 标量验证](evidence/runs/cpu-payload.json) | 不从某一路径推断所有编译／fused 峰值 |
| S3 | zero_grad(set_to_none=True) 不删除 Adam moment；foreach 可有额外中间空间 | [明确 set_to_none 参数](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/optimizer.py#L1108-L1172)、[foreach 提示](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/optim/optimizer.py#L272-L280) | [optimizer 原文](evidence/upstream/pytorch/pytorch/torch/optim/optimizer.py) | 设 None 与传位置参数 None 不同；其他引用可能持有梯度 |
| S4 | autocast 选择算子计算 dtype，不能直接推断参数存储变为 BF16 | [autocast 与 CPU 示例](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/amp/autocast_mode.py#L51-L137) | [autocast 原文](evidence/upstream/pytorch/pytorch/torch/amp/autocast_mode.py)、[AMP 示例原文](evidence/upstream/pytorch/pytorch/docs/source/notes/amp_examples.md) | 三组 CPU 实验具体到 2.9.1；非全部算子／设备测试 |
| S5 | 保存 backward 所需数据不一定分配新的完整副本 | [SavedVariable](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/csrc/autograd/saved_variable.cpp#L15-L87)、[saved tensor 定义](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/autograd/graph.py#L297-L317) | [C++ 原文](evidence/upstream/pytorch/pytorch/torch/csrc/autograd/saved_variable.cpp)、[graph 原文](evidence/upstream/pytorch/pytorch/torch/autograd/graph.py) | 不把 saved slots 的逻辑字节相加当成新增激活净额 |
| S6 | allocated、reserved 和驱动口径不同，empty_cache 不释放活跃张量 | [CUDA 内存管理](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/docs/source/notes/cuda.md#L543-L563)、[统计入口](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/cuda/memory.py#L530-L580)、[empty_cache](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/torch/cuda/memory.py#L222-L234) | [CUDA 说明原文](evidence/upstream/pytorch/pytorch/docs/source/notes/cuda.md)、[memory.py 原文](evidence/upstream/pytorch/pytorch/torch/cuda/memory.py) | 只读源码；无 GPU allocator 实测 |
| S7 | 显式低精度 moment 与默认高层 Adam 不是同一个配置 | [functional AdamW 状态创建示例](https://github.com/pytorch/pytorch/blob/665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f/docs/source/optim.md#L295-L329) | [optim 文档原文](evidence/upstream/pytorch/pytorch/docs/source/optim.md) | 只核对示例指定 dtype，不验证 fused CUDA kernel 或收敛 |
| E1 | 指定布局的逐项载荷、唯一 storage 的生命周期模型 | 本篇原创推导与合成输入 | [答案](code/answers.py)、[测试](code/test_answers.py)、[lab](evidence/runs/lab.json) | 无 allocator、GC 或异步释放模拟，不是 GPU 预测器 |
| E2 | CPU 状态 dtype、四个采样点、别名和标量 Adam 更新 | 本机已有 PyTorch 2.9.1 的真实执行 | [probe](code/payload_probe.py)、[原始 JSON](evidence/runs/cpu-payload.json)、[命令与环境](evidence/runs/summary.json) | 选定张量载荷，不是 RSS／全程序内存／GPU 峰值 |

## 官方源码获取与许可

- 官方远端：`https://github.com/pytorch/pytorch.git`；GitHub API 已核验默认分支为 main。
- 当日完整 commit：`665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f`，提交时间 `2026-11-04T02:58:20Z`。
- 复用 `.cache/upstream/pytorch/`。首次 HTTPS fetch 在 100 秒后超时；随后从同一官方仓库通过 SSH 执行 `git fetch --depth=1 --filter=blob:none git@github.com:pytorch/pytorch.git main` 成功，sparse checkout 后 detached HEAD 与 API SHA 一致，工作区干净。没有把超时尝试记为成功，也没有安装此源码。
- 稳定发布元数据：[v2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，发布时间 `2026-10-28T17:09:49Z`，tag commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`；未单独分析该 tag 实现。
- 实际运行版本：PyTorch **2.9.1**，git `5811a8d7da873dd699ff6687092c225caffcf1bb`，Python 3.10.13，macOS arm64，CUDA 不可用。
- 归档 10 个原始文件共 399,859 bytes，含 LICENSE、Adam、Optimizer、autocast、saved tensor、CUDA memory 接口与相关官方文档。Git blob SHA-1、SHA-256、完整文件路径、固定 URL 均在 [source-manifest.json](evidence/source-manifest.json)。
- [版本与获取记录](evidence/versions.json)、[源码阅读笔记](evidence/source-reading.md)、[上游 LICENSE](evidence/upstream/pytorch/pytorch/LICENSE)。原始字节和版权头保持完整，未做格式化；整文件存档不代表全部分支已验证。

## 论文、官方页面与图源

论文元数据、真实读取范围、算例和实现差异在 [paper-reading.md](evidence/paper-reading.md)。固定 PDF 的 hash、下载／提取命令与 xref warning 在 [paper-acquisition.json](evidence/paper-acquisition.json)。原文只放忽略的 `.cache/upstream/papers/`，Git 归档本文自己的笔记与证据，不再分发论文全文。

本次还实际检索官方 CUDA 内存语义页、AMP 和 optimizer 文档，用于定位当前主分支与稳定文档的区别；实现结论优先使用以上固定代码，不依靠浮动网页的版本标题。二手搜索结果未用作核心证据。

正文两幅图均原创，非 GPU trace、非论文截图：[状态生命周期](assets/state-lifecycle.mmd)、[别名与最后引用](assets/storage-alias.mmd)。Mermaid 块与图源逐字核对。

## 运行、审查与交付

- [起始接口](code/exercises.py)、[参考答案](code/answers.py)、[完整输入契约及执行方式](code/README.md)。
- [21 tests](evidence/runs/tests.txt)、[合成账本与时间线](evidence/runs/lab.json)、[CPU probe](evidence/runs/cpu-payload.json)、[probe stderr](evidence/runs/cpu-payload-stderr.txt)、[命令与环境摘要](evidence/runs/summary.json)。
- [Ruff check](evidence/runs/lint.txt)、[格式检查](evidence/runs/format.txt)、[答案创建前的 red 记录](evidence/runs/initial-red-tests.txt)。red 是导入失败，不是错误算法的复现。
- [独立 Python 审查](evidence/python-review.md)、[技术与教学两轮自审](review.md)。
- [核心校验和](evidence/SHA256SUMS)、[归档链接与源码定位检查](evidence/runs/archive-check.txt)、[hash 复核](evidence/runs/checksums.txt)、[交付检查](evidence/runs/delivery-check.txt)。

GitHub 提交链接在完成回执中提供，避免在文档内自引用尚未生成的提交。没有把 CPU 正确性、今日源码阅读、稳定 tag 元数据与 GPU 验证混称为同一结果。
