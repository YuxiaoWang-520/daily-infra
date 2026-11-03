# 参考资料与证据

主题：训练的梯度为什么需要通信，通信能否与计算重叠？检索日期为 2026-11-03（Asia/Shanghai）；固定源码采集时间 `11:05:48+08:00`。

## 核心论断与一手来源

| 编号 | 核心论断 | 一手来源与定位 | 本地证据 | 边界 |
| --- | --- | --- | --- | --- |
| M1 | 全局样本均值要求按有效分母加权；等量时化为 rank 平均 | 正文由可加目标自行推导；DDP 的 rank 除数与 SUM 路径在 S2 | [答案](code/answers.py)、[独立样本测试](code/test_answers.py)、[实跑 DDP](evidence/runs/ddp.json) | 没有自动推断业务 loss 分母；目标可能是样本、有效 token 或另定义权重 |
| P1 | 桶、hook、按序归约和累积是 DDP 的基础设计 | [原论文 v1](https://arxiv.org/html/2006.15704v1) §2.2—§2.3、§3.2.2—§3.2.4、Algorithm 1、Figure 3 | [论文笔记](evidence/paper-reading.md)、[获取与 hash](evidence/paper-acquisition.json) | PyTorch v1.5 的设计／实验，非当前性能数字 |
| S1 | 初始化桶与后续重建不同 | [初始桶容量](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/nn/parallel/distributed.py#L128-L162)、[_ddp_init_helper](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/nn/parallel/distributed.py#L1387-L1476)、[重建](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L2061-L2125) | [Python 原文](evidence/upstream/pytorch/pytorch/torch/nn/parallel/distributed.py)、[reducer 原文](evidence/upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp) | 不能从初始注册顺序、第一轮或 cap 推断稳态所有模式 |
| S2 | 普通固定组路径先做 world-size 归一化，再用 SUM 归约 | [dense ready](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L362-L439)、[除数](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L583-L600)、[SUM hook](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/default_comm_hooks.cpp#L46-L60) | [hook 原文](evidence/upstream/pytorch/pytorch/torch/csrc/distributed/c10d/default_comm_hooks.cpp)、前述 reducer | 排除自定义 hook、join、特殊混合精度等；数学平均与浮点实现顺序区分 |
| S3 | 参数 hook 推进桶 pending，按统一顺序异步提交，之后收尾 | [hook](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L694-L781)、[ready](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L923-L1040)、[按序](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L1090-L1115)、[finalize](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/csrc/distributed/c10d/reducer.cpp#L1799-L1864) | [源码阅读笔记](evidence/source-reading.md)、完整原文 | 本地 enqueue 不等于全局 ready；CUDA wait 不简单等同全设备同步 |
| S4 | no_sync 要同时包含 forward 和 backward | [no_sync](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/nn/parallel/distributed.py#L1695-L1722) | 原文、[完整 probe](code/ddp_probe.py) | 不消除程序内一切通信，不自动决定 loss 分母 |
| S5 | 当前存在编译分图与手动收尾等条件路径 | [DDPOptimizer](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/_dynamo/backends/distributed.py#L394-L442)、[manual finalization](https://github.com/pytorch/pytorch/blob/6ddf1c556aafe0295f682a2822d1510eda4beab2/torch/nn/parallel/distributed.py#L2656-L2740) | [编译原文](evidence/upstream/pytorch/pytorch/torch/_dynamo/backends/distributed.py)、前述 Python 原文 | 只读代码确认，均未执行；不泛化全部编译路径 |
| E1 | 单通道递推能解释就绪、固定成本和尾部 | 本文自定义时序模型，所有输入是假设值 | [lab](evidence/runs/lab.json)、[测试](evidence/runs/tests.txt) | 不是 GPU／网络计时，也不是 NCCL 算法模拟 |
| E2 | 默认 CPU DDP 的五组梯度与 SGD 参数结果 | 已安装 PyTorch 2.9.1，两个真实 Gloo 进程，float64 | [结果](evidence/runs/ddp.json)、[命令](evidence/runs/ddp-command.json)、[原始输出](evidence/runs/ddp.txt) | 验证数值和基本通信，不测通信次数、真实 overlap、收敛或 main 二进制 |

## 官方源码获取与许可

- 远端：`https://github.com/pytorch/pytorch.git`；默认分支 main 已经由 GitHub API 核验。
- 固定 commit：`6ddf1c556aafe0295f682a2822d1510eda4beab2`；提交时间 `2026-11-03T02:50:26Z`。实际 partial/shallow clone、sparse checkout、干净检查与逐文件读取，不安装框架。
- 稳定元数据：[v2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)，发布时间 `2026-10-28T17:09:49Z`，commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`；未单独分析该 tag。
- 归档 9 个原始文件，共 646,552 bytes：LICENSE、DDP 前端、reducer cpp／hpp、默认 C++／Python hooks、distributed_c10d、设计说明、DDPOptimizer。包含辅助定位文件，不代表逐行验证了整个通信后端。
- 逐文件原始字节、完整路径、固定链接、获取命令与 SHA-256：[source-manifest.json](evidence/source-manifest.json)。版本与旧文档路径修正：[versions.json](evidence/versions.json)。[上游 LICENSE](evidence/upstream/pytorch/pytorch/LICENSE) 和原始版权头保留，原文未做格式化。
- 当前官方 [DDP 设计页](https://docs.pytorch.org/docs/2.15/notes/ddp.html) 本次已访问，但其正文标明基于 v1.4；实现结论由上述固定代码支撑，而非浮动网页标题。

## 论文、图源与原始资料

原论文完整元数据、章节与当前实现差异见 [paper-reading.md](evidence/paper-reading.md)。本次实际读取固定 PDF 和官方 HTML，并检查 PDF 第 5 页；PDF 原文在 `.cache/upstream/papers/pytorch-ddp-2006.15704v1.pdf`，提取全文和渲染图在忽略的研究缓存。未将全文纳入仓库，仅归档自己的笔记、定位和获取校验信息。

正文图解为原创：[数据并行](assets/data-parallel.mmd)、[两桶重叠依赖](assets/overlap.mmd)；不是论文图复制或实际 trace。

## 运行、审查与交付

- 标准库答案：[answers.py](code/answers.py)、[起始接口](code/exercises.py)、[完整契约](code/README.md)。
- [命令与环境摘要](evidence/runs/summary.json)、[24 tests](evidence/runs/tests.txt)、[lab](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt)、[格式检查](evidence/runs/format.txt)。
- [Gloo 环境与首轮超时边界](evidence/probe-environment.md)、[独立 Python 审查及 LOW 修复](evidence/python-review.md)、[技术与教学两轮自审](review.md)。
- [核心校验和](evidence/SHA256SUMS)、[归档检查](evidence/runs/archive-check.txt)、[hash 复核](evidence/runs/checksums.txt)、[最终交付检查](evidence/runs/delivery-check.txt)。

GitHub 提交链接在完成回执中提供，避免文档自引用尚未生成的提交。验证边界以实际记录为准，不把版本标签、源码阅读和可运行二进制混称为同一验证。
