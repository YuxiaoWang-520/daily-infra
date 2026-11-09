# 审查记录

主题为 FSDP／ZeRO 的训练状态分片。日期为 2026-11-09，时区 Asia/Shanghai。文章状态为 published，内容审查通过并本地归档。交付与 Git 同步由最终检查核验。

技术事实与教学表达两轮由主执行者自审。Python 代码另由独立代理审查，实际角色、模型报错和范围保存在 [Python 审查记录](evidence/python-review.md)。

## 第一轮　技术事实与证据

实际阅读 ZeRO 原始论文第 3.1、4.1、5.1—5.3、7.1—7.2 节，核对状态与通信推导，再沿固定 PyTorch main 的入口、参数分片、前后 hooks、collective 与归约因子追踪。稳定 tag 的 root 自动策略另行读取，并对两个安装文件完成字节比较。

复算 P=10^9、N=8 的四项容量为 16、5.5、3.75、2 GB。复算 P=8、N=4、2 B 宽度时 24 B DDP 对照与 36 B 两次 AG 路径，改变梯度宽度为 4 B 后均为 48 B。该检查确认了固定通信倍率所依赖的 dtype 条件。

| 等级 | 实际观察 | 处置与复验 |
| --- | --- | --- |
| LOW（note） | 早期容量段使用“常驻”描述，包括了会随 zero_grad 生命周期改变的梯度 | 改成模型状态容量账，并补充存活区间说明，峰值按实际生命周期核算 |
| 检查通过 | flat 均匀 padding 与 FSDP2 按 Tensor 维度分片不同 | 正文给出 (3,8) 的 dim0 例子，并明确练习布局 |
| 检查通过 | 生命周期图是两次 AG，CPU 实验采用保留完整参数的一次 AG | 图标题条件、题解和真实通信 trace 分别说明 |
| 检查通过 | 不等样本数时 rank 均值会改变目标 | 两进程实际负对照中梯度最大差约 0.24417，正确路径差约 1.11e-16 |

24 项完整测试和组合实验通过。独立 Python 审查没有发现问题，并实际验证失败注入后的进程与目录收尾。性能估算、逻辑所有权计数和 CPU 正确性分开记录。实现引用固定完整 SHA，论文版本为 v3。

## 第二轮　教学逻辑与表达

从 4 个 rank 分担 8 个参数的例子进入，先解释状态所有权和 Adam，再推导容量、生命周期、源码与通信代价。rank、AG、RS、DTensor、mesh、延迟初始化和预取在相关段落解释，前置链接对应已发布文章。

| 等级 | 实际观察 | 调整与核对 |
| --- | --- | --- |
| LOW（note） | 初稿直接使用 lazy init 与预取名称 | 就地补充第一次执行时的延迟初始化，以及提前发起参数收集的含义 |
| 检查通过 | 只看 Adam 首步权重可能漏掉梯度尺度问题 | 详解展示非线性，手算 0.175 与 1/3，验收同时比较梯度、m、v 和三步参数 |
| 检查通过 | 新增参数组可能同时影响容量和延迟 | 工程表明确组大小、预取、重算及拓扑安排的代价与观测 |

正文中的主问题贯穿图、公式、两题和面试追问。图保留 Mermaid 源文件，题目模板与完整答案分离。复现命令使用现有 ai-infra Python，组合输出要求新路径。引用、笔记、来源清单和实际输出均在本篇目录中。

## 真实验证记录

| 项目 | 运行方式 | 结果 | 输出 |
| --- | --- | --- | --- |
| 两道练习完整测试 | unittest discover，test_*.py | 24 项通过，退出码 0 | [tests.txt](evidence/runs/tests.txt) |
| Python 检查 | 现有解释器的 ruff check code | 退出码 0 | [lint.txt](evidence/runs/lint.txt) |
| 格式检查 | 现有解释器的 ruff format --check code | 退出码 0 | [format.txt](evidence/runs/format.txt) |
| 真实两进程实验 | run_lab.py，CPU／Gloo，FP64，三步 | 正确与错误对照符合验收，退出码 0 | [lab.json](evidence/runs/lab.json) |
| 来源与链接 | Git blob 重建摘录、校验和、文档链接与索引 | 结果以最终交付记录为准 | [delivery-check.json](evidence/delivery-check.json) |

完整命令、退出码和环境见 [summary.json](evidence/runs/summary.json)。练习答案实现前的红测日志保留在 runs 目录。

## 发布结论

内容审查通过，没有遗留 CRITICAL 或 HIGH，也没有独立代码审查遗留问题。交付检查通过后归档并更新索引，再按指定路径提交推送，GitHub main 确认后报告已同步。

边界包括论文式混合精度布局、统一 flat padding、理想 ring 发送估算和 CPU 显式 collective。直接 FSDP2 GPU 执行、峰值显存、NCCL 性能与复杂混合并行仍需对应运行条件。

## 勘误与更新

暂无。
