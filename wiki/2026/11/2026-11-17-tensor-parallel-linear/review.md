# 审查记录

主题为两层 MLP 的列／行切分、梯度和通信布局。审查日期为 `2026-11-17`，时区为 `Asia/Shanghai`（UTC+08:00）。技术证据与教学表达两轮由 MAIN 自审。Python 代码由独立通用代理另做限定范围审查；专用代理未能启动的原因和实际替代范围见下文。

## 第一轮：技术事实与证据

实际 fetch 官方 PyTorch main，并新建 Megatron-LM 官方源码缓存。按完整 commit 读取并归档 `16` 段、`612` 行证据，核对列／行分片参数、bias 放置、输出布局转换、自定义 backward 及 MLP 调用接口。稳定 release、默认分支快照和本机实际执行的 PyTorch build 分别记录。[源码清单](evidence/source-manifest.json)

阅读 Megatron-LM 原始论文 §3 及附录 B，并阅读 Sequence Parallel 原始论文 §4.2.1—4.2.2。对照普通 TP 的前向部分和、反向输入梯度，以及 SP 保存分片输入时的权重梯度路径。[论文笔记](evidence/paper-notes.md)

自行复算前向 `Y=[[11,-5],[10,6]]`、MSE `51.25`、输入梯度 `[[19,1],[1.5,15.5]]`，检查第二层 bias 只生效一次。参考答案比较全部梯度，并对 `26` 个变量做中心有限差分。容量与 ring 预算按 `N=2048、H=8192、F=32768、p=8、s=2 B` 核对单位及发送／接收口径。

| 严重程度 | 实际发现与证据 | 修改与复验 |
| --- | --- | --- |
| MEDIUM（已修正） | SP 的四个基本布局转换不足以覆盖“仅保存分片输入，反向计算 wgrad”路径。论文同节及当前 Megatron backward 均包含额外输入 AllGather | 扩展预算的 `regather_input_for_wgrad` 选项，先加入两项失败回归，再实现并复验。正文、图和答案区分 `2 AG + 2 RS` 与 `3 AG + 2 RS`，每 rank 发送量分别为 `112 MiB` 与 `140 MiB`。[回归红灯](evidence/runs/03-regather-red.txt)、[通过结果](evidence/runs/04-tests.txt) |
| LOW（已修正） | 首次两进程探测的输出存在进程间粘连，逐行解码不适用于采集完整 JSON；原始张量与通信记录仍已输出 | 最终采集按 `TP_RESULT` 标识与 JSON 解码边界提取，重新实际运行，验收四条完整记录和进程退出码。[首次输出](evidence/runs/02-distributed-probe.txt)、[最终记录](evidence/runs/09-distributed.json) |
| LOW（已处理） | 暂存差异检查发现两份原始进程日志的 PyTorch 启动提示带有行尾空格 | 对这两个具体日志路径设置空白检查例外，保留实际输出并纳入校验和，重新执行暂存差异检查 |

检查了两类容易越界的结论。当前 Megatron 优化线性层可以在自定义 backward 内归约 dgrad，调用条件需结合配置追踪。AllGather 的反向则要区分副本梯度和不同计算贡献，正文按指定映射分别解释。CPU／Gloo 观察只证明该环境的数值、梯度和框架调用；SP 计数依据论文及固定源码，ring 流量为公式估算。

## 第二轮：教学逻辑与表达

按“同一批 token → 列切分完整特征 → 行切分部分和 → bias 与非线性反例 → 梯度 → 布局 → 源码 → 实验 → 预算与取舍”通读。前置篇链接到 GEMM、DDP 和通信拓扑。首次解释 rank、TP、token 行、Replicate／Shard／Partial，并在使用 `nn.Linear` 时解释存储权重的转置关系。

核对前向图的两条贡献路径、SP 图的四个基本转换与虚线额外收集；图源与正文通过交付检查逐字比对。Mermaid 未在本次环境渲染，完成源语法与数据流人工检查，不将其称为渲染验收。

两道练习都给出输入输出、前提、起始接口、完整答案、运行命令、预期结果和边界情况。第一题区分中间宽度的可整除性与隐藏输入宽度，第二题声明均分 ring chunk、相同边界 dtype 及预算排除项。面试追问覆盖基础、推导、实现、选择与排障。此轮没有发现需要新增修正的教学问题。

## Python 独立代码审查

请求的固定 `python-reviewer` 因其配置模型在当前账号不可用而未启动。随后由独立通用代理只读审查本篇六个 Python 文件，复核 shape、bias、autograd、分片、预算及进程收尾，并实际运行六项针对性测试，全部通过，退出码 `0`。实际审查未发现 CRITICAL、HIGH、MEDIUM 或 LOW 问题。[独立审查范围、命令与文件哈希](evidence/python-review.json)

该代理对两 rank 脚本执行静态复核，真实进程运行由 MAIN 完成。独立审查没有运行 SP、CUDA／NCCL 或通信性能实验，专用 Python 审查未完成的事实如上记录。

## 运行与验证记录

本机使用既有 ai-infra Python `3.12.16`、PyTorch `2.15.0`，`Darwin arm64`，Gloo 可用、CUDA 不可用。没有安装或升级依赖。实际命令和 SHA256 见 [运行清单](evidence/runs/verification.json)，命令在草稿目录执行，归档后 `code/` 与证据的相对关系保持一致。

| 验证对象 | 结果 | 实际证据 |
| --- | --- | --- |
| 测试先行 | 初始 `18` 项测试在答案未实现时失败，另有 SP 重新收集的两项回归红灯 | [初始记录](evidence/runs/01-red-verification.json)、[新增回归](evidence/runs/03-regather-red.txt) |
| 完整基础答案 | `20` 项 unittest 通过，全部参数／输入梯度和有限差分对齐 | [普通模式](evidence/runs/04-tests.txt) |
| 优化模式 | 相同 `20` 项测试通过 | [-O 模式](evidence/runs/05-optimized-tests.txt) |
| 代码规范 | 现有 Ruff 的 check 与 format 检查通过 | [lint](evidence/runs/06-ruff.txt)、[format](evidence/runs/07-format.txt) |
| 数值与预算 | 手算输出、梯度与三个预算均实际输出；预算属于教学模型 | [JSON](evidence/runs/08-answers.json) |
| 两 rank TP | 两种方案、两个 rank 的输出和全部梯度与完整参考一致，参数梯度最大绝对误差为 `0.0` | [结构化记录](evidence/runs/09-distributed.json)、[原始输出](evidence/runs/09-distributed.txt) |

## 发布结论

两轮自审和独立代码复核完成，已发现问题均已处理，未解决的 CRITICAL／HIGH 为 `0`。核心机制有原始论文和固定源码支撑，两道基础练习以及普通 TP 的两 rank 路径有真实验证，达到本篇发布标准。

SP 的实际设备执行、GPU／NCCL 性能和其他融合／expert 配置未测试；相关论断限定在论文、指定源码或公式模型范围。最终链接、图源、许可、源码片段、文件哈希及脱敏检查以 [交付结果](evidence/delivery-check.json) 为准。文章完成本地归档后按仓库流程同步，推送后的 GitHub main 核验结果用于最终报告。[提交前核验](evidence/git-preflight.json)

## 勘误与更新

暂无发布后勘误。研究与审查期间的实际修正如上记录。
