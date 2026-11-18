# 审查记录

主题为 Pipeline Parallel 的 microbatch 调度、气泡与待反向激活保存。审查日期为 `2026-11-18`，时区为 `Asia/Shanghai`（UTC+08:00）。技术证据和教学表达两轮由 MAIN 自审，六个 Python 文件另经独立通用代理复核。

## 第一轮：技术事实与证据

更新两个官方缓存的 `origin/main`，固定完整 commit，读取并归档 `20` 段、`566` 行源码。检查 GPipe 的两次循环、1F1B 的预热／交替／排空、局部状态队列、梯度缩放、元数据派生和 P2P 操作。稳定 release、默认分支快照与实际安装 build 分别记录，未构建或运行 main 源码版本。[源码清单](evidence/source-manifest.json)

阅读 GPipe `1811.06965v5` 的 §2.1—2.3、图 2，以及 Megatron `2104.04473v5` 的 §2.2.1—2.2.2、图 3—4、§3.1、3.2、3.4。核对同步更新、保存窗口、气泡比例分母和 microbatch 的计算效率取舍。GPipe 摘要页与 HTML 版内题名存在差异，笔记分别记录其题名，以 ID 和版本定位。[论文笔记](evidence/paper-notes.md)

手工复算两张单位时间表、`[4,4,4]` 与 `[3,2,1]` 的保存峰值，以及 `48+18=66 ms` 的预算。逐项比对真实两 rank 的局部操作序列、全部参数梯度和一次 SGD 更新。核对固定 batch 三组成本、激活容量和通信 payload，确认平均计算时隙利用率没有被说成硬件 FLOP 利用率。

| 严重程度 | 实际发现 | 修正与复验 |
| --- | --- | --- |
| MEDIUM（已修正） | 首次两 rank 实验的静态输入输出示例没有正确声明 `requires_grad`，派生出的梯度元数据为 `None`，真实反向失败 | 核对 `_derive_grad_metas`，声明跨段输入及输出的梯度标志；保留失败日志，修正后探测和最终完整运行均通过。[失败](evidence/runs/02-distributed-metadata-red.txt)、[最终结果](evidence/runs/09-distributed.json) |
| LOW（已修正） | 现有 Ruff 对相邻事件遍历要求使用 `pairwise` | 调整测试的遍历方式，最终 lint／format 与普通及优化模式测试均通过。[lint](evidence/runs/06-ruff.txt)、[测试](evidence/runs/04-tests.txt) |

另外核对了两处边界：数学模型支持 `m<p`，当前 PyTorch `Schedule1F1B` 构造器不接受该配置；PyTorch 和 Megatron 的预热计数差 `1`，来自交替阶段第一份前向的划分方式。在本文真实配置下，完整局部次序一致。

单进程梯度题覆盖不等长份，真实进程只验证等长份，未将二者范围混用。microbatch 反向采用统一的全局 MSE 分母，真实框架脚本显式关闭再次按 `m` 缩放。参数保持到全部本地反向完成后才更新。

## 第二轮：教学逻辑与表达

按“切开模型仍有依赖 → 拆分输入形成重叠 → 小时序表 → 保存状态的窗口 → 两个气泡分母 → loss 与参数版本 → 当前源码 → 真实进程 → 工程成本”通读。检查 stage、microbatch、rank、DP／TP／PP 和每个公式符号的首次解释，前置篇链接到已发布内容。

| 严重程度 | 实际观察 | 修改与复验 |
| --- | --- | --- |
| LOW（已修正） | 第一题的“参数阶段内不更新”容易被理解为模拟器还建模了权重状态 | 改为“一次 step 内参数固定”的假设，明确模拟器不执行参数更新；第二题及真实脚本实际检查最后的一次更新 |

两道练习都有起始接口、输入输出、前提、完整答案、验收条件与边界。`θ=0` 的反例让读者看到错误分母如何同时改变 loss 和梯度。固定 batch 的三行预算用自编耗时揭示“气泡占比更小”与“step 更快”需分别验证。

人工检查 Mermaid 的箭头与状态依赖，正文两张时间表与模拟器归档结果逐行一致。图源等同性由交付检查验证，本次没有使用 Mermaid 渲染器，不宣称完成渲染验收。CPU 调用计数和 GPU 显存、公式估算和设备测量都有明确范围。

## Python 独立代码复核

固定 `python-reviewer` 请求因配置的 `gpt-5.4` 不受当前账号支持而未能启动，未完成专用审查。随后由独立通用代理只读复核本篇六个 Python 文件，补跑 `7` 项有区分度的测试，退出码为 `0`，未发现 CRITICAL、HIGH、MEDIUM 或 LOW 问题。六文件 SHA256 与 MAIN 的最终运行清单一致。[实际范围、命令与报告](evidence/python-review.json)

该代理核对归档的四条真实进程结果，未重新执行多进程。完整测试、代码检查及真实两 rank 运行由 MAIN 执行，角色和范围分别记录。

## 实际运行与验证

使用现有 ai-infra Python `3.12.16`、PyTorch `2.15.0`，`Darwin arm64`，Gloo 可用、CUDA 不可用。没有新增或升级依赖。命令在草稿阶段执行，归档后 `code/` 与证据的相对关系不变。

| 验证对象 | 实际结果 | 证据 |
| --- | --- | --- |
| 测试先行 | 初始 `21` 项测试运行，未实现接口及子测试出现预期错误，退出码 `1` | [命令](evidence/runs/01-red-verification.json)、[原始输出](evidence/runs/01-red-tests.txt) |
| 基础答案 | 普通及 `-O` 模式各 `21` 项通过 | [普通](evidence/runs/04-tests.txt)、[优化模式](evidence/runs/05-optimized-tests.txt) |
| 代码规范 | 已有 Ruff lint 和 format 检查均通过 | [lint](evidence/runs/06-ruff.txt)、[format](evidence/runs/07-format.txt) |
| 数值与模型预算 | 不等长份的 loss、全部参数梯度对齐；导出两种时序和三组假设预算 | [JSON](evidence/runs/08-answers.json) |
| 实际 GPipe／1F1B | 两种调度 × 两个 rank，全部梯度及一步更新通过 `1e-12` 容差，最终待反向计数均为 `0` | [结构化结果](evidence/runs/09-distributed.json)、[原始输出](evidence/runs/09-distributed.txt) |

全部实际命令、退出码、环境和代码 SHA256 见 [最终运行清单](evidence/runs/verification.json)。峰值计数为未配对前后向调用数；保存字节、消息 payload 和 step 时间表采用标明的模型假设。

## 发布结论

技术和教学两轮自审完成，已发现问题已修正并按范围复验，未解决的 CRITICAL／HIGH 为 `0`。原始论文、当日固定源码、两道完整 CPU 练习和真实框架进程结果足以支撑本篇限定的核心问题。

没有验证 GPU／NCCL、真实显存峰值、通信性能、动态 shape、分布式不等长份、Megatron GPU 执行或交错多 chunk 调度。文章通过审查后归档，最终本地链接、图源、源码与许可哈希、代码冻结哈希及脱敏结果见 [交付清单](evidence/delivery-check.json)。Git 同步按仓库流程执行，以推送后的 GitHub main 确认为准。[提交前核验](evidence/git-preflight.json)

## 勘误与更新

暂无发布后勘误。研究及审查期间的实际修正如上记录。
