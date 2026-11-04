# 审查记录

主题：参数之外，训练显存究竟花在哪里？文章状态：published。审查日期：2026-11-04（Asia/Shanghai）；技术及教学检查于 `11:25:37+08:00` 后执行，最终归档复验时间见运行摘要。

技术事实与教学表达均由主代理自审；Python 代码另由独立代理进行审查及增量复核。固定 python-reviewer 的调用失败，真实执行者、范围与限制见 [Python 审查记录](evidence/python-review.md)，不冒充人工或独立技术全文审稿。

## 第一轮：技术事实与证据

- [x] 主问题对应训练容量与 OOM 的常见场景，不用冷僻知识凑题。
- [x] 已实际 fetch 官方 main，固定完整 commit；10 份源码原文与 Git blob、SHA-256 对齐，main／稳定 tag 元数据／本地 runtime 分别记录。
- [x] 追踪 Adam.step → _init_group → 更新公式；核对 None 梯度、延迟状态、step 标量、zeros_like dtype 和 foreach 中间结果。
- [x] 实际读取 Adam 原始论文的 Algorithm 1、§2—§3 并检查 PDF 图像；算法与代码的初始化方式、计步及 dtype 边界均说明。
- [x] 复算 16 元素账本、1e9 元素换算、单块激活 16 MiB、正常峰值 384 和保留 view 峰值 480；图表、题目、答案与 lab 一致。
- [x] 两道练习有完整答案与契约，21 个测试、合成 lab 与真实 CPU probe 通过。
- [x] 明确区分选定张量载荷、唯一 storage、allocator 与驱动占用；没有 GPU 峰值、速度或收敛的伪实测。
- [x] 源码原文、论文笔记、版本、许可、图源及真实运行记录在本篇目录内。

### 实际核对与修正

| 严重程度 | 观察及证据 | 处理 | 复验 |
| --- | --- | --- | --- |
| LOW | 基础练习的 16P 不含 step 标量，而真实 CPU after_step 为 260 bytes，不能直接拿 256 充当实测 | 正文与代码说明明确每个参数张量多一个 4 bytes step；模型和 probe 的口径分开 | 三组四时点计数与 JSON 一致；未“修正”真实数据来配公式 |
| LOW | 开头 BF16 权重 1.86 GiB 与后文 FP32 状态 14.90 GiB 属于不同配置，直接比较会误导 | 显式说明切换了配置，并逐项展示 dtype 与 master copy | 核对表格的 16P／8P／16P，以及进阶 12P 的假设 |
| 检查点 | 源码 HTTPS 首次超时；PDF 有 xref warning | 保留失败／warning，后续 SSH fetch 成功；PDF 提取并对照渲染页读取 | 源码 SHA 与 API 一致、工作区干净；PDF 获取与提取记录归档 |
| 检查点 | 原论文的手算需要与真实 optimizer 更新互相核对 | 增加 FP64 标量 moment 和参数更新断言，并增量独立复核 | m≈0.2、v≈0.004、w≈0.999000000005；CPU probe 退出 0 |

表中前两项是自审中核对的口径易错点，最终正文已写明边界，不声称曾发生过不存在的 GPU 实验错误。Python 独立首轮与增量复核均未发现需修复问题。

## 第二轮：教学逻辑与表达

- [x] 从“权重只有 2 GB 为什么仍 OOM”进入；随后只谈一个数和梯度，再逐步引入 Adam 两份历史。
- [x] 补上 `loss=w²`、梯度 2 与数值更新的小例子，不让梯度定义只停留在偏导符号。
- [x] 在首次出现生命周期图前解释 optimizer step；16 元素表格之后再扩大到 10 亿参数。
- [x] 先说明 autocast 的职责，再列布局表和当前功能接口示例；不让进阶配置抢走主线。
- [x] 先用 XW 的导数说明保存数据的必要性，再讨论激活容量和共享 storage。
- [x] 两幅 Mermaid 有图源，时间线手工计算与运行记录一致。
- [x] 优化方向分别给出收益机制、条件与代价；没有未经实测的节省比例或加速倍数。
- [x] 两道题不要求先读答案才能理解，含独立验收数字、完整契约、边界和常见错误。
- [x] 面试追问覆盖基础、实现、方案选择及验证；前置文章和下一篇问题自然衔接。

### 实际发现与修正

| 严重程度 | 观察及证据 | 修改 | 复核 |
| --- | --- | --- | --- |
| MEDIUM | 初稿表头与一道追问把设置梯度 None 简写成 `zero_grad(None)`，读者可能照着传错位置参数 | 表头改为“梯度设为 None 后”，提问使用完整 `zero_grad(set_to_none=True)`；源码笔记强调两者不同 | 原始调用与 probe 均使用显式 True，教学调用无歧义 |
| LOW | 初稿用文字和偏导解释梯度，缺少一个能直接算出 loss 变化的小例子 | 补充 w=1、loss=1、g=2，学习率 0.05 更新后 w=0.9、loss=0.81 | 算术复算正确，说明真实训练不保证每一步 loss 单调下降 |

图解按 Mermaid 源码结构、数字和正文一致性审查，没有执行像素渲染，不把结构核对写成视觉截图验收。最终路径的相对链接、图源块一致性、索引、文件清单与校验和由交付检查再核验。

## 实际运行与验证

完整命令、工作目录、时间、软件环境与退出码在 [summary.json](evidence/runs/summary.json)。

| 对象 | 环境／方法 | 结果 | 原始记录 |
| --- | --- | --- | --- |
| 两道标准库练习 | Python 3.10.13，unittest | 21 tests 通过 | [tests.txt](evidence/runs/tests.txt) |
| 布局和生命周期 lab | 合成输入、整数算术 | 256／128／256／192 等布局；正常峰值 384、保留 view 峰值 480 | [lab.json](evidence/runs/lab.json) |
| 状态与 autocast | 已有 PyTorch 2.9.1，CPU，16 元素 Linear | FP32 两组 64／128／260／196；BF16 32／64／132／100 bytes | [cpu-payload.json](evidence/runs/cpu-payload.json) |
| 实际别名与标量 Adam | CPU storage 去重；FP64 标量 | 128 逻辑 bytes、96 唯一 storage bytes；moment 与偏差修正更新断言通过 | 同上 |
| Python 质量 | Ruff E／F／I／UP，format check | 通过 | [lint](evidence/runs/lint.txt)、[format](evidence/runs/format.txt) |
| 原始资料与交付 | 固定源码、hash、Markdown 链接与图源块 | 归档后保存真实检查输出 | [archive](evidence/runs/archive-check.txt)、[checksums](evidence/runs/checksums.txt)、[delivery](evidence/runs/delivery-check.txt) |

## 发布结论

- 未解决 CRITICAL／HIGH／核心 MEDIUM：无。基础答案与必要 CPU 行为已验证，技术与教学两轮自审、独立 Python 审查完成。
- 未运行：GPU allocator API、GPU 峰值、foreach／fused GPU 性能、functional BF16 moment、分布式状态分摊、收敛及当日 main 二进制。
- 发布条件：内容审查已通过，正文与本篇资料已移至最终目录；索引与交付检查的实际结果见下方归档记录。是否推送成功以任务回执中实际 GitHub SHA 确认为准，不以文档里的计划代替。
- 研究只使用公开资料和合成输入，没有归档私有代码、训练日志或用户业务数据；原始论文不再分发。

## 勘误与更新

暂无。
