# 审查记录

主题：MoE 路由、Expert Parallel 的 AllToAll dispatch／combine 与梯度。

文章状态：published。审查日期：`2026-11-20`（Asia/Shanghai）。技术与教学两轮由主执行者自审，另有通用独立代码审查。固定模型专用 Python 审查器因不受支持而未能启动；不将替代审查冒称专用审查器执行成功。

## 第一轮：技术事实与证据

- [x] 核心问题连接稀疏 FFN 的计算、激活移动、负载和训练正确性。
- [x] 同日 fetch Megatron 与 PyTorch 官方默认分支和稳定 tag，固定完整 SHA，保留既有缓存。
- [x] 跟踪 Top-k 概率范围、capacity mask、aux 输入、splits、pack、dispatch、专家、combine 和 autograd 反向。
- [x] 核对 Shazeer 2017 与 Switch v3 原文，标明噪声、Top-1／Top-k、gate 范围和 local-batch 边界。
- [x] 复算四 token 输出、专家计数、容量丢弃、aux 数值、发送矩阵及远端字节预算。
- [x] 两道练习有完整接口和答案，23 项正常及 23 项优化模式测试通过，包含独立 dense 对照和有限差分。
- [x] 真实两个 Gloo rank 验证混合／空 owner 场景的输出、token／logits／专家梯度及一步 SGD 算术对照。
- [x] 没有用 CPU 正确性推出 GPU 性能、显存收益、分布式 aux／capacity 或完整框架训练结果。
- [x] 论文笔记、源码原始片段、许可证、图源和真实运行记录均已归档。

### 实际核验发现与处理

| 严重程度 | 实际观察 | 处理 | 复验 |
| --- | --- | --- | --- |
| MEDIUM | main dispatcher 默认 `alltoall`，最新稳定 `core_v0.20.1` 默认 `allgather`，不能笼统称为同一默认。 | 正文分别给完整 SHA、永久链接，要求显式配置分析路线。 | main／稳定 tag 原始配置片段分别归档，逐字节与 git show 对照。 |
| MEDIUM | 论文与工程存在不同 gate 归一化范围；默认 post-Top-1 门控恒为 1。 | 用同一 token 的 `4.4`／`5.5` 例子和 Top-1 梯度测试固定语义，不将不同论文规则混写。 | 单进程及独立针对性测试通过，记录未运行完整论文算法。 |
| MEDIUM | 当前上游发送 gate 到专家，教学两进程代码把 gate 留在源端。 | 明确相同线性加权语义下的位置差异；通信预算排除概率及 metadata，不冒称真实 Megatron 总通信量。 | 跟踪上游 token／probs 两次发送，核对代码的源端 gate 与 dense 梯度参考。 |
| LOW | 首次 Ruff 检查报告 `run_answers.py` 的未使用 torch import。 | 删除该 import，随后冻结六文件哈希。 | 最终 Ruff／格式退出 0，哈希与完整测试、独立审查一致。 |

本轮没有发现必要测试失败或剩余 CRITICAL／HIGH。初始未实现接口的 RED 是写测试在先的真实记录；完成实现后通过，没有虚构先失败后修复的功能缺陷。[技术复算与源码字节核验](evidence/technical-audit.json)覆盖 25 个固定版本片段、2 份完整许可证与六文件哈希。

## 第二轮：教学逻辑与表达

- [x] 从 `x=2` 与四个简单专家开篇，再解释 router、logits、Top-k、rank 和 assignment。
- [x] 手算、归一化、发送矩阵、图、容量、梯度和源码沿同一条数据路径递进。
- [x] 将 token 数、分配数、每专家行数、每 rank 行数和远端流量分开，明确统计范围。
- [x] Mermaid 仅表示依赖，t1 的专家输出 4／6 与最终 5.5 一致，源文件与正文相同。
- [x] 练习输入、输出、tie 规则、验收、答案及接入测试方法完整，不要求先阅读答案理解题意。
- [x] 说明容量会改变专家分支输出、aux 不能保证硬平衡、GPU／TP／融合等未验证范围。
- [x] 面试追问覆盖基础、数值、实现、训练、选择与排障。

### 实际教学调整

主例子的发送矩阵恰好对称，容易误导读者把 send splits 当作 receive splits。第二轮补充 `[[4,0],[4,0]]` 的非对称例子，明确 rank 0 的发送 `[4,0]` 与接收 `[4,4]`，并与真实空 owner 结果对应。

将源码的 combine 链接标注从容易混淆反向传播的措辞改成“返回 AllToAll”，使前向的返回通信与 autograd 的反向交换分开。核对正文：容量批次、SGD 算术对照、原始论文噪声省略及源端加权差异均有边界说明。

Mermaid 做了数据流人工检查和逐字节比对，没有调用渲染器，也不声称图像渲染验收。最终本地链接与交付状态由 [delivery-check.json](evidence/delivery-check.json)检查，没有剩余 CRITICAL／HIGH 教学问题。

## Python 代码审查

`python-reviewer` 固定模型任务在执行前失败，错误为 `gpt-5.4` 不受当前账户支持。随后由一次通用独立审查替代对应代码阶段：检查六文件、比较冻结哈希，并执行 7 项已有针对性测试，退出 0，无可操作发现。[实际角色、命令、哈希、结果与限制](evidence/python-review.json)。

审查覆盖门控范围、容量不重归一化、稀疏专家与空 owner 的反向依赖、splits 对调及全局分母。独立审查者未重跑完整／分布式套件；相同容量分数的 tie 行为由静态检查覆盖，所选测试不覆盖这一特定 tie。没有将通用审查称为成功执行的专用 Python 审查。

## 实际运行记录

| 对象 | 实际命令／环境 | 结果 | 记录 |
| --- | --- | --- | --- |
| 接口 RED | 先运行 unittest，再实现接口 | 23 项测试因未实现接口报错，退出 1 | [RED](evidence/runs/01-red-tests.txt)、[命令](evidence/runs/01-red.json) |
| 基础测试 | `python -m unittest discover -s code -p test_answers.py -v` | 23 项通过，退出 0 | [普通模式](evidence/runs/02-tests.txt) |
| 优化模式 | 同上并使用 `python -O` | 23 项通过，退出 0 | [优化模式](evidence/runs/03-optimized-tests.txt) |
| 静态与格式 | 现有 Ruff 的 `check code`／`format --check code` | 均退出 0 | [Ruff](evidence/runs/04-ruff.txt)、[格式](evidence/runs/05-format.txt) |
| 数值例子 | `python code/run_answers.py` | 手算、计数、aux 与假设负载一致，退出 0 | [数值输出](evidence/runs/06-answers.json) |
| 双进程 | Python 3.12.16、PyTorch 2.15.0、Darwin arm64，CPU／Gloo，临时 FileStore | 两场景四条结果均通过，最大绝对误差约 4.34×10⁻¹⁹ | [原始输出](evidence/runs/07-distributed.txt)、[解析结果](evidence/runs/07-distributed.json) |
| 独立复验 | 通用独立审查者的 7 项针对性测试 | 退出 0 | [审查报告](evidence/python-review.json) |

命令、退出码、现有环境和冻结代码 SHA256 见 [verification.json](evidence/runs/verification.json)。原始输出只替换本机路径及机器名；没有升级或安装依赖。

## 发布结论

- 两轮自审与独立代码审查已完成，未解决 CRITICAL／HIGH 为 0。
- 必要基础答案与实际 CPU／Gloo 通信、第一阶梯度验证通过；GPU、TP、fused kernel、分布式 aux／capacity、高阶 collective 梯度、真实显存与吞吐均未验证。
- 文件、图源、索引、链接、源码字节与脱敏交付检查见 [delivery-check.json](evidence/delivery-check.json)，完整内容哈希见 [SHA256SUMS](evidence/SHA256SUMS)。
- Git 前置核验见 [git-preflight.json](evidence/git-preflight.json)，其 SHA 为提交前基线，不是未来文章自身的 commit；最终是否同步以 GitHub main 核验为准。

## 勘误与更新

暂无。
