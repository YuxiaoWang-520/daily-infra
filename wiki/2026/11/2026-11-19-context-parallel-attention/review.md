# 审查记录

主题：Context Parallel 的 K／V 交换、全局 Softmax、因果布局与梯度归约。

文章状态：published。审查日期：`2026-11-19`（Asia/Shanghai）。

技术审查：主执行者自审。教学审查：同一主执行者第二轮自审。Python 代码另由通用独立审查者检查；所要求的固定模型专用 Python 审查器因模型不受支持而未能执行，不能称其已通过。[实际审查角色与结果](evidence/python-review.json)。

## 第一轮：技术事实与证据

- [x] 主问题针对长序列训练，明确本地 Q 与全局 K／V 的依赖。
- [x] 同日更新三个官方源码仓库，固定 main 与稳定 tag 的完整 SHA；保留缓存原状态。
- [x] 逐段核对 CP 组、TE backend、通信分派、P2P 缓冲、Softmax 合并、因果 tile 与反向累加。
- [x] 核对 Ring Attention v4 的第 3 节、Algorithm 1 与附录 A；区分 JAX 论文实现、TE 路线与教学模型。
- [x] 复算四 token 例子、zigzag 有效配对数、LSE 合并、梯度公式及前向字节预算。
- [x] 两道练习有完整接口、答案和验收；正常与 `-O` 模式的 `26` 项测试均通过，包含有限差分。
- [x] 两个真实 CPU／Gloo rank 对齐 AllGather 基线输出与全部 Q／K／V 梯度，P2P 环形前向也对齐。
- [x] CPU 正确性、假设容量与通信预算、GPU 源码分析分别表述；没有 GPU 性能结论。
- [x] 固定源码片段、许可证、自己的资料笔记、真实输出和六文件哈希已归档。

### 实际发现与修正

| 严重程度 | 实际观察与证据 | 修正 | 复验 |
| --- | --- | --- | --- |
| MEDIUM | 所有 K／V 块为空时，原实现返回的零输出没有 autograd 路径，`backward()` 报错。新增测试真实失败，见 `03-edge-red-tests.txt`。 | 空块通过零贡献保留 Q／K／V 反向依赖。 | 空块输出与梯度为零，正常／优化模式测试及独立针对性测试均通过。 |
| MEDIUM | 输入张量有限并不能保证 `QKᵀ` 有限，新增 `1e300` 输入测试暴露未报错。 | 分数溢出后明确抛出 `ValueError`，不把 NaN 当作有效输出。 | 新测试先失败后通过；普通与 `-O` 模式均覆盖。 |
| MEDIUM | 首次 `torchrun --standalone` 在 `100` 秒超时，且初始包装器没有捕获保存超时输出。 | 改用临时 FileStore 启动两个实际进程；如实保留启动失败、原始输出缺失和恢复说明，不归因网络或硬件。 | 新命令退出 `0`，四条结果齐全；输出及 Q／K／V 梯度最大绝对误差约 `1.11×10⁻¹⁶`。 |
| LOW | 可微 `nn.functional.all_gather` 发出弃用警告；安装包 build SHA 与抓取的 main、稳定 tag 不同。 | 保留警告，说明它是固定环境中的教学基线；归档安装包源码片段，分开记录三个版本边界。 | 检查警告原始输出与记录，未宣称 main 被实际执行。 |
| LOW | 最初定位到了论文 v3，而版本历史显示 v4。 | 正文与主要笔记固定 v4，记录日期、原始作者与实际阅读范围。 | 对照 v4 第 3 节和附录 A，未将工具缓存日期混同检索日期。 |

技术自审另逐字节核验 `24` 个固定版本片段与 `3` 份许可证，核对全文 SHA256、原始行范围和正文永久链接。六个 Python 文件与运行、独立审查的哈希一致。没有剩余 CRITICAL／HIGH 技术发现。

## 第二轮：教学逻辑与表达

- [x] 从长文档容量问题、`8` token 所有权与 `4` token 手算开始，首次出现的 rank、mask、Q／K／V 均有解释。
- [x] 数值错误例子引出 Softmax 摘要，随后进入环、因果布局、反向与源码，围绕同一核心问题。
- [x] Mermaid 展示 Q 固定与 K／V 跨 rank 的依赖，源文件与正文一致；没有将示意图当作实际并发 trace。
- [x] zigzag 有效配对数与 GPU 时间分开；说明完整序列、可整除、无 dropout 等前提。
- [x] 两题的输入、输出、限制和验收独立于答案叙述；完整答案与真实命令可定位。
- [x] 问题链涵盖基础、数值、布局、梯度、实现、选择和排障。
- [x] 对论文、当前源码、安装包实测、容量假设及未验证 GPU 部分均有明确边界。

### 教学审查的实际调整与检查

前期教学设计若只展示环形前向，容易让读者误以为能据此证明训练正确。因此本稿单列反向公式、K／V owner 求和和全局 loss 分母，并在两进程实验中分别标注“可微 AllGather 训练语义”和“P2P 前向”。这些说明与脚本中两种 `mode` 逐项核对，未将后者写成环形训练验收。

正文与运行说明同时补充：起始文件只提供两道主接口，复用现有测试时仅替换它们的导入；布局辅助函数可保留参考版本或自行实现。这样读者无需先阅读完整答案才能知道怎样接入测试。

复查了全局位置例子、最后一行输出 `30`、错误局部平均 `550/21`、连续／zigzag 的 `[10,26]`／`[18,18]`、`128／512／384 MiB` 的不同口径，数值一致。没有剩余 CRITICAL／HIGH 教学发现。Mermaid 仅做源码与数据流人工检查、正文块逐字节比对，未调用渲染器或声称图像渲染验收。

## Python 独立代码审查

固定模型 `python-reviewer` 任务在开始前失败，错误为 `gpt-5.4` 不受当前账户支持。随后只安排一次通用独立审查替代对应代码阶段：读取六个 Python 文件，检查 online softmax、mask、空块、布局与输出恢复、loss 分母及 AllGather 梯度语义，并实际运行 `7` 项针对性测试，退出 `0`，未发现可操作问题。

独立审查者没有重跑完整测试、Ruff 或分布式程序，这些由主执行者实际运行并归档。没有把通用审查者标为专用 Python 审查器；六文件冻结哈希和真实审查范围见 [python-review.json](evidence/python-review.json)。

## 运行与验证记录

| 对象 | 实际命令／环境 | 结果 | 记录 |
| --- | --- | --- | --- |
| 初始接口 | 先运行 `unittest discover`，再实现答案 | `24` 项测试的未实现接口失败 | [初始 RED](evidence/runs/01-red-tests.txt) |
| 新增边界 | 空 key 反向、有限输入导致分数溢出 | 原实现 `1` 失败、`1` 错误；随后修复 | [边界 RED](evidence/runs/03-edge-red-tests.txt) |
| 完整基础测试 | `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v` | `26` 项通过，退出 `0` | [正常模式](evidence/runs/04-tests.txt) |
| 优化模式 | 相同测试加 `python -O` | `26` 项通过，退出 `0` | [优化模式](evidence/runs/05-optimized-tests.txt) |
| 静态与格式 | 已有 Ruff `0.17.3`，`check code`、`format --check code` | 均退出 `0` | [Ruff](evidence/runs/06-ruff.txt)、[格式](evidence/runs/07-format.txt) |
| 数值例子 | `python code/run_answers.py` | 输出与预算可复算，退出 `0` | [答案 JSON](evidence/runs/08-answers.json) |
| 实际双进程 | 已有 Python `3.12.16`、PyTorch `2.15.0`、CPU／Gloo、float64；`python code/distributed_cpu.py` | 两个 rank 的两种 mode 均通过，退出 `0` | [原始输出](evidence/runs/09-distributed.txt)、[解析结果](evidence/runs/09-distributed.json) |
| 代码独立复验 | 通用独立审查者运行 `7` 项针对性测试 | 退出 `0` | [审查者报告](evidence/python-review.json) |

最终代码哈希、实际命令、环境和退出码见 [verification.json](evidence/runs/verification.json)。运行记录替换本机路径与机器名，不伪造早期失败的缺失输出，不自动安装或升级依赖。

## 发布结论

- 核心技术证据、两道答案和必要 CPU 验证已经通过；未解决的 CRITICAL／HIGH 问题为 `0`。
- 未执行 GPU／NCCL／TE kernel、分布式环形反向、完整模型参数梯度归约、通信重叠、真实训练峰值内存与吞吐；正文不超出这一边界。
- 两轮主执行者自审完成；独立代码审查的实际角色、失败的专用审查器与替代范围均如实记录。
- 最终链接、图源、状态、索引、源码字节、敏感信息与交付清单的检查以 [delivery-check.json](evidence/delivery-check.json)为准。Git 前置状态见 [git-preflight.json](evidence/git-preflight.json)，它记录提交前的基线 SHA；最终同步以远端提交核验为准。

## 勘误与更新

暂无。
