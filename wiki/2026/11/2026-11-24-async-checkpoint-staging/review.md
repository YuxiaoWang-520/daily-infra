# 审查记录

主题：异步 checkpoint 的暂存、重叠与完成边界。

文章状态：published。实际时间记录使用 Asia/Shanghai，见 [self-review.json](evidence/self-review.json) 和 [technical-audit.json](evidence/technical-audit.json)。技术证据与教学表达两轮均由主执行者自审，代码另有独立通用复核。

## 第一轮：技术事实与证据

- [x] 核心问题连接 checkpoint 对训练吞吐、内存与恢复一致性的影响。
- [x] 实际 fetch 官方 main，固定完整 SHA；稳定 tag 与安装包构建分别记录。
- [x] 追踪 stager 选择、复制、线程等待、两种返回、数据写入和 metadata 完成，说明分支条件。
- [x] 基础事件／容量模型没有冒称论文算法；论文不适用的原因已说明。
- [x] 时间、单位、背压与退出 drain 可复算，模型与真实 CPU 实验分开。
- [x] 两道完整答案已运行，18 项测试普通与优化模式通过，Ruff 检查通过。
- [x] 官方笔记、原样片段、许可、日志、状态与恢复值保留，没有 GPU 或 I/O 性能结论。

### 实际发现与处理

| 严重程度 | 观察与证据 | 处理 | 复验与边界 |
| --- | --- | --- | --- |
| MEDIUM（info） | 当前源码的 already-done staging Future 分支直接设置调用方结果；实际探针中 staging 报成功、upload RuntimeError、无 metadata | 正文单独说明固定版本行为，保存判据始终处理 upload 结果；不把单个暂存标志宣传成无条件成功 | [结果](evidence/runs/results.json) 与 main／stable 两个原样分支对照；未修改上游，已知边界仍明确保留 |
| LOW（note） | 首次 Ruff 报一个类型检查分支使用 RuntimeError | 改为 TypeError，其他数值／运行时异常语义保持各自含义 | [首次检查](evidence/runs/initial-ruff-check.txt) 与 [最终检查](evidence/runs/ruff-check.txt) |

其他实查：默认 `async_save` 构造参数与 `DefaultStager()` 默认值没有混淆；writer 自带阻塞 stager 的分支有代码证据；CPU 浅复制别名、stage 后快照隔离、失败 Future.done 与成功不同均真实验证。文件同步与 metadata 路径只被解释为对应 writer 的完成流程，没有宣称完整跨后端事务或掉电持久性。

## 第二轮：教学逻辑与表达

- [x] 先用第 7／8 步的小状态解释覆盖风险，再引入 staging 与 Future。
- [x] 先手算 42 s／64 s，再给公式与调用链；单位统一为 ms 或明确换算到 s。
- [x] 数据所有权图与练习状态一致，图源和正文代码块完全一致；不冒充实际时间 trace。
- [x] 策略表覆盖适用条件、内存／背压代价与验证口径。
- [x] 两题的输入、边界、完整参考实现、命令和验收写清。
- [x] 追问覆盖基础、实现、选择和排障；预归档链接与最终交付检查分别记录。

### 实际观察与修正

| 严重程度 | 观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | 初稿强调 copy 完成前不能覆盖源数据，但还需说明 GPU 生产数据与 copy 读取之间的依赖 | 补齐“设备生产 → copy → 后续覆盖”两段顺序，链接异步计时前置篇，并明确 CPU 实验不验证 CUDA stream 协议 | 通读后不再把 Python 更新调用返回视作设备更新完成 |
| LOW（note） | 正常完成语义先出现，版本异常分支较靠后，读者可能忽略异常处理前提 | 在第一次讲 result 的位置提前提醒异常与后面的版本边界 | 与后文探针、保存结果判据一致 |

另外逐项检查了：最后 6／16 s 等待计入完整任务；`S+max(kC,W)` 明确限于一次一个策略；`q×sum(B_r)` 按本节点计算；40 字节 CPU 功能用例没有冒充 64 GiB 吞吐实验；图未渲染和 GPU／多 rank 未执行如实说明。

## 独立代码复核

所需 `/root/nov24_python_review` 专项代理在执行前失败，原因是当前账户不支持其 `gpt-5.4` 模型。随后 `/root/nov24_python_fallback` 以独立通用 Python／代码安全审查角色读取本篇 5 个冻结源码文件，没有写仓库或安装依赖，也没有冒称专项身份。

独立复核报告通过：2,005 组逐步 oracle、50 组非法输入、两代三张量的 storage 隔离、阻塞／异步 × 成功／失败的四种真实 CPU 保存、提前 TimeoutError 后的门控／线程／目录清理，以及受控 CLI 输出的严格 JSON 检查。检查前后源码摘要均与执行清单匹配，未发现可报告的 CRITICAL／HIGH／MEDIUM／LOW 缺陷。

该部分记录审查者实际返回的检查报告，不伪造成原始 stdout；它没有独立检查正文／上游源码，也没有重复全部归档测试。[结构化报告](evidence/python-review.json)

## 实际运行与复验

使用现有 ai-infra 环境，Python 3.12.16、PyTorch 2.15.0（构建 SHA 单独记录）、Darwin arm64，无 CUDA、无已初始化分布式进程组。可执行命令在清单中以 basename 记录，实际使用本机指定环境；所有基础答案真实运行。

| 对象 | 命令 | 结果 | 原始记录 |
| --- | --- | --- | --- |
| 测试 | `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v` | 退出 0，18 项通过 | [普通模式](evidence/runs/unit-tests.txt) |
| 优化模式 | `PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v` | 退出 0，18 项通过 | [优化模式](evidence/runs/optimized-tests.txt) |
| 代码检查 | `ruff check code`；`ruff format --check code` | 均退出 0 | [检查](evidence/runs/ruff-check.txt)、[格式](evidence/runs/ruff-format.txt) |
| 两题完整运行 | `PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir evidence/runs` | 退出 0，时序、隔离、保存、失败与版本探针符合记录 | [stdout](evidence/runs/answers-run.txt)、[结果](evidence/runs/results.json) |

[verification.json](evidence/runs/verification.json) 固定命令、退出码、环境、源码摘要及真实日志。首次答案未实现的 18 项失败测试保留在 [red-tests.txt](evidence/runs/red-tests.txt)，不会拿初始失败状态当成最终结果。

单进程和 macOS 提示如实保存；checkpoint 在自有临时目录真实创建、加载、清理，仅公开小张量和阶段状态，未从外部读不可信 pickle。

## 发布结论

两轮自审与独立通用代码复核通过，本篇范围内无未解决 CRITICAL／HIGH 问题。已知上游 staging 报告边界没有被修改或隐藏，结论已按实际保存结果限定。没有 GPU、pinned／shared memory、保存进程、多 rank、网络文件系统、I/O 性能或掉电持久性验收；图源检查不冒充视觉渲染。

源码／数字／摘要核验见 [technical-audit.json](evidence/technical-audit.json)；文件、链接、隐私与索引见 [delivery-check.json](evidence/delivery-check.json)。Git 发布前基线见 [git-preflight.json](evidence/git-preflight.json)。GitHub 是否已同步以推送后的实际查询为准，不把尚未完成的推送写进这份文件。

## 勘误与更新

暂无。
