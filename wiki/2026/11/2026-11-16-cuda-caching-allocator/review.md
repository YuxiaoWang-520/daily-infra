# 审查记录

主题为 CUDA caching allocator 的复用、碎片与 OOM。审查日期为 2026-11-16，时区 Asia/Shanghai（UTC+08:00）。文章已审查并本地归档，Git 同步结果以推送后的 GitHub main 核验为准。

技术证据与教学表达两轮均由 MAIN 自审。Python 代码另经通用代理独立审查，固定 Python 专用代理的模型错误和实际替代范围如实记录。[独立代码审查](evidence/python-review.md)

## 第一轮，技术事实与证据

实际更新官方 PyTorch main，核验最新稳定 release 为 v2.15.0，按固定完整 commit 读取 8 组源码证据，共 1,247 行。核对申请入口、舍入、分池、候选选择、分裂、邻块合并、事件处理、整段回收和 Python 绑定。对照两版指定大小策略函数，并保存范围与匹配哈希。[版本](evidence/versions.json) · [对照](evidence/source-comparison.json)

核对图中的两个“6 MiB 空闲 + 2 MiB 使用中”布局、12 MiB 总空闲与 6 MiB 最大空闲块，以及释放邻块后 9 MiB 已分配的恢复状态。另核对整段回收后取得 12 MiB 的场景、8 MiB 等待状态，以及 R=A+W+F 的分类恒等式。原始实验属于 CPU 合成状态，性能部分使用教学操作次数，GPU 计时与真实显存分配列为待验证范围。

| 严重程度 | 实际发现 | 修正与复验 |
| --- | --- | --- |
| MEDIUM（已修正） | 独立代码审查复现 `assert` 内的状态变更在优化模式下被删除 | 操作移到断言之前，实际释放量进入输出，加入优化模式回归，独立代理复验通过 |
| LOW（已修正） | 实验入口的导入分组触发已有 Ruff I001 | 调整分组后 Ruff check 与 format 检查通过 |
| LOW（已处理） | 暂存检查发现原始 unittest 红灯日志的一处子测试行尾空白 | 对该具体日志路径设置空白检查例外，原始输出字节纳入校验和 |

源码中的 `inactive_split` 排除了 expandable 段，正文保留了此范围。模型的 `release_free_segments` 不主动完成等待状态，真实默认池回收路径先同步事件，正文分别解释了两种行为。论文范围为官方工程机制，未引入具体论文算法。

## 第二轮，教学逻辑与表达

按“16 MiB 预算布局 → segment／block → 请求大小与空闲单块 → 三态核算 → 源码 → 调节与验收 → 练习”检查正文。8 MiB 模型段与真实默认 20 MiB 分支在开篇和源码节分别说明。前置篇为 01、18、24，首次出现的 MiB、best-fit、三态、requested 与 block 大小均有就地解释。

| 严重程度 | 实际发现 | 修正与复验 |
| --- | --- | --- |
| LOW（已修正） | 初稿“连续范围”未指明地址视角，容易把模型布局与底层物理页关系混在一起 | 明确写为地址连续范围，并说明 CUDA 驱动管理物理页映射；与 expandable 的映射说明交叉核对 |

检查了两个折叠答案及完整代码、起始代码和参考答案导入关系、异常行为、等待状态与普通段范围。面试追问覆盖基础、实现、选择和排障。图中文字与源文件一致性、全部本地链接和文件哈希由最终交付检查验证。

## 运行与验证记录

| 对象 | 实际范围与结果 | 原始证据 |
| --- | --- | --- |
| 测试先行 | 初始 23 项测试在未实现答案上失败，记录具体异常 | [初始红灯](evidence/runs/01-red-tests.txt)、[命令](evidence/runs/01-red-verification.json) |
| 优化模式故障 | 实际复现实验入口的必要调用被删除 | [优化模式红灯](evidence/runs/01-optimized-red-lab.txt) |
| 完整参考答案 | 24 项 unittest 通过，含 500 步随机轨迹与优化模式回归 | [测试输出](evidence/runs/02-tests.txt) |
| 代码规范 | 现有 Ruff 0.17.3，check 与 format 检查通过 | [Ruff](evidence/runs/03-ruff.txt)、[format](evidence/runs/04-format.txt) |
| 模型实验 | 复用、碎片失败与恢复、整段回收重试、等待完成、相同差值的三种布局 | [JSON](evidence/runs/05-model-lab.json) |

所有实际命令、退出码、执行时间与当前代码哈希见 [运行清单](evidence/runs/06-verification.json)。执行器为本机既有 ai-infra Python 3.12.16，PyTorch 2.15.0，CUDA 不可用。

## 发布结论

两轮自审与独立代码审查的已发现问题均已修正，未解决的 CRITICAL／HIGH 为 0。正文的核心机制有固定源码支撑，两道 CPU 基础练习有完整答案和真实验证。实际 CUDA 分配、CUDA event、真实设备碎片与性能仍待 GPU 环境验证。文章已本地归档，交付通过结果由 [检查清单](evidence/delivery-check.json) 记录。按 docs/DAILY_TASK.md 同步，最终 GitHub main 含提交后报告完成。[提交前核验](evidence/git-preflight.json)

## 勘误与更新

暂无发布后勘误。研究与审查期间的修正如上记录。
