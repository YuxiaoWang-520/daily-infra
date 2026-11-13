# 审查记录

主题为 torch.compile 的图捕获、graph break 与重新编译成本。审查日期为 2026-11-13，时区 Asia/Shanghai（UTC+08:00）。文章状态为 published，技术证据和教学表达已完成两轮主代理自审，Python 代码由独立通用代理审查。

## 第一轮，技术证据自审

执行者为主代理，方式为自审。

- [x] 主问题承接异步计时、融合和 CUDA Graph，覆盖重复调用与输入变化的工程场景。
- [x] 当天实际检索 PyTorch 2.15 官方资料，fetch 官方 main 并固定完整 SHA。
- [x] 核对稳定 tag、当天 main 与本机 torch build，明确结果的版本范围。
- [x] 追踪公开 compile 入口、Dynamo、用户后端交付、GuardedCode 与 C++ 缓存检查。
- [x] 实际重建 16 份源码片段，共 1018 行，核验原始 blob、片段哈希及许可证。
- [x] 实验使用真实 torch.compile，记录自定义后端交付和完整生成 FX 图。
- [x] 11 个场景完成 27 次数值对照，2 次 fullgraph 预期拒绝均被正确记录。
- [x] 两道练习有完整题面、起始代码、答案和真实测试记录。
- [x] 明确 CPU 捕获实验、优化代码生成与性能验证的范围。
- [x] 成本模型单位一致，教学表分别复算为 470、570、600 ms。

未发现需要阻断发布的技术问题。自审重点核对 graph break 的两段交付与 scalar guard 的多份结果，确认缓存条目可再次匹配旧值，逐项对照 dynamic 的三模式及 0／1 特化结果。Python float 分支在当天 main 存在变化，正文将累计计数限定为实际构建和显式参数。

### 实际补充

| 程度 | 观察 | 处理与验证 |
| --- | --- | --- |
| LOW（note） | 题目允许空 Tensor 与 float64，原测试未直接组合覆盖这组边界 | 新增独立期望值的测试，覆盖 (0,3)、(2,0)、(0,0)、(1,3) 及 float64，22 项测试通过 |
| LOW（note） | 旧自定义后端文档地址返回 404 | 重新实际检索并读取版本化 torch.compiler_custom_backends.html，保存失败和恢复后的来源元数据 |

首次待实现答案实际运行 20 项测试，产生 17 项错误。补全实现后原 20 项通过，随后增加 0／1 特化及空 Tensor／float64 边界，共 22 项通过。[初始红灯](evidence/runs/01-red-tests.txt)、[最终通过输出](evidence/runs/04-tests.txt)

## 第二轮，教学表达自审

执行者为主代理，方式为自审。

- [x] 从全 1 的 (2,3) Tensor 和 scale=2、3、2 开始，读者可以先复算输出。
- [x] FX、后端与 guards 都在实际例子中解释，前置文章链接完整。
- [x] 显式分界展示两段图，复用旧 scalar 展示多份缓存，保持一条问题链。
- [x] 图解箭头与缓存／捕获关系一致，保留可重建 Mermaid 图源。
- [x] 公式先定义成本归属和单位，全部毫秒数标明教学假设。
- [x] 优化解释包含编译事件、边界、融合机会、动态约束与 Tensor 参数管理成本。
- [x] where 改写解释整体求和、零条件及两侧表达式求值范围。
- [x] 练习在答案前给定输入、输出、约束与验收方式。
- [x] 面试追问覆盖基础、实现、方案选择和排障。
- [x] 独立代码审查与两轮自审如实区分。

修正生成文本中的多余数学定界符转义，检查显示公式只有一组有效定界符。正文内 Mermaid 与图源逐字匹配。图交付次数和具体 kernel／性能由各自证据说明，优化后端验证保持待执行范围。

## 运行与环境

机器为 wyx-macmini.local，Python 3.12.16，PyTorch 2.15.0，macOS 27.0 arm64。解释器为 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python，执行时设置 PYTHONDONTWRITEBYTECODE=1。沿用现有环境与 Ruff 0.17.3。

    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff check --no-cache code
    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff format --check --no-cache code
    TORCH_LOGS="graph_breaks,recompiles,guards,dynamic" PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py --output /tmp/compile-lab-new.json

| 验证对象 | 实际结果 | 原始记录 |
| --- | --- | --- |
| unittest | 22 项通过，退出 0 | [测试](evidence/runs/04-tests.txt) |
| 正式归档复验 | 在最终文章目录运行 22 项测试通过，退出 0 | [输出](evidence/runs/10-archive-tests.txt)、[执行记录](evidence/runs/10-archive-verification.json) |
| Ruff check／format | 7 个文件通过，退出 0 | [Lint](evidence/runs/06-ruff.txt)、[格式](evidence/runs/06-format.txt)、[版本](evidence/runs/06-ruff-version.txt) |
| 编译前端实验 | 11 个场景、27 次数值对照、2 次预期拒绝，退出 0 | [JSON](evidence/runs/02-compile-lab.json)、[stdout](evidence/runs/02-compile-lab.stdout.txt) |
| 原始诊断日志 | 记录 guards、recompiles、graph_breaks、dynamic | [stderr](evidence/runs/03-torch-logs.txt) |
| 实际运行代码 | 7 个 .py 保存 SHA-256 与命令、环境、时间、退出码 | [运行清单](evidence/runs/08-verification.json) |
| 独立代码审查 | 6 项定向测试及缓存隔离、非预期异常传播实测通过 | [审查范围](evidence/python-review.md) |

## 发布结论

未解决的 CRITICAL／HIGH 问题为 0。核心论断有当前官方资料、固定源码与真实 CPU 行为证据，基础练习及两轮自审达到正文所声明范围的发布标准。

本篇的自定义后端返回 GraphModule.forward。优化代码生成、Inductor 性能、CUDA 执行与真实服务吞吐仍待专项验证。固定语言审查代理不可用后采用通用代理独立代码审查，实际范围已保留。

正式归档后已核验全部本地链接、25 篇连续索引、当天仅 1 篇、图源、数学定界符、运行代码哈希、16 份源码重建和各场景输出。SHA-256 清单覆盖 48 个文件，实际 shasum 校验退出 0。清单自身、交付结果和校验命令输出各自保存，避免循环包含。[交付检查](evidence/delivery-check.json)、[校验输出](evidence/runs/11-checksums.txt)

## Git 同步状态

2026-11-13 11:30—11:31（UTC+08:00）的首次提交前复核发现目标仓库可见性为 public，与当时 `docs/DAILY_TASK.md` 的私密要求冲突，因此当时暂停了提交和推送。同日用户明确授权按公开仓库运行，并要求更新定时任务、仓库规则后提交推送本篇。

已更新并回读核对 `ai-infra-wiki` 的已保存指令，仓库内日更和按需发布规则已按公开仓库运行。本篇复用已完成的内容、两轮审查和独立代码审查，连同本次明确授权的规则修订一起提交。提交前核验记录保留目标、实际可见性、身份和远端状态，最终同步结果由推送后的 GitHub `main` 核验确认。[提交前核验记录](evidence/git-preflight.json)

本次 MAIN 自审核对了 7 份发布规则、21 个本地链接，以及定时任务指令的精确变更范围。固定仓库、Git 身份、审查与验证、同日幂等、失败恢复和报告条件均通过检查。本篇 22 项 CPU 测试在恢复提交前再次运行通过，代码哈希与已归档实验一致。

暂存检查发现上游 fetch 原始输出的末尾空行和 TorchDynamo 原始日志的行尾空白。本篇 `.gitattributes` 对这两个具体输出路径设置空白检查豁免，原始字节纳入 `SHA256SUMS` 核验。

## 勘误

暂无。
