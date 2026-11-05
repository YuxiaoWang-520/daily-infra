# 审查记录

主题：用重算换显存的代价如何计算？

文章状态：published（内容审查通过并归档；Git 同步由任务单独核验）。审查日期：2026-11-05（Asia/Shanghai）。

技术审查执行者与方式：主执行者 Codex 自审；另由独立代理对本篇 Python 代码做限定范围审查。

教学审查执行者与方式：主执行者 Codex 自审。没有把两轮正文自审称为独立审查。

## 第一轮：技术事实与证据

- [x] 主问题对应激活占用导致的训练 OOM，以及容量与重算代价的工程取舍。
- [x] 实际 clone 当日官方 PyTorch main 并固定完整 commit，另 fetch 稳定 tag；相关源文件、许可证与逐字节清单在本篇归档。
- [x] 追踪非重入入口、保存输入、pack／unpack、重算、early stop 和 RNG 恢复路径，固定链接与归档行号对应。
- [x] 实际阅读原论文 Algorithm 1、§4.3 公式（1），渲染并查看第 5 页；区分静态分段算法、PyTorch hooks 实现与本文槽位代理。
- [x] 手算 `x=2,w=3` 的梯度 24，复算 `n=16,s=4` 的 8 MiB 和假设时间 64.8 μs，预算 10 MiB 的选项为 `s=8`。
- [x] 两道练习有模板、完整答案、明确输入输出与边界；组合测试实际运行 31 项并全部通过。
- [x] 运行 CPU FP64 Dropout 对照，逐张量比较输出、梯度和 SGD 更新，同时检查外部 RNG；没有把合成微秒数或保存载荷写成 GPU 性能。
- [x] 完成独立 Python 审查，修复其实际发现并由原审查代理定向复核关闭。

### 实际发现与修正

| 严重程度 | 观察到的问题与证据 | 修改 | 复验结果 |
| --- | --- | --- | --- |
| MEDIUM（info），自审 | 可单独表示的超大整数时长，在合成时可能先触发 `OverflowError`，没有按接口契约报告。先补测试，见 [overflow-red-test.txt](evidence/runs/overflow-red-test.txt)。 | 各有限时长先转浮点，再检查合成结果是否有限。 | 对整数和浮点 `10**308` 的用例通过，最终组合测试通过。 |
| MEDIUM（info），独立代码审查 | 超大段数乘零开销也可能在乘法前转浮点并溢出，见 [Python 审查](evidence/python-review.md) 与 [reviewer-red-test.txt](evidence/runs/reviewer-red-test.txt)。 | 用开销的整数比值先乘后除，保留零／极小开销的有限结果，真实溢出转 `ValueError`；补 3 例。 | 19 项容量测试通过，独立代理对新增 3 例复验通过并关闭问题；最终全套 31 项通过。 |

另外实际核查了统计口径：没有保留 Tensor 到统计列表；按 storage 去重；明确排除参数、输入、loss、RNG 和内部重算成本；未把 1408 bytes 到 384 bytes 称为总内存或 GPU 峰值变化。默认 determinism check 仅核对数量与元数据的边界也已与代码核实。

## 第二轮：教学逻辑与表达

- [x] 先从一组具体输入和反向求导解释“为什么要记住激活”，再进入分段、公式与源码。
- [x] 图解明确区间输入、最终输出与局部重算方向，保存可重建 Mermaid 图源；检查正文代码块与图源一致。
- [x] `ceil(n/s)+s` 被标成教学代理，解释过细、过粗、尾段和共享边界等限制，不把平方根规律当成通用最优规划。
- [x] 时间公式先定义 F、B、C、H，再解释 4／3 依赖哪些假设；所有微秒数明确是输入假设。
- [x] Dropout 对照回答“loss 一样是否足够”，参考答案解释参数更新的检查与负面对照。
- [x] 两题相互独立、题目不依赖先读答案；答案文件完整，提供命令、结果、输入约束、错误用法与验证边界。
- [x] 面试追问覆盖基础、实现、选择与排障；已有前置文章和下一题保持学习顺序。

### 实际观察与修正

| 严重程度 | 观察到的问题与证据 | 修改 | 复验结果 |
| --- | --- | --- | --- |
| LOW（note） | 初稿的 MiB、RNG、FP64 等词未都在首次使用处解释，对从零学习的读者有门槛。 | 补 `1 MiB = 2^20 bytes`、随机数生成器、64 位浮点数，并解释 detached alias 与 SGD 更新式。 | 顺序重读后，具体例子先于实现术语，前置文章链接保留。 |
| LOW（note） | 新增审查回归用例后，正文测试数仍为早期的 28 项。 | 更新为 19 项容量测试加 12 项 CPU 测试，共 31 项。 | 与最终 [tests.txt](evidence/runs/tests.txt) 和 [summary.json](evidence/runs/summary.json) 一致。 |

## 运行与验证记录

| 验证对象 | 实际环境 | 实际命令 | 退出码与结果 | 原始记录 |
| --- | --- | --- | --- | --- |
| 两题全部答案 | 本机 ai-infra Python 3.12.16、PyTorch 2.15.0 CPU FP64 | `PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p 'test_*.py' -v` | 0；31 项通过 | [tests.txt](evidence/runs/tests.txt) |
| 三组训练对照和容量表 | Mac mini、Apple M6、CPU，JSON 实验设置 1 个 CPU 线程 | `PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/run_lab.py` | 0；保存 RNG 的梯度与更新匹配，关闭时出现预期反例 | [lab.json](evidence/runs/lab.json) |
| Python 静态检查 | 既有 Ruff | `python -m ruff check code` | 0 | [lint.txt](evidence/runs/lint.txt) |
| Python 格式检查 | 既有 Ruff | `python -m ruff format --check code` | 0 | [format.txt](evidence/runs/format.txt) |

命令在当时的草稿文章目录执行，完整解释器、命令、退出码与耗时见 [summary.json](evidence/runs/summary.json)。保留真实测试耗时；它不是训练性能测量。归档后进行链接、图源、源码哈希和交付结构核验，结果见 [delivery-check.json](evidence/runs/delivery-check.json)；校验和复验见 [checksums.txt](evidence/runs/checksums.txt)。

## 发布结论

- 未解决的 CRITICAL／HIGH 问题：无；实际发现的 MEDIUM 已修复并复验。
- 达到内容发布标准：正文、两题完整代码、实际验证、固定源码、论文笔记和审查记录齐全，完成技术及教学两轮自审与独立 Python 代码审查。
- 限制：只验证固定 CPU eager；未运行 CUDA、编译、SAC、多设备、分布式、真实显存峰值、吞吐或长期收敛实验。Mermaid 图源做结构与一致性核验，未声称完成额外渲染测试。
- 同步范围仅本篇、Wiki 索引和路线中的必要链接。GitHub 是否已同步由发布任务在推送后独立核验，最终提交见 Git 历史与完成报告。

## 勘误与更新

暂无。
