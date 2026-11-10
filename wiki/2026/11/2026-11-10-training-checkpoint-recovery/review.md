# 审查记录

主题：训练中断后，checkpoint 怎样恢复一致的进度？

文章状态：published。审查日期：2026-11-10（Asia/Shanghai，UTC+08:00）。

第一轮技术证据与第二轮教学表达均由主执行者自审；Python 代码另有独立代理审查与限定复验，实际身份和调用限制见 [Python 审查](evidence/python-review.md)。

## 第一轮：技术事实与证据

检查第 5 步状态与 cursor = 10 的对应关系，复算 10 / 110 = 9.09%；对照真实 lab.json 的样本、参数差、LR 和随机状态。实际 fetch 官方 main 并固定完整 SHA；读取六个文件的状态、staging、write_data、finish 路径，与稳定 tag 作实际 diff。区分源码 main、稳定 tag 与运行包 build SHA。

| 级别 | 实际观察 | 修正与复验 |
| --- | --- | --- |
| MEDIUM（info），独立代码发现 | optimizer.load_state_dict 可复用输入快照中的 Tensor，使快照逻辑步数和矩不再属于同一时刻 | 先新增两项真实失败回归，再 deepcopy 装入；26 项最终测试通过，独立两项复验通过 |
| MEDIUM（info），主执行者自审 | FileSystemWriter.finish 会移除旧 metadata；不能从临时 metadata 改名推导整个旧／新目录原子切换 | 正文明确独立版本目录、等待完整保存后更新 latest 的应用策略，并说明未运行 DCP 或断电实验 |
| LOW（note），主执行者自审 | 初稿固定源码行尾引用超过 state_dict_saver.py 的实际末行 | 改为 L494–L519，归档范围和源码实际行数一致 |
| LOW（note），主执行者自审 | main 与稳定 tag 的 optimizer 文件存在 dtype 策略改动，概括两版行为相同会越过证据 | 保存完整相关 diff；只报告 saver／staging 完整文件相同，并明确 CPU float64 实验边界 |

保存间隔公式为自拟教学模型，本文不引入具体论文 checkpoint 算法，依据与原因在 references.md 中说明。所有源码片段保留行号、原始 blob 与选段哈希、许可证。没有把 GPU staging 或 DCP 阅读记成运行成功。

## 第二轮：教学逻辑与表达

沿“5 步保存、8 步中断”的具体例子，检查从已消费样本、Adam 历史到状态元组、文件发布、分片保存的推导。图中第 5 步 cursor 10、最终第 10 步 cursor 20 与实验一致，保留 Mermaid 图源。术语首次出现就地解释，前置 18—21 篇都有本地链接。

| 级别 | 实际观察 | 修正与复验 |
| --- | --- | --- |
| LOW（note），独立代码发现 | 练习一编辑模板后，测试仍导入答案文件，容易验收错实现 | 题目说明和模板都写明复制到 checkpoint_io.py 后再测试；独立复核关闭 |
| LOW（note），主执行者自审 | 缺项对照的漏 optimizer 同时丢失 optimizer 保存的 LR，表格容易让读者只归因于 Adam 矩 | 表格明确其学习率也丢失，不把该对照当作只改变单个数值字段 |
| LOW（note），主执行者自审 | 完整复现实验可能被理解为真实进程崩溃与断电验收 | 正文和验证说明都明确新子进程是真实的，但第 6—8 步丢失以丢弃内存模拟 |

两道练习有独立起始文件、完整答案、输入输出、验收条件、常见错误与运行命令；追问覆盖基础、实现、方案与排障。未使用不必要论文或 GPU 性能数字。教学自审通过。

## 运行与验证

| 对象 | 实际结果 | 原始证据 |
| --- | --- | --- |
| 测试优先 | 初始接口失败；快照污染回归修正前 2 项失败 | evidence/runs/red-tests.txt、restore-red-tests.txt |
| 最终 CPU 练习 | 26 项通过，退出码 0；5.664 秒 | [tests.txt](evidence/runs/tests.txt) |
| 新进程实验 | 五个新进程，全部状态恢复与基线一致，四个缺项对照产生差异 | [lab.json](evidence/runs/lab.json) |
| Ruff | check 与 format --check 退出码均为 0 | evidence/runs/ruff-check.txt、ruff-format.txt |
| Python 独立复验 | 两项回归通过，原 MEDIUM／LOW 关闭；无未解决发现 | [python-review.md](evidence/python-review.md) |
| 文件、链接、源码与校验和 | 归档后机器核验结果保存于交付检查 | [delivery-check.json](evidence/delivery-check.json) |

## 发布结论

未解决的 CRITICAL／HIGH：0。影响核心理解的 MEDIUM 已修复。CPU 基础练习、当日一手检索、源码固定、两轮自审及独立 Python 审查已满足。完整内容已本地归档并加入索引；交付校验通过且远端确认包含本提交后，才报告 Git 同步完成。

未执行边界：GPU、AMP、worker／梯度累积中途、多 rank DCP、跨拓扑数值、断电与真实强杀。图源与公式是教学机制说明，性能比例是估算。

## 勘误

暂无。
