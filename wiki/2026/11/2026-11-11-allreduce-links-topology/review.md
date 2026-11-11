# 审查记录

主题：AllReduce 的数据怎样沿链路流动，为什么增加 GPU 后通信反而更慢？

文章状态：published。审查日期：2026-11-11（Asia/Shanghai，UTC+08:00）。

技术证据与教学表达两轮均由主执行者自审。Python 代码另有独立代理审查与定向复验，真实范围和专用模型调用限制见 [python-review.md](evidence/python-review.md)。

## 第一轮：技术事实与证据

逐项复算 4 个 rank 的输入、第一轮接收值、RS 所有权、AG 最终结果；核对发送量 48 字节／rank、192 字节／全组。核对论文 §2.2 的大消息模型、§3 的下界条件、§4.1 图 1 与定理 1。实际渲染查看原稿第 12—13 页，没有只读论文摘要。

沿当日固定 NCCL 的 API、入队、准备与调优、常规 Ring kernel 读取代码；核验 benchmark 的换算函数和许可证。master 与稳定 tag 解引用到同一完整 SHA，文档缓存与运行包版本另行标记。

| 级别 | 实际观察 | 修改与复验 |
| --- | --- | --- |
| MEDIUM（info），独立代码发现 | 成本模型中间 float 溢出与单 rank NaN，违反有限输入下的预期边界 | 新增 5 项失败回归，保留整数／有理数运算，最终显式转 float；修后 28 项通过，独立 5 项复验关闭 |
| MEDIUM（info），主执行者自审 | 原论文带宽最优依赖无争用映射，不能直接据环形逻辑图推断物理 NIC 无争用 | 明确假设，加入同 NIC 多流累加的例子；模型得到 8.083 与 16.136 ms，均标估算 |
| LOW（note），主执行者自审 | today 源码目录已重构，历史 enqueue.cc／graph/tuning.cc 路径不适用于该 commit | 按实际 src/enqueue/enqueue.cc 与 src/tuning/ 路径定位并归档原文 |
| LOW（note），主执行者自审 | release 发布日期为 UTC，且文档提取缓存标注 2.32.3，容易混同上海日期与今日 release | 正文标明 UTC，版本和 research log 分开记录请求日、获取日、源码与 release |
| LOW（note），主执行者自审 | busbw 若被理解成实际 NIC 计数，会错误解释层次拓扑 | 固定读取 AllReduceGetBw，并结合官方项目讨论说明归一化口径 |

模型公式包括清楚的符号、单位和限制；CPU 只验证语义。没有把 Gloo 当作 NCCL 实现验证，没有将论文旧集群阈值迁移到当前 GPU。全部核心证据有本地笔记、必要源码片段与可复现结果。

## 第二轮：教学表达

按输入向量、第一轮消息、分块所有权、发送字节、启动轮数、共享 NIC、当前实现、动手验证逐层阅读。图与正文中的 rank、块、输入和结果一致。前置 03、04、06、17、21 篇有本地链接，正文从零解释 rank、NIC、channel、protocol 与 slice。

| 级别 | 实际观察 | 修改与复验 |
| --- | --- | --- |
| LOW（note），主执行者自审 | 朴素树基线易让读者以为 NCCL Tree 在大消息必输 | 明确整块 binomial 模型，补充分块、双树、流水线及 recursive halving／doubling 的边界 |
| LOW（note），主执行者自审 | 一个逻辑块可能被理解为一个设备传输或 kernel | 解释通道、尾块和流水片段，区分教学轮次与实际 GPU 同步 |
| LOW（note），主执行者自审 | 代码模板验收若仍固定导入答案，可能验收错文件 | 两题模板和题目均说明复制到答案文件后测试，完整答案另列 |
| LOW（note），主执行者自审 | 小数毫秒表容易被误当成实测 | 开篇、表前、答案、验证记录都注明自拟参数模型，实测仅报告 CPU 正确性 |

两道练习均有输入输出、约束、起始文件、完整实现、验收和常见错误；追问涵盖基础、实现、选择与排障。两幅原创 Mermaid 图有重建源文件。教学自审通过。

## 运行与交付

| 验证对象 | 实际结果 | 原始记录 |
| --- | --- | --- |
| 初始接口 | 23 项运行、45 errors（含 subtest） | evidence/runs/red-tests.txt |
| 数值缺陷回归 | 修正前 1 failure、4 errors；修后全部通过 | evidence/runs/numeric-red-tests.txt |
| 最终基础练习 | 28 项通过，退出码 0 | [tests.txt](evidence/runs/tests.txt) |
| CPU／Gloo | 4 个真实进程均得到正确 SUM，与模拟器一致 | [lab.json](evidence/runs/lab.json) |
| Ruff | check、format --check 均退出 0 | evidence/runs/ruff-check.txt、ruff-format.txt |
| 独立代码复验 | 5 项新回归通过，原 MEDIUM 关闭 | [python-review.md](evidence/python-review.md) |
| 归档完整性 | 归档后核验链接、图源、代码哈希、源码与证据校验和 | [delivery-check.json](evidence/delivery-check.json) |

## 发布结论

未解决的 CRITICAL／HIGH 为 0，影响核心理解的 MEDIUM 已修复。两轮自审、独立 Python 审查、必要测试与一手证据满足标准。完整内容已本地归档并加入索引；交付校验通过后按白名单提交，远端确认包含本提交后再报告同步完成。

剩余边界：没有 CUDA／NCCL／RDMA／跨节点或 GPU 性能实测；模型不预测所有真实网络时序，整数练习不验收浮点归约顺序或固定宽度溢出。论文原文未再分发，保留自己的笔记、准确定位与原稿哈希。

## 勘误

暂无。
