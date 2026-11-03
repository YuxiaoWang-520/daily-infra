# 审查记录

主题：训练的梯度为什么需要通信，通信能否与计算重叠？

状态：published，正文与练习已审查，最终目录复验和交付检查通过。日期：2026-11-03（Asia/Shanghai）。源码与数字核对贯穿研究过程，技术／教学整篇复读自 `11:18+08:00` 起进行；最终运行时间以记录为准。

技术证据与教学表达两轮由主代理自审，不称独立审查。六个原创 Python 文件及其契约另有真实独立只读审查；专职角色模型不受支持、替代角色、执行范围和修复接受方式在 [Python 记录](evidence/python-review.md)。

## 第一轮：技术与证据

- [x] 从单参数、两个样本的损失／梯度／SGD 推到数据并行，没有把多 rank 的一致性冒充全局目标正确。
- [x] 当天获取官方 PyTorch main、核验默认分支、完整 SHA、稳定元数据，保留 9 个原始文件与许可证。
- [x] 跟踪初始化、hook、ready、pending、next_bucket、除数、SUM hook、future 与自动 finalize；区分按序提交与串行通信模型。
- [x] 核对旧设计说明 v1.4、原论文 v1.5、当前 main、稳定 v2.15.0 元数据与实跑 2.9.1 的不同范围。
- [x] 读取原论文相关章节／算法与图，实际下载并提取 PDF、渲染检查第 5 页；保存原创笔记与获取 hash。
- [x] 检查全局分母、`R*n_r/N`、`/K` 适用条件，以及跨样本耦合和浮点顺序边界。
- [x] 两题有完整契约、起始代码、独立答案与实际 CPU 测试；额外真实双进程 Gloo 验证数值，不宣称 GPU 性能。
- [x] 时序单位、桶 ready、传输成本与尾部能手算；字节有效速率不是网卡标称带宽。

### 实际发现与处理

| 程度 | 观察与证据 | 修正／处理 | 复验 |
| --- | --- | --- | --- |
| LOW，独立 Python 发现 | 大整数在 `size/rate` 处先抛 OverflowError，与题目派生时间 ValueError 契约不符 | 加失败回归后捕获溢出并转换；serial 汇总同样处理 | [修复前失败](evidence/runs/overflow-red.txt)，修复后 24 tests、lab、Ruff 通过 |
| MEDIUM，技术自审 | 仅说损失可加，容易把局部 BatchNorm 等不同计算误认为拼接大 batch 完全等价 | 补充同一单样本函数／跨样本耦合的边界；本实验无 BN、dropout | 目标推导与线性模型实验一致，不外推整模型逐 bit 等价 |
| LOW，技术自审 | 实跑参数是 `0.24999999999999994`，表格写 0.25 若不说明，易误读为原始逐 bit 数值 | 写明表格舍入、JSON 原数值与 isclose 的 `1e-12` 容差 | 表中五组梯度／参数均对照两 rank JSON，无不符 |
| 核验项 | 当前设计页仍说基于 v1.4；实际默认初始桶与后续重建不同 | 明确旧文档、论文与当前源码层次，写出实际分支条件 | 原文版本提示、初始策略、rebuild 条件与调用一致 |
| 环境问题，已复验 | 首次 Gloo 在 DDP 参数校验处 30 s 超时 | 增加诊断归约、设置本机 lo0、设 60 s 超时，后续实际通过；未作唯一原因归因 | [环境摘要](evidence/probe-environment.md)、完整成功命令／输出；独立审查者也实际跑通 |

四梯度例子独立复算：每份 4 MiB，两个 rank 就绪均为 `[2,4,6,8] ms`，固定成本 2 ms、有效率 4 MiB/ms。四小桶／两中桶／一大桶完成为 `14/12/14 ms`，串行基线为 `20/16/14 ms`；慢 rank 最后一份 12 ms 时，两中桶完成为 16 ms。都是假设输入的算术，不是性能实测。

## 第二轮：教学表达

- [x] 从两个副本 w 分叉开始，先介绍 loss、求导、梯度和 SGD，再引入 rank／AllReduce。
- [x] 先手算等量例子，再用 1／3 样本反例引出全局分母，没有直接抛缩放公式。
- [x] 先用两桶具体时段，再写 ready、有效开始与完成的递推。
- [x] 两幅图的对象、时间与箭头清楚，保存完全一致的 Mermaid 源码。
- [x] 每段源码回答前面的问题，进阶编译／手动收尾放在主线之后并注明未执行。
- [x] 练习输入输出独立完整，详解给出手算、答案路径、命令、错误方式和真实输出。
- [x] 工程取舍连接常见慢 rank、桶大小、累积缩放与带宽／尾部分析，未用冷僻细节充当主线。
- [x] 面试追问覆盖基础、数学、实现、优化和排障。

### 实际观察与修正

| 程度 | 复读观察 | 修改 | 复核 |
| --- | --- | --- | --- |
| LOW | 首版时序图采用 Gantt 的 Unix 毫秒坐标，需额外解释绘图机制，分散初学者注意 | 改成框内写明时间段的依赖图，显式指出 4—8 ms 的重叠 | 图源与正文完全一致；没有声称像素渲染验收 |
| LOW | `beta` 只叫“有效速率”可能让读者直接代入网卡带宽 | 说明按桶输入字节定义、实际 AllReduce 算法与拓扑成本不在模拟内 | 题目与正文单位一致，模型结果不称实测 |
| 核验项 | 数据平均、有效 token 平均、序列等权等目标可能混淆 | 强调先定义目标再选分母；题目只支持正有效计数，空 rank 不支持 | 未把业务上所有目标强行改成一种 |

## 运行与归档证据

| 验证 | 实际命令／方法 | 结果与证据 |
| --- | --- | --- |
| 两题及边界 | `python -m unittest discover -s code -p test_*.py -v` | 24 tests 通过；[原始输出](evidence/runs/tests.txt) |
| 算术与合成时序 | `python code/run_lab.py` | 退出 0，梯度与五种时序断言通过；[lab JSON](evidence/runs/lab.json) |
| 静态／格式 | 已有 Ruff：check E/F/I/UP，忽略长行；format --check | 通过；[lint](evidence/runs/lint.txt)、[format](evidence/runs/format.txt) |
| 真实 CPU 两进程 | `GLOO_SOCKET_IFNAME=lo0 python code/ddp_probe.py --output evidence/runs/ddp.json` | 初始 SUM=3，五组默认 DDP 梯度／SGD 参数通过；[JSON](evidence/runs/ddp.json)、[原始输出](evidence/runs/ddp.txt)、[命令](evidence/runs/ddp-command.json) |
| 命令与环境 | 最终目录复跑为交付依据 | [summary.json](evidence/runs/summary.json)：Python 3.10.13、macOS arm64、PyTorch 2.9.1、CUDA false |
| 源码／代码／图与链接 | 来源 hash、固定行号、核心校验和、索引与日期检查 | [归档检查](evidence/runs/archive-check.txt)、[校验和](evidence/SHA256SUMS)、[复核](evidence/runs/checksums.txt)、[交付检查](evidence/runs/delivery-check.txt) |

初次 Gloo 失败只有实际工具输出的摘要，未伪称保存完整 stderr；成功复验与最终运行有完整捕获。上游原文保留版权，实验数据均为本篇合成，无内部工程代码、日志或真实训练数据。

链接检查覆盖原创 Markdown 与固定源码行号；上游原文的内部链接按原字节保留，不把这 9 个文件称为整个上游文档站点的镜像。行号范围检查不能替代前述语义核对。

## 发布结论

- 未解决 CRITICAL／HIGH：0；影响主线的 MEDIUM：0；独立 LOW 已修复。
- 技术与教学两轮自审通过，独立 Python 审查完成，修复由主代理回归接受。
- 最终路径测试、Gloo、归档完整性和索引一致性通过，达到发布标准；GitHub 远端确认后才能报告同步完成。
- 未执行：当前 main／稳定 tag 的安装构建，CUDA／NCCL、真实 overlap、性能、通信次数、完整模型、收敛、join、编译及手动收尾。
- Git 白名单只含本篇、Wiki 索引与必要 ROADMAP 更新；保护 playground 和其他工作。远端提交由最终回执记录，不在文档自引用待生成 SHA。

## 勘误

暂无。
