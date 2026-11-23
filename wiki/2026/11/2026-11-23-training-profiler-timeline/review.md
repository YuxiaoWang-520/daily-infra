# 审查记录

主题：训练 profiler 的 CPU 提交、GPU 执行与通信等待。

文章状态：published。实际审查时间见 [self-review.json](evidence/self-review.json) 和 [technical-audit.json](evidence/technical-audit.json)，时区为 Asia/Shanghai。

技术与教学两轮均由主执行者自审。Python 另有独立通用代码审查，范围仅限本篇 5 个 Python 文件；所需专项代理在执行前失败，未冒称其成功。

## 第一轮：技术事实与证据

- [x] 核心场景对应真实训练观测；区分事实、机制假设和根因结论。
- [x] 实际更新 PyTorch 官方 main，并固定完整 SHA；稳定 tag 另读，安装包构建另记。
- [x] 从 Python 状态机追到 C++ 回调、字段记录、关联和统计语义，说明普通后端与可选分支。
- [x] 本篇不介绍论文算法，基础集合／DAG 模型的定位已说明。
- [x] 15 ms 例子、区间集合、图中数字和固定时序反事实可复算。
- [x] 两题完整答案已实跑；22 项测试普通／优化模式通过，Ruff 检查通过。
- [x] CPU trace 实测、合成 GPU 时间和未运行部分明确区分，没有 GPU 性能或通信根因结论。
- [x] 片段、完整许可、阅读笔记、运行记录、代码审查与归档边界齐全。

### 实际发现、修正与复验

| 严重程度 | 实际观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | 初版投影从 `end-start` 恢复 duration，`ts=1e15,dur=0.01` 变成 0 | 数据结构保留原始 duration；投影直接使用原值 | [首次复现](evidence/runs/initial-duration-reproducer.json) 与最终 duration 回归测试 |
| MEDIUM（info） | 独立审查用 `2**53+1` 端点复现窗口 2、覆盖 4、未覆盖 -2 | 拒绝有损整数转换和浮点转换溢出，窗口统一使用已验证数值 | [失败回归](evidence/runs/reviewer-regressions-red.txt)，最终测试与独立增量确认 |
| LOW（note） | 独立审查发现合法原时间戳平移后起点可溢出 | 检查平移起点有限 | 原例加入同一回归，测试通过 |
| LOW（note） | 增量审查发现起点有限时，平移后的终点仍可能溢出 | 同时验证平移起点加 duration 有限 | [第二个失败复现](evidence/runs/reviewer-end-overflow-red.txt)，最终两种子例与独立确认通过 |
| LOW（note） | 首次 lint 报告异常类型与字面量问题 | 非法类型使用 TypeError，非法数值使用 ValueError，移除冗余构造 | [初次 lint](evidence/runs/initial-ruff-check.txt) 与 [最终检查](evidence/runs/ruff-check.txt) |

技术核验还将 19 个片段、源文件 SHA 和许可证逐字节与固定 Git 对象比较，验证 main／stable 许可证一致、最终代码与独立审查摘要一致，并复核真实 CPU 的 140 个保留事件及阶段计数。没有用同一实现的公式输出充当唯一 oracle；单元测试和独立审查分别使用整数格与随机集合对照。

## 第二轮：教学逻辑与表达

- [x] 从 15 ms 小例子进入，术语在所需位置解释；前置文章链接有效。
- [x] 先数范围，再讲集合、关键路径和实现；源码段落都对应前面的具体疑问。
- [x] Mermaid 数字与合成输入一致，图源与正文代码块逐字一致；明确其不是实测 trace。
- [x] 工程表列出适用条件、代价和下一项可验证证据。
- [x] 两题输入输出、验收、完整答案、命令和常见口径错误分开说明。
- [x] 面试追问覆盖基础、实现、选择和排障。
- [x] 交付阶段再次检查最终路径、索引、状态与链接。

### 实际观察与修正

| 严重程度 | 实际观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | 手算部分用 ms，练习与 JSON 用 μs，单位说明原来出现得太晚 | 在首个集合例子后立即给出 `1 ms=1000 μs`；保留显示单位与文件单位的区分 | 图、表、练习输入与结果逐项换算一致 |
| LOW（note） | 实现段落开始引入 Kineto、CUPTI，通信段落使用 collective／rank，初学者需要最小解释 | 在首次需要处补充后端、采集接口、集合通信和进程身份说明 | 重新按例子→机制→源码顺序通读，没有用术语表替代推导 |

其他关键通读检查：累计 17 ms、并集 11 ms、重叠 6 ms、通信独占 2 ms 均有明确分母；2 ms 不被说成实际优化收益；未覆盖 4 ms 不被说成空闲；CPU self 不被说成核心利用率。真实数据投影的删减与未做渲染明确写在答案旁。

## Python 独立审查的实际范围

`/root/nov23_python_review` 使用所需 `python-reviewer` 角色，但在执行前返回 `gpt-5.4` 不受当前账户支持。随后 `/root/nov23_python_fallback` 以独立通用审查角色读取冻结的 5 个 Python 文件，运行只读、临时目录内的检查，并报告上述问题。不能据此声称专项代理已执行。

独立检查报告包括：3,000 个随机离散区间案例、13 个非法输入、生成器、不同进程／线程映射、原始 duration、真实 CPU 采样和数值、独立标量 SGD oracle，以及注入异常后的线程／采集器恢复。修复后只做针对相关差异的增量确认，最终源码摘要与运行清单匹配，无遗留发现。[完整结构化报告](evidence/python-review.json)

报告是审查者实际返回的检查说明，不是伪造的 stdout。主执行者另外保存了自己实跑的全部最终命令输出。

## 运行与验证

所有命令从本篇目录执行，使用现有 ai-infra 环境，不安装或升级依赖。清单中的可执行文件用 basename 记录，本机实际调用指定环境的 Python／Ruff。

| 对象 | 实际命令 | 结果 | 记录 |
| --- | --- | --- | --- |
| 单元与 CPU 集成测试 | `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v` | 退出 0，22 项通过 | [普通模式](evidence/runs/unit-tests.txt) |
| 优化模式 | `PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v` | 退出 0，22 项通过 | [优化模式](evidence/runs/optimized-tests.txt) |
| 静态与格式 | `ruff check code`；`ruff format --check code` | 均退出 0 | [检查](evidence/runs/ruff-check.txt)、[格式](evidence/runs/ruff-format.txt) |
| 两题答案 | `PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir evidence/runs` | 退出 0，区间结果正确、真实 CPU steps 2／3、权重和损失一致 | [完整输出](evidence/runs/answers-run.txt)、[结果](evidence/runs/results.json) |
| 源码／图／数据 | 逐项 Git 对象与摘要比对、图源和数字核对 | 19 个片段／500 行、许可、5 个代码摘要匹配 | [技术核验](evidence/technical-audit.json) |

实际环境：Python 3.12.16、PyTorch 2.15.0（构建 SHA 独立记录）、Darwin arm64、CPU、单线程、float64、合成 `X[16,8]` 与 `W[8,4]`，每次训练 4 步。全部环境和命令退出码在 [verification.json](evidence/runs/verification.json)。

## 发布结论

两轮自审与独立通用代码审查通过，已修复全部发现，未解决 CRITICAL／HIGH 为 0。CPU 基础练习、源代码和教学例子具备足够证据。没有 CUDA／NCCL／多机验证、GPU 性能结论、真实关键路径恢复或查看器视觉渲染；这些限制属于选定范围，未被写成通过。

交付检查负责最终文件、链接、隐私扫描和索引一致性：[delivery-check.json](evidence/delivery-check.json)。Git 核验另见 [git-preflight.json](evidence/git-preflight.json)；其中记录的是本次发布前基线，不尝试把未来提交 SHA 写入自身文件。是否已经同步以实际 GitHub main 查询结果为准，不以本记录代替推送确认。

## 勘误与更新

暂无。
