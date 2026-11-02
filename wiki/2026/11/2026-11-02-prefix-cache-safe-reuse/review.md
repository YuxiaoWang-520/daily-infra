# 审查记录

主题：多个请求什么时候能安全复用同一段前缀？

状态：published，正文、两道基础练习与最终归档路径验收通过。日期：2026-11-02（Asia/Shanghai）。技术核对与教学复读自 `11:34+08:00` 起进行；归档验证时间以运行记录为准。

技术与教学两轮均由主代理自审，不称为独立审查。五个原创 Python 文件由真实独立只读代理另行审查；专职固定模型不受支持、替代角色和修复后接受方式见 [Python 记录](evidence/python-review.md)。

## 第一轮：技术、证据与范围

- [x] 主问题绑定长文档反复提问、固定说明和多轮聊天，不以细枝末节选题。
- [x] 当天实际 fetch 官方 main，固定完整 SHA、默认分支和稳定发行元数据；13 个原始来源文件逐个保留版本与 hash。
- [x] 追踪 EngineCore、Scheduler、KVCacheManager、coordinator、FullAttentionManager、BlockPool 的查找／分配／释放路径与生效条件。
- [x] 实际阅读 PagedAttention 原论文固定 v1 的对应章节、图文字及实验范围，与当前自动哈希和进阶部分块路径分别核对。
- [x] 因果依赖、H 的对齐公式、最后位置重算、容量手算和图示一致；区分 key 身份、允许读取与数据就绪。
- [x] 两题分别有完整契约、起始接口、独立参考答案、手算反例和真实 CPU 验证。
- [x] 没有把元数据测试称为模型精度或 GPU 性能测试；没有引用历史吞吐倍率代替当日结果。

### 实际发现与复验

| 程度 | 观察与证据 | 处理 | 复验 |
| --- | --- | --- | --- |
| MEDIUM，独立 Python 发现 | 同 key 的 warm＋active 副本中固定选最小 ID，会不必要地占用空闲候选并误报容量不足 | 先加回归复现，再优先共享 active 副本；同状态按 ID | [修复前失败](evidence/runs/active-copy-red.txt)，修复后完整 29 tests、lab、Ruff 通过；无残留问题 |
| 核验项，无新增代码缺陷 | 当前源码已存在受模型与管理器条件限制的细粒度命中；官方设计概述仍有“只缓存满块”的简述 | 基础例子明确三种粒度都为 B；进阶部分单列条件与未执行边界 | 对照 single-group coordinator、partial enable 条件、alignment 选择及第二阶段 probe |
| 核验项，无新增代码缺陷 | 首个 miss 后仍可能存在缓存中的后缀键 | 正文明确是连续前缀查找策略，不把源码注释概括成后缀键一定不存在 | 练习以 `{后缀键}`、`{首块键,第三块键}` 构造反例并断言命中边界 |
| 核验项，无新增代码缺陷 | 模型更新、布局、输入处理和授权域并不全放在当前每个块的 tuple 中 | 区分引擎池域、extra keys 和失效管理；不把 salt 说成完整权限控制 | 核对 LoRA name/path、embedding、首块 salt、group ID 及教学 namespace 差异 |
| 核验项，无新增代码缺陷 | 引擎可在调度阶段登记计划块，查到键不独立证明 GPU 完成 | 教学池用显式 complete，仅验证其保守顺序模型 | 当前 cache_blocks 调用与正文、代码契约一致，不冒称异步执行验证 |

容量例子独立按整数复算：每 token 为 131072 bytes；10 条、共同 1024 个位置、各 128 个后缀，理想未共享／共享分别为 1440／288 MiB，节省 1152 MiB。是指定几何的估计，不是显存实测。

## 第二轮：教学逻辑与表达

- [x] 从同一份手册配不同问题开始，先解释复用对象，再进入数字 token 和哈希。
- [x] 用 8 个／7 个共同位置、前文变化但局部重复、完整命中重算等具体例子引导。
- [x] 就地解释 tokenizer、embedding、适配器、隐藏向量、logits、hash、salt、引用计数和 TTFT／ITL。
- [x] 公式变量、完整块假设、不同隔离范围不适用公式等条件清楚。
- [x] 两幅 Mermaid 与独立图源一致，标注为教学图而非 trace；未声称完成像素级渲染验收。
- [x] 源码节逐个回答前面的问题，主线不要求先理解混合模型、投机或跨实例通信。
- [x] 练习不需先读答案，折叠详解有推导、预期状态、运行方法、常见错误和限制。
- [x] 面试追问覆盖基础、依赖、实现、容量、隔离与验证。

### 实际发现与修正

| 程度 | 复读观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| LOW | “64 位十六进制字符串”可能把字符数与 bit 数混淆 | 改成 64 个十六进制字符，每字符 4 bit，共 256 bit | 题目与 SHA-256 输出表达一致 |
| LOW | 练习二的 pool 不接收输入 token 长度，容易误以为其 acquire 自带 L-1 生成上限 | 正文与代码说明明确上层查找策略负责该上限，池只管理已经选定的完整块 | 两题各自完整，但没有宣称直接拼成整个引擎接口 |
| LOW | 相同内容但不同隔离域这一行不满足 H 公式的假设 | P 列保持数值 10，并明说此行不满足允许共享的前提 | 公式、条件与五组实际命中例子相符 |

## 运行与归档证据

| 对象 | 环境／方法 | 实际命令 | 结果 |
| --- | --- | --- | --- |
| hash 与 pool 测试 | Python 3.10.13、macOS arm64、标准库；独立计数与反例 | `python -m unittest discover -s code -p test_*.py -v` | 29 tests 通过；[原始输出](evidence/runs/tests.txt) |
| 五组命中与生命周期轨迹 | 合成 token／键标签，真实 CPU 元数据执行 | `python code/run_lab.py` | 断言通过、退出 0；[JSON](evidence/runs/lab.json) |
| 静态与格式 | 已有 Ruff 0.15.9 | `python -m ruff check --select E,F,I,UP --ignore E501 code`；`python -m ruff format --check code` | 通过；[lint](evidence/runs/lint.txt)、[format](evidence/runs/format.txt) |
| 命令、cwd、解释器、时间 | 归档路径下重新执行为最终依据 | 详见机器可读记录 | [summary.json](evidence/runs/summary.json) |
| 原始来源与代码／图源完整性 | source manifest、SHA-256、链接和行号 | 归档检查与 `shasum -a256 -c evidence/SHA256SUMS` | [归档检查](evidence/runs/archive-check.txt)、[hash 复核](evidence/runs/checksums.txt) |
| 发布交付 | 上海工作日、当日唯一文章、索引、图源、最终路径运行与内容范围 | 最终交付检查 | [实际记录](evidence/runs/delivery-check.txt) |

SHA 行号检查只能证明引用在对应原文范围内，不替代语义审查。原文提取与校验保留上游版权／许可证；原创输入和运行日志均为合成，没有内部项目或真实用户资料。

## 发布结论

- 未解决 CRITICAL／HIGH：0；影响主线的 MEDIUM：0。
- 独立 Python 发现已修复并回归，技术与教学两轮自审通过。
- 内容达到发布标准；最终目录复跑、归档校验与索引一致性通过。GitHub 远端确认后才能报告同步完成。
- 未执行：引擎安装／启动、真实 KV、GPU／完整模型、异步与并发、跨实例传输、服务性能和真实访问隔离实验。
- Git 只纳入本篇目录、Wiki 索引与必要 ROADMAP 链接；不纳入 playground、缓存或无关文件。同步结果由最终任务回执报告，避免自引用尚未生成的提交。

## 勘误

暂无。
