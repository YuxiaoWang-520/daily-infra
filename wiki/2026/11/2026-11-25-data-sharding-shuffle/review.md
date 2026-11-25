# 审查记录

主题：训练数据分片、shuffle 与恢复游标。

文章状态：published。技术证据与教学表达两轮均由主执行者自审，代码另有独立通用复核；没有把自审称为独立审查。实际时间和判据见 [self-review.json](evidence/self-review.json)。

## 第一轮：技术事实与证据

- [x] 主问题连接分布式输入归属、读取效率、样本权重与中断恢复。
- [x] 当天实际 fetch 官方 main，记录完整 SHA；稳定 tag、安装包构建和文档缓存日期分别记录。
- [x] 采样长度、seed+epoch、补齐／截断、rank 间隔取样、合批和 worker fetch 分支已核对。
- [x] 普通 IterableDataset 与 map-style／IterDataPipe 的边界明确，不宣传统一自动分片。
- [x] 基础索引、请求成本与重放模型没有冒称论文算法，论文不适用的原因已说明。
- [x] 例子与公式可复算，两层 drop_last、N<R、空数据和 0 批的情况已覆盖。
- [x] 两道练习有完整答案、输入合同、验收与实际运行；20 项测试普通／优化模式通过。
- [x] 请求成本是假设模型；真实 CPU 功能实验与吞吐、GPU、分布式、崩溃恢复分开。
- [x] 原样源码、许可、独立审查报告与日志已经归档。

### 实际检查与处理

| 严重程度 | 观察与证据 | 处理 | 复验 |
| --- | --- | --- | --- |
| LOW（note） | 首次 Ruff 报 22 项字典字面量／import 风格问题 | 仅在本篇 5 个 Python 文件修正 C408、I001，并格式化 | [首次检查](evidence/runs/initial-ruff-check.txt)、[最终检查](evidence/runs/ruff-check.txt) 与普通／优化测试通过 |

其他实查没有发现未解决的 CRITICAL／HIGH：各 rank 的位置互斥不代表补齐 ID 无重复；分配 shard 不保证等步数；派发 6 批不能当成提交 6 批；计划摘要校验不能替代不可变数据版本；只恢复输入后缀不等于恢复完整训练。固定 main 的 11 个片段、482 行及许可证与 Git 原字节一致，四个相关完整文件与稳定 tag 一致。[technical-audit.json](evidence/technical-audit.json)

## 第二轮：教学逻辑与表达

- [x] 从 10 个 ID、3 个 rank 开始，先看到重复／丢失，再引入长度公式与术语。
- [x] 逻辑采样与物理 shard 分开解释，数字、公式和图服务于同一条输入进度链。
- [x] 图源与正文 Mermaid 完全一致；箭头表示所有权与阶段，没有冒充时间测量。
- [x] 优化表包含条件、收益、代价及应检查的量；无有限 shuffle 等同充分随机的结论。
- [x] 练习题先给输入、输出、边界和验收，完整答案折叠；starter 与完整答案职责清楚。
- [x] 面试追问覆盖基础、计算、实现、随机性、恢复、取舍和排障。
- [x] 预归档与最终本地链接、占位符、文件及索引检查均通过，见 delivery-check.json。

### 实际观察与修正

| 严重程度 | 观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | 第二题最初 starter 只有前缀切片 helper，不能充分引导计划身份和非法 token 的验收 | 扩展为 ResumeToken／DataPlan 完整 API 骨架，题目写清摘要、配置与边界；保留有意的学习空白 | 独立审查重新检查这一份 starter，Ruff 复验；其他 4 个 Python 文件摘要未变，见 [verification.json](evidence/runs/verification.json) |
| LOW（note） | 讲 map-style 派发时有 Sampler 拼写错误 | 修正术语拼写 | 正文复读核对完成 |

另外检查了：表中“样本次数”没有偷换成唯一 ID 数；取出 batch 没有被说成实际训练完成；实际探针在边界后消费完原迭代器以获得对照和正常退出，明确未注入崩溃；数据 manifest 的版本字符串由应用负责绑定内容；所有源代码读段前均有要回答的问题。图源只做语义和文本一致检查，没有渲染验收。

## 独立 Python 代码复核

所需专项代理 `/root/nov25_python_review` 在执行前因账户不支持其 `gpt-5.4` 模型而失败。随后由 `/root/nov25_python_fallback` 使用独立通用代码／正确性／安全审查角色复核 5 个冻结 Python 文件，未写仓库或安装依赖，不冒称专项身份。[结构化记录](evidence/python-review.json)

该审查报告通过：11,088 组独立官方 DistributedSampler 对照；合批尾部、两层 drop_last、shard 归属与覆盖；实际批次参与摘要、相同批次但配置不同的拒绝；所有有效游标和越界 token；随机种子边界；无状态值不消耗默认 RNG；starter API；独立临时目录下的真实 CLI／spawn worker 复现。无可报告 CRITICAL／HIGH／MEDIUM／LOW 缺陷。

此处归档的是审查代理实际返回的报告，不伪造成原始 stdout，也不把该代码复核扩展为正文或上游的独立审查。除主执行者通知的 starter 骨架变化外，前后摘要不变；最终摘要与执行清单匹配。

## 实际运行

使用现有 Python 3.12.16、PyTorch 2.15.0、Darwin arm64、Ruff 0.17.3；CUDA 不可用，没有初始化分布式进程组。依赖未安装或升级。命令中的 python／ruff 使用本机 ai-infra 环境，清单以 basename 展示，避免发布个人绝对路径。

| 对象 | 命令 | 实际结果 | 原始记录 |
| --- | --- | --- | --- |
| 基础答案 | `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v` | 退出 0，20 项通过 | [普通模式](evidence/runs/unit-tests.txt) |
| 优化模式 | `PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v` | 退出 0，20 项通过 | [优化模式](evidence/runs/optimized-tests.txt) |
| 代码检查 | `ruff check code`；`ruff format --check code` | 均退出 0 | [检查](evidence/runs/ruff-check.txt)、[格式](evidence/runs/ruff-format.txt) |
| 两题与真实 worker | `PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir evidence/runs` | 退出 0；派发 4→6、提交 2、输入后缀完全一致；iterable 尾批符合预期 | [stdout](evidence/runs/answers-run.txt)、[结果](evidence/runs/results.json) |

[verification.json](evidence/runs/verification.json) 保留时间、版本、命令、退出码、最终源码摘要及 starter 单独修订。初始未实现答案的失败见 [摘要](evidence/runs/red-tests.txt) 和 [无损压缩完整日志](evidence/runs/red-tests.txt.gz)，不当作最终通过。

## 发布结论

技术证据、教学表达与独立通用代码复核通过，无未解决 CRITICAL／HIGH 问题；最终链接、文件和发布索引检查通过，见 [delivery-check.json](evidence/delivery-check.json)。没有 GPU、多 rank 训练、吞吐、真实存储、worker RNG／流状态或进程崩溃验收，静态 shard 公式限各 rank worker 数相同。

[Git 预检](evidence/git-preflight.json) 记录发布前基线，不把未来提交 SHA 写成已经推送。文章归档、创建提交和 GitHub 同步属于不同状态，GitHub 同步以实际推送后的查询为准。

## 勘误与更新

暂无。
