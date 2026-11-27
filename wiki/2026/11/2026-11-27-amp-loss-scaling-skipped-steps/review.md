# 审查记录

主题：loss scaling、溢出检测、梯度变换顺序与训练进度。

文章状态：published。技术证据与教学表达两轮均由主执行者自审，Python 代码另有独立通用审查及定向修复复核。没有把自审称为独立审查。时间、检查项和结论见 [self-review.json](evidence/self-review.json)。

## 第一轮：技术事实与证据

- [x] 主问题连接 FP16 范围、梯度变换、跳步与恢复计数，有论文、固定源码及真实 CPU 对照。
- [x] 当天实际 fetch 官方默认分支，保存完整 SHA，稳定 tag、页面缓存与运行包 build 分别记录。
- [x] 核对 scale、unscale、普通 step、缓存标记、update、state_dict 及 CPU 内核路径。
- [x] 原论文 §3.1、§3.2、§3.3 与当前实现对应，框架增长策略未写成论文要求。
- [x] 小值、溢出、裁剪、累计和五步动量表可复算，与实际 JSON 相符。
- [x] 两题完整答案已运行；最终 23 项测试普通／优化模式通过，边界与错误对照被实际覆盖。
- [x] 没有 GPU、吞吐、显存收益或收敛实测，CPU 功能和数学模型的范围明确。
- [x] 论文笔记、资料、源码片段、许可、图源和运行输出均归档。

### 发现、修正与复验

| 严重程度 | 实际观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| LOW（note） | 独立代码审查发现两个有限的 `1e308` 在 math.fsum 求和时先抛 OverflowError，绕过后续的 ValueError 校验；主执行者同时检查到超大整数转 float 的同类边界 | 两条路径显式将 OverflowError 转为带原因的 ValueError，保持输入错误合同 | 新测试先真实失败，见 [overflow-red.txt](evidence/runs/overflow-red.txt)；最终 23 项测试和独立 delta 复核通过 |
| LOW（note） | Ruff 从根目录与从 code 目录执行时，对本地模块的导入分类不同，初次归档检查出现 4 项 I001 | 增加本篇局部 .ruff.toml，固定两个本地模块分类，仅整理本篇 imports | [首次记录](evidence/runs/initial-ruff.txt)、[最终检查](evidence/runs/ruff.txt) |
| LOW（note） | 初轮手算测试用十进制小数精确相等比较，实际得到 `0.07500000000000001` | 改用浮点近似断言；不改正确裁剪公式或验收目标 | 最终数学与真实裁剪测试均通过，源码原生 epsilon 与手算简化分别说明 |

其余重点实查：普通 step 在 READY 才做反缩放；检查后的篡改反例不被包装成正常配方；多个优化器的局部跳步不由共享 scale 下降反推；CPU BF16 类型实验与 FP32 协议实验分开；state_dict 内存重载没有被说成磁盘持久恢复；SGD 返回 None 不是通用跳步信号；调度器计数是应用策略。

13 个片段共 336 行和 LICENSE 均与固定 Git 对象原字节一致。CPU 内核在 main／稳定 tag 全文件相同，GradScaler Python 全文件不同，构造器默认设备变化已核对。最终 5 个 Python 摘要与运行清单、独立审查一致。[technical-audit.json](evidence/technical-audit.json)

## 第二轮：教学逻辑与表达

- [x] 从“读了 5 个 batch，只更新 4 次”开始，再用 `1e-8` 和 `[3,4]` 建立数值直觉。
- [x] 首次出现的关键术语就地解释，前置篇目有效链接。
- [x] 先讲缩放与反缩放，再讲裁剪、累计、状态机、恢复及高级边界。
- [x] 图中固定 scale、检查时刻、普通优化器决策和调度策略与正文一致，保留可编辑图源。
- [x] 两题有输入、输出、限制、验收、完整实现、推导与真实运行结果，起始合同可独立阅读。
- [x] 工程取舍说明收益来源、代价、长期跳步排查及分布式／硬件验证边界。
- [x] 追问覆盖基础、实现、选择、恢复和排障，均有回答要点。
- [x] 本地链接、无占位符、发布文件和索引的最终检查见 delivery-check.json。

### 实际发现与澄清

| 严重程度 | 观察 | 处理与复核 |
| --- | --- | --- |
| MEDIUM（info，已修正） | 小梯度存入 FP16 的例子，可能让初学者误以为常规 AMP 最终参数 .grad 必须为 FP16 | 明确这是反向中间梯度的保存点模型；随后展示参数／梯度 FP32、linear BF16、loss FP32 的真实类型结果，并区分手工 half 参数的错误对照 |
| LOW（note） | AMP、Inf、NaN、RNG 若不展开会让入门解释依赖术语 | 在首次关键出现处补充自动混合精度、无穷大、非数及随机数生成器含义 |

教学自审还核对了 FP16 次正规数的舍入阈值、scale 不增加尾数精度、已经为零不能事后补救，以及损失分母在 scaling 前定义。没有把 BF16 说成无数值风险，也没有把五次数据处理等同于五次参数更新。图只核对源码与流程语义，没有渲染或 GPU trace 验收。

## 独立 Python 代码复核

专项代理 `/root/nov27_python_review` 因账户不支持其 `gpt-5.4` 模型，在执行前失败。随后 `/root/nov27_python_fallback` 使用独立通用 Python／正确性／安全审查角色只读复核 5 个 Python 文件和局部 Ruff 配置，未修改仓库、安装依赖或操作 Git，没有冒称专项身份。[真实代理报告摘要](evidence/python-review.json)

其独立标量 oracle 验证动量递推、scale、accepted updates、调度器计数、恢复等价和多个优化器独立跳步；另验证 meta 默认设备下仍使用 CPU、BF16 类型，以及临时 CLI 的严格 JSON 与重复路径拒绝覆盖。没有重复完整单测或 Ruff。首次发现的唯一 LOW 已修复，原审查者仅定向复核两个溢出异常及正常累计案例，接受 delta，无新增发现。

审查摘要保存首轮与最终文件 SHA256。它是代理实际报告的归纳，不是伪造的原始 stdout；范围局限本篇代码，不是正文或上游的独立审查。

## 真实运行

环境：Python 3.12.16、PyTorch 2.15.0（build `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`）、Darwin arm64、Ruff 0.17.3；CUDA 不可用。没有安装或升级依赖，没有初始化分布式进程组。

| 对象 | 实际命令（cwd 为 code） | 结果 | 原始记录 |
| --- | --- | --- | --- |
| 参考答案 | `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p test_answers.py -v` | 退出 0，23 项通过 | [普通模式](evidence/runs/unittest.txt) |
| 优化模式 | `PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s . -p test_answers.py -v` | 退出 0，23 项通过 | [优化模式](evidence/runs/unittest-optimized.txt) |
| 静态检查／格式 | `ruff check .`；`ruff format --check .` | 均退出 0 | [Ruff](evidence/runs/ruff.txt)、[格式](evidence/runs/ruff-format.txt) |
| 完整例子 | `PYTHONDONTWRITEBYTECODE=1 python run_answers.py --output-dir ../evidence/runs/cpu` | 退出 0，严格 JSON 与结果文件一致 | [stdout](evidence/runs/cli.txt)、[结果](evidence/runs/cpu/results.json) |

实际可执行文件使用已有 ai-infra 环境。[最终运行清单](evidence/runs/manifest.json) 保存时间、命令、退出码和代码摘要，个人绝对路径被替换。初始 stub 的两项失败保留在 [red-tests.txt](evidence/runs/red-tests.txt)；初版 22 项通过及 Ruff 分类失败保留在 [initial-manifest.json](evidence/runs/initial-manifest.json)，不当作最终 23 项结果。

## 发布结论

两轮自审、独立代码复核及修复复核、必要测试均通过，无未解决 CRITICAL／HIGH 或影响核心理解的 MEDIUM。最终交付完整性与本地链接见 [delivery-check.json](evidence/delivery-check.json)。

未验证 GPU FP16 反向、fused optimizer、分布式跳步协同、磁盘崩溃恢复、性能或收敛。提交前的远端与身份基线见 [git-preflight.json](evidence/git-preflight.json)；GitHub 同步完成以实际推送后的查询为准，不预先写入未来提交 SHA。

## 勘误与更新

暂无。
