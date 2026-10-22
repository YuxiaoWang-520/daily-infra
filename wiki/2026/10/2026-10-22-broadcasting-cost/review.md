# 广播篇审查与验证

状态：通过。日期：2026-10-22，Asia/Shanghai。技术事实与教学表达两轮均由主代理自审；另由独立代码审查代理检查本篇 3 个 Python 文件，没有把代码审查说成独立审查了整篇文章。

## 第一轮：技术事实、实现与验证

逐项核对末尾 shape 对齐、0 与 1 的选择、`expand` 的零 stride、TensorIterator 的字节步长、输出分配和 `_sum_to` 的归约维度。对照的是实际保存的固定主分支文件；没有构建该主分支。CPU 运行单独标明 PyTorch 2.9.1。

复算 6 次加法、12／24 字节存储、60／72 字节两种假设、8 GiB 三维差值、64 MiB 输出和 2 MiB tile。模型与实测边界已明确：这里的字节推导不能当 HBM 流量或 RSS／GPU 显存峰值。

代码审查范围：`code/answers.py`、`test_answers.py`、`run_lab.py`。固定 Python reviewer 的模型在当前账号不可用，未执行成功；由独立 Python 代码审查代理替代，采用只读静态审查和定向复核。该代理没有运行测试，最终运行验证由主代理执行。

| 级别 | 实际观察 | 修正与复验 |
| --- | --- | --- |
| LOW | 原 docstring 对“每个临时量”的界限过宽；D=0 时仍有二维归约结果；前一块 diff 的 Python 引用可能跨入下一块分配。 | 限定为差值 tile 的逻辑尺寸，另说明归约 tile；每块写回后 `del diff`。不宣称分配器立即释放或已测峰值。 |
| LOW | 原 `no_grad` 只关闭反向模式，不能据此声称所有自动微分都关闭。 | 本函数本就限定推理用途，改用 `inference_mode`。新增有界 forward AD 回归：修复前实际失败，修复后通过，见 [修复前记录](evidence/runs/inference-contract-before-fix.txt)。 |

独立审查者已静态确认上述两项修正，未发现 CRITICAL／HIGH 或必须继续修正的问题。返回值属于推理 Tensor，不能据本题承诺其可直接接入需要保存此值的训练图。

## 第二轮：从基础到进阶的叙述

正文先给两条样本加 bias 的六个实际结果，然后手工右对齐 shape，再提出“没复制怎么有两行”并解释零 stride。依次区分逻辑位置、storage 字节、运算结果与三维中间量；没有一开始引入内核或硬件术语。

两幅 Mermaid 分别表示“两个逻辑行指向同一 storage”和“分块结果写入二维输出”，图源随正文保存。数字与正文一致。梯度先用同一个 bias 影响两行的直觉，再写求和式；源码各节明确对应前文的哪个疑问。

练习一检验广播规则，练习二解决中间量过大的实际问题，题目先写输入与验收，再折叠详解。完整可运行代码独立归档，未要求读者先阅读答案才能理解题目。面试追问覆盖基础、实现、容量、梯度与排障。

## 最终运行

真实命令、时间、环境和退出码见 [summary.json](evidence/runs/summary.json)。

| 项目 | 结果与记录 |
| --- | --- |
| 12 项 CPU unittest | [tests.txt](evidence/runs/tests.txt)，全部通过；含 float32／float64、非连续输入、尾块、零样本、零特征、推理契约和非法输入 |
| 固定小输入诊断 | [lab.json](evidence/runs/lab.json)，另以断言核对 stride、storage 字节、结果、梯度与平方距离 |
| Ruff E、F、I、UP（本篇忽略行长 E501） | [lint.txt](evidence/runs/lint.txt)，通过 |
| Ruff 格式 | [format.txt](evidence/runs/format.txt)，3 个文件通过 |
| 归档、链接与固定源码定位 | [archive-check.txt](evidence/runs/archive-check.txt) |

初始 RED 记录是测试先于答案时的真实缺失实现错误，保留为开发证据，不是最终失败。`inference-contract-before-fix.txt` 也是修复前结果；发布结论只使用修复后的完整记录。

未验证：CUDA／MPS、当前主分支构建、GPU kernel 路径、性能、真实内存峰值、动态 shape／编译器与完整 autograd 引擎。基础语义与两道答案已有真实 CPU 证据，不把这些未验证范围写成通过。

## 发布结论

通过，未解决的 CRITICAL／HIGH 为零。技术与教学两轮自审已完成，独立代码审查及定向复核已完成，必要运行与归档检查通过；允许归档、更新索引并单篇提交推送。
