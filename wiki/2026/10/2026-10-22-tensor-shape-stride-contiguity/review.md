# 审查与验证记录

主题：Tensor 的 shape、stride 与连续性。日期：2026-10-22，Asia/Shanghai。

技术审查：主代理对源码、API、公式、图解与实验作逐项自审；另有独立代码审查代理检查本篇 5 个 Python 文件。教学审查：主代理自审。独立代码审查不等于独立审查了整篇文章。

## 1. 技术证据审查

- 已核对官方仓库默认分支与分支 SHA，实际获取 6 个源码／许可证文件，并保存 2 个运行版本文件；不是整仓 clone，也没有编译当前主分支。
- 核对 `transpose` 的 size／stride 交换、view 的 `computeStride` 检查、reshape 的 alias／clone 分支和 contiguous 的 no-op／clone 分支。
- 复算地址 `a[1,2] == t[2,1]`、step／crop 两种布局展平差异、拆头／合头 stride 以及 32 MiB 载荷估算。
- 检查空 Tensor、标量、单例轴、非零 offset 和 channels_last，避免用普通非退化公式解释全部布局。
- 把最新源码静态分析、旧运行版本语义验证、理论字节数与未执行的 GPU 性能实验分开表述。
- 未引入需要论文核验的独立算法，Attention 示例仅分析布局变换的坐标与存储关系。

## 2. 独立 Python 审查与修正

审查范围：`code/layout_answers.py`、`torch_answers.py`、`run_lab.py`、`test_layout.py`、`test_torch.py`。首选固定 Python reviewer 因账号不支持其模型而未能执行，随后由一个独立代码审查代理承担同一范围；没有把失败的调用算作已完成审查。

| 级别 | 实际发现 | 修正与复验 |
| --- | --- | --- |
| MEDIUM | `itertools.product` 可能为一个逻辑为空的 shape 物化其他轴的索引池 | 参数验证后遇零维度立即返回空列表；有界回归在修复前实测峰值 4,791,488 字节而失败，修复后通过，见 [修复前记录](evidence/runs/empty-pack-before-fix.txt)。 |
| MEDIUM | 最初的内存回归测试读取全局 tracemalloc 历史峰值，还会关闭调用者的追踪 | 改为独立 `-I` Python 子进程，只测目标函数；正常模式和父解释器预先启用 tracemalloc 的模式分别验证。 |
| LOW | “所有 strided 输入”容易涵盖命名或量化 Tensor | 拆头／合头明确拒绝命名与量化输入，正文同步缩小契约；自动验收包含量化拆头。 |
| LOW | 相同 storage 起始指针可能被误解为元素集合重叠 | 明确它只是本实验已知来源的分配复用诊断，补充不重叠切片和外部缓冲区包装的边界。 |
| LOW | 最终目录检查发现 Ruff 会根据 cwd 推断本篇模块的导入分类 | 增加本篇 `code/ruff.toml` 明确 first-party 模块，从最终目录重新检查并通过。 |

独立审查者报告：未发现 CRITICAL／HIGH；普通 CPU 布局的值与坐标检查通过。审查者最后基于所提供的代码片段静态确认了子进程隔离修正，没有声称运行最终修订；最终执行验证由主代理完成，不把审查者早先运行的旧修订记录充作最终修订结果。

## 3. 教学与排版审查

- 开篇用“shape 相同、坐标值相同、stride 不同”的可运行例子提出主问题。
- 用 storage、shape、stride、offset、dtype 建立最小词汇，再推导地址公式。
- 两张图分别解释共享视图和真实物化，保留 SVG 源码和 PNG。检查每个格子的值、索引、橙色标记与正文相符；最终 PNG 已逐张目视核对，无裁切。第一次 Quick Look 预览出现裁切，已弃用，重新用现有渲染运行时导出。
- 第 4 节给出 step 与 crop 的成对案例，不把“非连续”简化成“不能 view”。
- 源码部分逐步追踪元信息、转置、连续性、区段匹配和复制分支；注明特殊 backend 与符号路径的范围。
- 三道练习从地址、物化到真实拆头／合头，题目与折叠详解分开，完整答案和验收文件均归档。
- 面试追问包含基础、实现、方案选择、正确性和排障，每项提供可核对的回答要点。

## 4. 最终验证记录

最终环境和实际命令、时间、退出码统一以 [summary.json](evidence/runs/summary.json) 为准。

| 验证 | 结果文件 | 验收标准 |
| --- | --- | --- |
| 全部 Python／PyTorch CPU 验收 | [tests.txt](evidence/runs/tests.txt) | 26 项测试全部通过 |
| 父解释器预启用 tracemalloc 的基础验收 | [tracemalloc-tests.txt](evidence/runs/tracemalloc-tests.txt) | 12 项基础测试通过，不污染父进程测量 |
| Ruff E、F、I、UP | [lint.txt](evidence/runs/lint.txt) | 无错误 |
| Ruff 格式检查 | [format.txt](evidence/runs/format.txt) | 5 个 Python 文件格式通过 |
| 真实布局诊断 | [lab.json](evidence/runs/lab.json) | 输出的 shape／stride／offset／值与文章一致 |
| 本地链接、图源和源码引用行号 | [archive-check.txt](evidence/runs/archive-check.txt) | 本篇文件可访问，源码符号与行号在快照内 |

不包含：GPU／MPS 运行、GPU timing、真实 Attention kernel、autograd、动态 shape／编译器路径、特殊 Tensor subclass、主分支构建。额外边界压力尝试被中断，未计入任何通过结果。

## 5. 执行偏差记录

独立审查阶段曾发生一次未充分确认归属的进程终止：执行者使用 `kill 24423` 发送默认 SIGTERM，目标命令行为 Homebrew Python 3.14 的 `Python -`。当时记录不足以确认它是否属于本次任务，已向用户说明，不再做额外压力实验或终止未知进程。主代理此前的一次过大空输入尝试返回退出码 137，不能据此断言二者为同一进程。

本篇使用修复后的有界输入、独立子进程回归与新进程中的完整测试重新验证；不从上述中断尝试推出正确性或性能结论。

## 6. 发布结论

结论：通过，完整文章已归档并加入索引。最终验证全部通过，正文与资料链接有效。已发现的教学与代码问题均已处理；没有未解决的 CRITICAL／HIGH，核心断言、练习与图解有对应证据。当前主分支只完成静态代码核验，运行结果仅覆盖归档的本机 CPU 环境。

## 勘误

暂无已发布后的勘误。

## 2026-10-22：按初学者反馈调整叙述

本轮由主代理做技术自审与教学自审，未新增 Python 实现，也未把原来的独立代码审查扩大成对新正文的独立审查。

| 轮次 | 实际发现 | 修正与复验 |
| --- | --- | --- |
| 教学 | 开篇直接展示两次转置与一次复制，shape／stride／storage 尚未解释，初学者难以沿着代码推演。 | 改为先画 12 个数字，解释轴、索引、shape 与 storage；用两次实际移动推导 `(4,1)`，最后再回到原来的复杂例子。 |
| 教学 | 属性表与公式不能替代“一维有没有行列”“向右一列为何不一定移动一格”等基础解释。 | 增加一维与二维的对照、存储位置与数值的区别、`view(-1)` 与单元素元组的解释；转置用第一行的三个访问位置逐步推演。 |
| 教学 | API 容易被理解成逐级替代关系。 | 先区分轴交换、形状分组、存储整理三个问题，再解释 `view` 的共享约束、`reshape` 的复制许可与 `contiguous` 的实际重排。 |
| 技术 | 增加的解释仍须保持坐标与存储语义准确。 | 新增例子实际执行，核对 shape／stride、值、view 成功与失败；现有 26 项 CPU 测试重新通过；18 个正文相对链接有效。 |

新增验证：[运行与环境](evidence/runs/teaching-revision.json)、[26 项回归输出](evidence/runs/teaching-revision-tests.txt)。源码快照与练习实现保持原版本，未增加 GPU 验证范围。本轮无未解决的 CRITICAL／HIGH。
