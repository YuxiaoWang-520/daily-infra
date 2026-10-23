# 融合篇审查

2026-10-23。两轮由主代理自审；单一独立代理静态审查本篇 3 个 Python 文件，没有执行或 GPU 验证。

## 第一轮：实现与模型

逐步数清两份中间值的写入／重读，复算 156／60 字节和两种假设成本。读取实际固定 Inductor scheduler 的 can_fuse、placement／归约契约、共同访问评分、speedup 配置和 benchmark 分支，并读 Triton softmax kernel 与资源元数据处理。

自审修正：MEDIUM，scheduler 注释的 shared memory 容易被翻译为 CUDA 共享内存，正文改为共同 buffer／依赖访问；MEDIUM，配置关闭或提前路径存在，不能把所有融合说成自动实测择优；MEDIUM，CPU 分块并不等于 GPU 融合，题目、代码、诊断与结论都明确分开。LOW，合成成本相加不是 Roofline 下界，且只惩罚融合内存项，公式旁已限定。

独立代码审查未发现 CRITICAL／HIGH／MEDIUM／LOW 或需修正项，检查了原地更新对象、每块释放、空形状、单位与模型选择反转；没有声称实际测 GPU。

## 第二轮：教学

先算一行三个中间结果，再画三步和一步的数据流，逐份解释少掉的四份元素流量；启动成本、状态存活与归约分工都在具体账本之后引入。源码先问可行性，再讲评分和适用性能比较，避免把“自动融合”说成一个无条件黑箱。

两题的目标、约束、公式、完整答案与输出齐全；没有用虚构计时制造真实性。CPU 题保留完整输出的成本，没有把小 buffer 误说成整个程序内存上界。

## 验证与发布

[9 项测试](evidence/runs/tests.txt)、[诊断](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt)、[格式](evidence/runs/format.txt) 通过，[环境与命令](evidence/runs/summary.json)、[链接／固定行号／源摘要](evidence/runs/archive-check.txt) 与 [校验和](evidence/SHA256SUMS) 归档。初始 RED 是实现前的真实失败记录。

通过，无未解决 CRITICAL／HIGH。未运行 torch.compile、GPU 融合、trace 或硬件资源测量，未提供真实加速比。
