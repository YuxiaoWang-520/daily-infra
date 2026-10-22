# GPU 并行映射审查

2026-10-22。两轮为主代理自审；另有同一独立代码代理只读静态审查 3 个 Python 文件，不包含独立运行或 GPU 验证。

## 第一轮：事实与实现

核对官方 CUDA 线程模型、块级 shared memory／同步和 CC 6.0+ 的 32 字节访存示例。逐项对照固定 Triton add_kernel、调用入口，确认逻辑元素数与 CUDA thread 数不能直接等同。

实际自审修正：MEDIUM，若直接称练习结果为“内存事务数”，会把静态区间覆盖推广成硬件行为；正文、函数说明与诊断统一称扇区覆盖，并列出缓存、指令拆分、重放等未模拟项。LOW，31 字节起始地址跨界的测试只用于区间几何，正文补充不把它当合法单条对齐 CUDA 标量加载。

独立审查检查向上取整、尾部、块内 warp 重置、字节半开区间与测试，未发现 CRITICAL／HIGH／MEDIUM／LOW 或需修正项。n=0 明确跳过真实 launch，不把数学模型输入范围当设备支持声明。

## 第二轮：教学表达

先画 10 个位置的分工表，再命名 thread／block／grid，最后引入 warp 和 SM。明确 4 线程块只是便于数数的教学尺寸。访存从 32 个地址如何覆盖扇区讲起，再关联第 01 篇 stride，最后解释 shared memory 和驻留资源。

Mermaid 的尾部与表格一致，CUDA 片段标为未编译。Triton 源码每一句都回答前文已提出的问题；独立题目限定输入、输出和模拟范围，答案指出易错点。没有从枚举顺序、线程总数得出实际调度或性能结论。

## 验证与发布

[9 项测试](evidence/runs/tests.txt)、[诊断断言](evidence/runs/lab.json)、[Ruff](evidence/runs/lint.txt) 和 [格式](evidence/runs/format.txt) 均通过；命令与实际固定解释器见 [summary.json](evidence/runs/summary.json)。源码摘要、固定行号和相对链接由 [归档检查](evidence/runs/archive-check.txt) 核验，[校验和](evidence/SHA256SUMS) 可重查。初始 RED 为答案模块实现前的真实失败日志。

通过；无未解决 CRITICAL／HIGH。没有 CUDA／Triton 执行、GPU 内存计数器、occupancy 或性能实测，不把它们列作通过项目。
