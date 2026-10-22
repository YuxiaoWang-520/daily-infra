# Roofline 篇审查

2026-10-22。主代理完成技术事实与教学表达两轮自审；独立代码代理静态检查本篇 answers、tests、lab 三个 Python 文件，没有独立执行。原固定 Python specialist 的模型不可用，本批沿用单一 Terra max 代理作有界替代，不把失败调用记作成功审查。

## 第一轮：技术与证据

逐项复算 4 元素向量、非方阵 GEMM、1024 方阵、128 方阵、两种十进制速率及时间下界。核对原报告第 3—4 节、源码 naive_softmax 与 softmax_kernel。原报告 DRAM 层级、可持续带宽和额外 ceilings 均保留，公式账本没有写成 profiler 实测。

实际发现并修正：

- MEDIUM，自审：上游注释把元素数写成 bytes。正文按主输入输出 `2MN × itemsize` 推导，并指出来源中的单位问题；未承诺四倍加速。
- LOW，独立静态审查：空任务 intensity 的 0.0 是占位约定，balanced 是浮点严格相等。已在正文和代码说明中补充，未为了边界标签引入额外容差语义。
- LOW，运行环境检查：在草稿子目录中裸 `python3` 解析到另一个系统解释器，没有 Ruff；正式核验改为已有固定 `/Users/wangyuxiao/conda3/bin/python3`，未安装依赖，最终报告记录实际 executable。基础标准库测试在两个解释器下均通过，正式结论取固定环境日志。

独立审查未发现 CRITICAL／HIGH／MEDIUM 或必须修改项。未更改测试标准以迁就实现。

## 第二轮：教学表达

从 4 个数的加法开始，先分清“操作总数”和“每秒速率”，随后引入字节、强度和两个时间下界；矩阵乘法先解释每个输出沿 K 累加，再给公式。Roofline 示意图不伪装测量，图中的两条限制与公式一一对应。

把缓存层级与实际诊断放在基础例子之后；源码先提出“到底少搬了哪一份中间量”，再跟踪五步与融合路径。练习题和答案分开，解释假设与常见计数错误。无未解释术语堆砌，无具体 GPU 速度承诺。

## 实际验证

- [12 项测试](evidence/runs/tests.txt) 全部通过，覆盖非方阵、单位、空任务、只搬运和非法输入。
- [诊断输出](evidence/runs/lab.json) 的合成时间吞吐断言通过。
- [Ruff 检查](evidence/runs/lint.txt)、[格式检查](evidence/runs/format.txt) 通过。
- [环境与命令](evidence/runs/summary.json)、[归档检查](evidence/runs/archive-check.txt)、[校验和](evidence/SHA256SUMS) 保存。

没有运行 Triton／CUDA，没有验证实际 DRAM 流量、GPU latency、GPU 峰值或图中测量点。初始 RED 日志是实现前缺少答案模块的历史记录；发布结论采用最终完整测试。

发布结论：通过，无未解决的 CRITICAL／HIGH，必要验证和两轮审查完成，可单篇归档并推送。
