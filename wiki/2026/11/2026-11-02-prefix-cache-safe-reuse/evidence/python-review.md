# Python 独立审查记录

日期：2026-11-02（Asia/Shanghai）。范围仅为本篇五个原创 Python 文件：`models.py`、`exercises.py`、`answers.py`、`test_answers.py`、`run_lab.py`；不含上游源码、引擎、正文技术事实或教学表达。

## 实际执行方式

首先调用已有 `python-reviewer` 专职代理，当前账户返回 `400 model_not_found`（固定 gpt-5.4 不受支持），没有完成审查，不能计作通过。

随后由已有 `capacity_python_fallback`（gpt-5.6-terra、max effort）代理独立进行只读 Python 审查。工作包给出函数输入输出契约、pool 的状态规则和已知执行边界；主代理负责修复、正文与最后接受决定，未让审查者修改代码或 Git。

审查者实际运行 `/Users/wangyuxiao/conda3/bin/python3 -m unittest discover -s code -p test_answers.py -v` 和 `code/run_lab.py`，当时 28 tests 通过、lab 退出 0。审查者读取已有 Ruff 通过记录，没有重复执行 Ruff。

## 真实发现：一个 MEDIUM

| 程度 | 观察与复现 | 修复 | 验证 |
| --- | --- | --- | --- |
| MEDIUM | 当相同 key 同时有 warm 与 active 两个物理副本时，原实现总选最小 ID，可能错误耗掉可分配候选、拒绝本来能接纳的请求 | 优先选择引用数大于 0 的副本，同状态按 ID 稳定选择；使 active 副本可继续共享，warm 副本留给 miss 覆盖 | 先新增回归复现失败，再修复；完整 29 tests、lab、Ruff check／format 通过 |

公开接口复现：`PrefixPool(2)` 中，r1、r2 在前者完成前分别申请 `("a",)`，因此产生块 0 和块 1 两个副本；两者完成后释放 r1，块 0 为 a/ref=0/ready，块 1 为 a/ref=1/ready，free 为 `[0]`。

此时 r3 申请 `("a","b")`。旧代码选块 0 作为命中，预检扣掉它后报告无空闲块；实际上可共享活跃块 1，再用块 0 存 b。新增回归 `test_active_duplicate_is_preferred_to_preserve_free_capacity` 的修复前错误保存在 [active-copy-red.txt](runs/active-copy-red.txt)。修复后返回 `(1,0)`，a 在块 1 的引用数为 2，旧 warm 副本的索引被正确移除，其他 a 副本仍可用。

审查者报告无 CRITICAL／HIGH、无其他实现级发现。此 MEDIUM 已影响教学池的容量语义，因此在发布前修复，没有只加免责声明保留错误。

## 修复后的接受范围

主代理发送了修复与回归结果给原审查代理；没有声称它又完成第二轮全范围审查。最终归档目录的实际复验结果以 [summary.json](runs/summary.json)、[tests.txt](runs/tests.txt)、[lab.json](runs/lab.json) 为准。

选择活跃副本是本篇教学契约，没有推断上游 `get_one_block` 采用相同优先级。未验证真实 KV 内容、部分块／追加、异步 GPU 完成、并发、性能或认证隔离。

更早的 [initial-red-tests.txt](runs/initial-red-tests.txt) 是实现答案模块之前的导入失败记录，不作为算法正确性证据。两轮正文自审见 [review.md](../review.md)。当前没有未解决 CRITICAL／HIGH 或影响核心机制的 MEDIUM。
