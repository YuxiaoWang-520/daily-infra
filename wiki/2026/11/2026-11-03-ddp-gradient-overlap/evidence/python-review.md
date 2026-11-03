# 独立 Python 审查记录

日期：2026-11-03（Asia/Shanghai）。范围仅本篇六个原创 Python 文件和 `code/README.md`；不包含文章事实、上游源码或 GPU 实现。

## 真实角色与执行

首先调用既有 `python-reviewer` 专职代理，实际返回 `400 model_not_found`，当前账户不支持它固定使用的 gpt-5.4；本次没有完成任何有效审查。

之后按成本路由复用 `capacity_python_fallback`（gpt-5.6-terra，max effort），给出紧凑契约、范围和已有运行证据，独立只读审查。它没有修改代码、Git 或环境；主代理负责整合、修复与最终接受。

审查者实际运行：

```bash
/Users/wangyuxiao/conda3/bin/python3 -m unittest discover -s code -p test_answers.py -v
/Users/wangyuxiao/conda3/bin/python3 code/run_lab.py
GLOO_SOCKET_IFNAME=lo0 /Users/wangyuxiao/conda3/bin/python3 code/ddp_probe.py
```

结果：当时 23 tests 通过，lab 退出 0，双进程 CPU Gloo probe 退出 0，两 rank 梯度及更新后参数一致。审查者读取已有 Ruff 通过记录，未重复执行 Ruff。

## 发现与修复

仅发现 1 个 LOW，无 CRITICAL／HIGH／MEDIUM。

`answers.py` 原先先计算 `size/rate` 再交给有限值检查；合法的大正整数 `10**400` 在除法阶段会抛 `OverflowError`，与题目约定的“不可表示派生时间抛 ValueError”不符。

主代理先加入 `test_derived_time_overflow_uses_value_error`，实际复现失败，见 [overflow-red.txt](runs/overflow-red.txt)。之后为通信耗时和串行时间的运算捕获 `OverflowError` 并转换 `ValueError`。该回归还检查极小有效传输率导致的不可表示结果。

修复后完整 **24 tests**、lab、Ruff check／format 通过。已将修复和回归结果发送给原审查代理；没有声称它又做了一轮完整审查。最终目录复验由主代理接受，见 [运行摘要](runs/summary.json)。

## 接受与限制

加权全局梯度、DDP loss 缩放、桶覆盖与固定顺序依赖、慢 rank、不可变结果、输入保持、默认 CPU DDP 与 `no_sync` 的数值断言均在审查范围内。

没有验证 GPU／NCCL、真实通信重叠、性能、通信次数、训练收敛或本文 main 源码的可执行二进制。独立审查者的成功不是上述范围的验证。更早的 [initial-red-tests.txt](runs/initial-red-tests.txt) 是答案模块不存在时的导入失败，不作为数学正确性证据。
