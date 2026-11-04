# 本篇 Python 独立审查

日期：2026-11-04（Asia/Shanghai）。范围只包含本篇 6 个 Python 文件和 `code/README.md` 的契约，不包含上游原文、整套 PyTorch 或其他文章。

## 实际执行者

按语言审查规则先重新调用固定 `python-reviewer` 代理 `prefill_python_review`；调用实际返回 HTTP 400：`The 'gpt-5.4' model is not supported when using Codex with a ChatGPT account.`。该代理没有完成审查，不能列作通过者。

随后由独立代理 `capacity_python_fallback`（现有 gpt-5.6-terra／max 审查替代）执行只读审查及必要复验。主代理负责整合与最终接受；没有把自审称作独立审查，也没有人工专家审查。

## 首轮独立检查与结果

代理报告未发现 CRITICAL／HIGH／MEDIUM／LOW 问题。核对内容：

- `answers.py` 的状态项逐元素乘法、梯度与 moment 生命周期开关、额外 master 独立于 Adam 的延迟初始化。
- `replay_lifetimes` 的新 storage 身份、alias 链、最后引用删除、名字删除后重用、首次峰值、非法输入与不可变输出。
- `payload_probe.py` 的三组 CPU 配置、dtype、状态键、step 标量、四个采样点、别名探针和未用参数。
- 契约明确区分 CPU payload、RSS、CUDA allocator 和 GPU 峰值。

实际执行并通过：

```text
/Users/wangyuxiao/conda3/bin/python3 -m unittest discover -s code -p test_answers.py -v
/Users/wangyuxiao/conda3/bin/python3 code/run_lab.py
/Users/wangyuxiao/conda3/bin/python3 code/payload_probe.py
/Users/wangyuxiao/conda3/bin/python3 -m ruff check --select E,F,I,UP --ignore E501 code
/Users/wangyuxiao/conda3/bin/python3 -m ruff format --check code
```

结果为 21 tests、lab、CPU probe、Ruff check／format 通过。额外只读检查了名字重用后的旧别名释放，以及非法 alias／drop／空白名字抛 ValueError。

## 新增标量算法断言的增量复核

首轮后，主代理仅在 `unused_parameter_probe()` 中把两个标量改为 FP64，增加论文算例的 moment 与参数更新断言及 JSON 字段，代码说明同步补充一句。

同一独立代理只复核该新增段并重跑 probe、单独调用该函数，没有重复全范围审查。结果：退出 0；梯度 2，`m=0.19999999999999996`、`v=0.0040000000000000036`，实际与期望权重均为 `0.999000000005`。未发现新增问题。

## 证据边界

真实运行记录由主代理另外保存在 [summary.json](runs/summary.json) 与对应原始输出。初始 red 记录是在答案文件创建前因无法导入 `answers` 而失败，属于测试先写的失败记录，不是算术缺陷的复现。

不覆盖 RSS、CUDA allocator、GPU peak、GPU 性能、收敛、当前 main 可执行二进制或真实异构／分片布局。probe 在独立进程中设置默认 dtype 和线程数，退出后不改变外部环境；没有安装依赖或进行 Git 操作。
