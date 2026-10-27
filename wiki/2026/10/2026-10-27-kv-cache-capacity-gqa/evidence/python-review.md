# Python 独立审查记录

范围：本篇 `code/answers.py`、`code/test_answers.py`、`code/run_lab.py`。只审查本篇教学代码，不审查归档的上游源码，也不把静态阅读当作 vLLM 运行验证。

预设 `python-reviewer` 的模型在本账户不可用，任务返回模型不支持错误，没有产生审查结论。随后由单个独立 `gpt-5.6-terra`、`max` 审查代理执行等范围只读审查；主代理负责修正与最终验收。下面为实际报告的归纳，非预设审查器的输出。

## 首轮独立审查

- 无 CRITICAL／HIGH／MEDIUM 发现。
- LOW：`grouped_attention` 的位置张量没有显式设备，会受调用者的默认设备影响。CPU 输入在非 CPU 默认设备下可能与 mask 不匹配；测试与 lab 的工厂函数也应明确 CPU。
- 非阻断覆盖建议：原测试只验证 Q 中的 NaN，而实现检查 Q／K／V；应覆盖三者的非有限值。
- TP 辅助函数只检查 KV head 的分片／复制条件。Query head 的 TP 整除需由调用者检查；正文与函数契约已经说明，未将它称为完整部署验收器。
- 审查代理实际执行 `python test_answers.py`，当时 21 tests 通过；实际执行 `python run_lab.py`，退出码 0。首轮未重复执行 Ruff。

## 修正及窄范围复核

主代理先加入非 CPU 默认设备回归，真实复现一个错误，保存于 [修复前日志](runs/default-device-before-fix.txt)。随后将两个位置张量的设备指定为 `q.device`，并给测试、lab 的 Tensor 工厂指定 CPU；补测 Q／K／V 各自的 NaN、正 Inf、负 Inf。

独立代理复核后确认：LOW 已关闭，未发现新增分级问题。它实际执行 `/Users/wangyuxiao/conda3/bin/python3 test_answers.py`：22 tests，退出码 0；读取主代理的最终运行汇总，确认 lab、Ruff check、Ruff format 的日志均为退出码 0，未重复执行这三项。

最终主代理测试、环境及原始输出见 [summary.json](runs/summary.json) 与 [tests.txt](runs/tests.txt)。本记录不把一次不同目录或不同阶段的运行冒称为最终归档目录的全部检查。
