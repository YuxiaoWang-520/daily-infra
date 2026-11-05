# Python 代码审查

日期：2026-11-05（Asia/Shanghai）。范围：本篇 `code/` 的 7 个 Python 文件与 README，共 8 个文件；不含上游归档或旧文章。

审查方式：独立代理审查，执行者为 `checkpoint_python_fallback`（默认代理，MAIN 层级），主执行者负责修复、集成和最终验收。最初调用指定 `python-reviewer` 角色失败，工具返回 “The 'gpt-5.4' model is not supported when using Codex with a ChatGPT account.”，没有把这次失败记作已完成的审查。

## 独立审查的实际发现

| 严重程度 | 观察与复现 | 修复 | 复核 |
| --- | --- | --- | --- |
| MEDIUM（info） | 原 `segments * overhead_us` 在 `layers=10**309,s=1` 时过早把整数转浮点；即使 `h=0` 也抛裸 `OverflowError`。非零真正溢出时也未满足 `ValueError` 契约。 | 先加回归测试复现，再把有限浮点开销转为整数比值，先用整数乘段数再相除；捕获真正结果溢出并转成 `ValueError`。 | 主执行者 19 项容量测试通过；独立代理只复核改动及新增 3 例，3 项全通过，明确关闭 MEDIUM。 |

新增三例分别覆盖零开销仍有限、巨大段数乘极小开销仍有限、真实总开销溢出。复现输出为 [reviewer-red-test.txt](runs/reviewer-red-test.txt)。

## 范围与结论

独立审查还核对了输入验证、容量选择排序、CPU RNG 恢复、Dropout 对照、梯度与参数的独立副本，以及 storage 去重统计的范围；原始阶段按需运行 8 项既有定向测试，修复后运行 3 项新增测试，没有声称重复全套或运行 Ruff。

初轮无 CRITICAL／HIGH，只有上述 MEDIUM；修复复核后，限定范围内无遗留 CRITICAL、HIGH、MEDIUM 或 LOW 问题。主执行者最终运行记录见 [summary.json](runs/summary.json)。

未验证 CUDA、编译、训练吞吐、真实峰值内存或长期收敛。独立代码审查没有代替正文技术与教学两轮自审。
