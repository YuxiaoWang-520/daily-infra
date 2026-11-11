# 独立 Python 代码审查

日期：2026-11-11（Asia/Shanghai）。实际审查者：独立默认代理 checkpoint_python_fallback，范围为本篇 7 个 Python 文件和 code/README.md。

先尝试专用 python-reviewer，底层 gpt-5.4 再次被当前 ChatGPT 账号拒绝，代理未读取代码即退出。实际审查由可用独立代理完成，没有把专用模型调用失败记成完成审查。

## 核验范围

静态检查环的同步消息、索引、块所有权、精确 SUM 与 TX 字节；检查成本资源记账、输入校验和单位；检查 Gloo 四进程及临时目录收尾、实验输出、模板验收入口。独立审查者未重复完整四 rank 实验，Gloo 管理由静态核验覆盖。

实际执行边界反例和所有权、消息轮次、flat 成本、共享 NIC 的 4 项定向测试，后者全通过。

## 实际发现与修正

MEDIUM（info）：costs.py 过早转换成 float。容量为 1e308 时，payload 10**309 的两 rank 环应得到约 10 秒，原实现却抛 OverflowError；整块树应约 20 秒，也提前溢出。单 rank、容量 1e-320 的树模型返回 NaN。alpha=10**400 的有限性检查还会抛裸 OverflowError。

修正：负载以 int 累计，用 Fraction 完成比值与总式后才转 float；容量与 alpha 明确限制为有限 float 可表示范围，并把转换溢出变为 ValueError；最终时间无法表示时明确报错；单 rank 提前返回零。范围说明同步更新。

回归：新增 5 项，主执行者在修正前得到 1 failure 与 4 errors，见 [numeric-red-tests.txt](runs/numeric-red-tests.txt)；修后完整 [28 项测试](runs/tests.txt) 全通过。独立审查者只针对修正运行 5 项定向测试，全部通过，原反例分别得到约 10 秒、20 秒、0，以及预期 ValueError。独立小范围 stdout 未另存文件，结论以代理实际返回为依据；主执行者完整输出已归档。

## 最终结论

原 MEDIUM 已关闭，限定修正范围未发现新风险。未解决的 CRITICAL／HIGH：0；影响核心理解的 MEDIUM：0。题设仍为等分块整数模型、理想容量模型和固定同机 CPU 正确性实验；不扩张为任意 NCCL 路径或 GPU 性能验收。
