# 第 11 篇审查记录

日期：2026-10-26（Asia/Shanghai）。技术与教学两轮由主代理自审；另有一个独立代码代理只读静态审查本篇三个 Python 文件和 `code/README.md`，未独立运行测试。未进行 GPU 验证。

## 第一轮：技术事实、源码与代码

核对原始 Transformer 的因果约束、位置与逐位置部件，以及 PagedAttention 论文的 Prompt／生成过程。实际 clone 当日 vLLM main，固定 `7128f568ac036f7b7f21860d1928747ac6f78213`；读取 Llama QKV／位置处理、Attention 包装的按层缓存及更新依赖、所选后端缓存写入／读取、调度进度和请求释放的调用路径。7 份归档源码逐份与固定 commit 的 Git blob 比较。

实际发现与处理：

| 级别与来源 | 观察 | 处理与证据 |
| --- | --- | --- |
| MEDIUM，自审 | 初始非连续输入测试先切片再除法，除法物化成了连续 Tensor，未真正覆盖目标布局。 | 将除法移到切片之前；真实失败保存在 [noncontiguous-fixture-before-fix.txt](evidence/runs/noncontiguous-fixture-before-fix.txt)，修正后完整 15 项通过。这是 fixture 修正，不把它写成实现缺陷。 |
| MEDIUM，自审 | 溢出测试使用默认 FP32 创建 1e308，会在构造时成为 Inf，从而没有到达“有限输入、矩阵乘法溢出”的分支。 | 明确创建 FP64 Q/K，并要求报错包含 `scores overflowed`。主代理复验完整测试通过；属于独立审查后的测试补强，未宣称独立代理复验。 |
| MEDIUM，自审 | “Prefill 不生成首 token”或“输出几个 token 就一定有几个缓存位置”会造成 off-by-one。 | 按实际四次调用列出输入、采样和缓存位置；缓存与完整重算的 `tokens[:-1]` 逐层对照。 |
| MEDIUM，自审 | 若把教学的两个阶段直接写成当前调度器的两个循环，会误解实现。 | 引用 `schedule` 的统一进度说明，区分工作负载概念与调度表示。 |
| LOW，自审 | “请求结束就立刻释放全部显存”忽略在途写入和连接器。 | 对照真实延迟释放条件，明确本文停在 coordinator 调用边界，不宣称物理清零或块池内部结论。 |

独立代码代理未发现 CRITICAL／HIGH／MEDIUM／LOW 或必须修正项，确认 `j<=past+i`、空新块、K/V 新存储、旧缓存不变、多层完整状态、独立 SDPA 路径及生成的 `P+G-1` 约定一致。其审查范围仅为指定代码和说明，不能扩展为论文或全部引擎审查。

固定 Python 专项角色启动时因当前账户不支持其配置的 `gpt-5.4` 返回错误，没有产生审查结果；由已有可用独立代理完成上述窄范围审查。如实保留方式差异，没有把启动失败写成通过。

## 第二轮：教学表达与例子

正文先解释 token ID、embedding 和 logits，再从三行可见性图推到缓存。先说清每层有独立状态，再引入 Prefill／Decode、mask 的位置偏移和工作量账本；源码放在这些直觉之后。

使用实际随机模型的 `[2,5,1] → 7 → 12 → 9 → 9`，明确它没有自然语言能力；用均匀分数的 `[10,20,30,40]` 手算 20／25，避免读者在 mask 之前被复杂指数运算阻挡。图解是教学示意，不是 GPU trace。

两题的输入输出、约束、答案、运行命令、容差和边界齐全。解释 LayerNorm、残差和 FFN 的最小作用；注明 pre-norm／绝对位置教学模型与论文和当前 Llama 的差异。`torch.cat` 的历史复制成本、缓存来源的调用者责任、最终采样未 forward 的长度语义均已说明。

## 运行与归档验收

主代理运行 15 项 CPU 测试：含独立 SDPA、多个新 Query 的 mask、流式块、非连续输入、输入所有权、空块、每一新位置 logits、每层完整状态、旧状态不变、上下文和生成边界、非法输入及推理模式。初始缺少实现的 RED 日志见 [initial-red-tests.txt](evidence/runs/initial-red-tests.txt)；发布以 [最终 tests.txt](evidence/runs/tests.txt) 为准。

[lab.json](evidence/runs/lab.json) 保存两条生成路径和全部层的状态误差，18／6、52／21 是理论账本。正式运行命令、环境和 Ruff／格式结果见 [summary.json](evidence/runs/summary.json)。[归档检查](evidence/runs/archive-check.txt) 核对本地链接、固定源码行号和源码 SHA-256；[SHA256SUMS](evidence/SHA256SUMS) 可复核代码与资料原件。

## 发布结论

基础答案、实验和必要检查通过，无未解决 CRITICAL／HIGH。未验证部分为真实 vLLM 执行、GPU kernel、缓存块池内部、量化／RoPE 数值、性能、HBM 流量与峰值内存；不存在相关通过或加速结论。文章通过技术与教学自审，可按正式 Git 流程归档并推送。

勘误：暂无。上表为首次发布前的真实审查和测试修正记录。
