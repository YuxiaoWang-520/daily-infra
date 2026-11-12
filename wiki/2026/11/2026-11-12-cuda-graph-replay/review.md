# 审查记录

本篇回答 CUDA Graph 的提交开销、普通捕获约束及固定缓冲区生命周期问题。审查日期为 2026-11-12，时区为 Asia/Shanghai（UTC+08:00）。

文章状态为 published。主代理已完成技术证据与教学表达两轮自审，Python 代码由独立通用审查代理审查并复核修正。正式归档路径下重新运行测试，交付检查及校验和核验均已通过。

## 第一轮，技术事实与证据

执行者为主代理，方式为自审。

- [x] 选题承接第 06、09 篇，覆盖反复提交短 kernel 的工程场景。
- [x] 当天实际检索 PyTorch／NVIDIA 官方资料，更新官方 PyTorch 源码并固定完整 SHA。
- [x] 分别核对 v2.15.0 tag、当天 main 和本机 torch build，保留差异与版本范围。
- [x] 从 Python graph 上下文追踪到 C++ capture、instantiate、cudaGraphLaunch，保存相关源码及许可证。
- [x] 成本递推定义单 CPU、单有序 GPU 队列假设，完整测试对照逐操作模拟，5 行示例数据逐项复算。
- [x] 两道 CPU 练习有独立题面、起始代码、完整答案及真实运行记录。
- [x] 将 CPU storage 实验、模型微秒数字和待执行 CUDA 路径的范围写入正文及证据。
- [x] 核对当前 torch.cond／条件节点与底层图更新能力，普通 Python 分支和普通 replay 参数约束均有范围说明。
- [x] 验证 8 份源代码片段的原始 blob、行区间和归档哈希，合计 1443 行，均可从固定对象重建。

### 实际发现与修正

| 程度 | 实际观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | 非连续来源若与保留输入共享 storage，原 copy_ 触发重叠内存错误 | 在相同 storage 时生成 detached clone，再复制进固定输入 | 红灯真实复现，新增回归通过，独立代理复核通过 |
| LOW（note） | 独立审查发现 CUDA 示例的期望值继承默认 dtype／device，可能误报 | 固定期望 shape、CPU、float32，增加独立可运行的输出检查 | 原错误红灯复现，4 项辅助回归通过，独立代理关闭发现 |
| LOW（note） | 当前 main 对跨 stream replay 增加准备工作等待说明 | 在 CUDA 示例捕获结束后建立调用 stream 对 side stream 的等待 | 静态核对源码和调用顺序，CUDA 硬件执行待验证 |

首次基础答案以待实现代码运行，32 项用例产生真实错误。随后基础实现通过 32 项测试。审查补充回归后，最终 37 项 CPU 测试通过。红灯记录保留原始状态。[初始红灯](evidence/runs/01-red-tests.txt)、[dtype 回归红灯](evidence/runs/05-review-red-tests.txt)、[共享 storage 红灯](evidence/runs/07-overlap-red-tests.txt)、[最终测试](evidence/runs/04-tests.txt)

## 第二轮，教学逻辑与表达

执行者为主代理，方式为自审。

- [x] 用 10 个短 kernel 的例子开篇，先解释 stream、节点、依赖、capture／instantiate／replay。
- [x] 从队列递推到重复公式，再计算 501 次严格获益，数学假设和单位完整。
- [x] Mermaid 图展示准备、捕获、实例化、输入复制、图内执行和输出快照，图源与正文逐字匹配。
- [x] 固定地址通过变量引用、storage、copy_、借用输出和 clone 的实际 CPU 结果逐层解释。
- [x] 优化选择包含重复次数、图外复制、padding、多图显存、并发及重捕获成本。
- [x] 每道练习在答案前定义输入输出与验收条件，答案给出推导、运行命令和边界。
- [x] 面试追问覆盖基础、推导、实现、方案选择和排障。
- [x] 资料阅读笔记、源码路径、源文件片段、完整答案、真实输出和代码审查均已归档。

### 实际发现与修正

| 程度 | 实际观察 | 修改 | 复验 |
| --- | --- | --- | --- |
| MEDIUM（info） | 初稿的 October 前置阅读相对路径多退了一层 | 从最终归档路径重算为 ../../10/ | 归档交付检查逐个验证链接 |
| LOW（note） | 初稿 inline 数学定界符在生成时有缺失 | 修复完整定界符，按行核对并计数 | 全部 inline／display 数学定界符配对 |

## 运行与验证

执行机器为 wyx-macmini.local。解释器为 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python。环境 Python 3.12.16、PyTorch 2.15.0、macOS 27.0 arm64，CUDA 不可用。所有 Python 验证设置 PYTHONDONTWRITEBYTECODE=1，保持现有依赖环境。

在本篇目录执行下列命令。最终命令、工作目录、时间戳、退出码和代码哈希保存在验证 JSON 中。

    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m unittest discover -s code -p test_answers.py -v
    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff check --no-cache code
    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python -m ruff format --check --no-cache code
    PYTHONDONTWRITEBYTECODE=1 /Users/wangyuxiao/miniforge3/envs/ai-infra/bin/python code/cuda_demo.py

| 验证项 | 实际结果 | 记录 |
| --- | --- | --- |
| unittest | 37 项 CPU 测试通过，退出 0 | [原始输出](evidence/runs/04-tests.txt) |
| 归档路径复验 | 正式文章目录运行 37 项测试通过，退出 0 | [原始输出](evidence/runs/10-archive-tests.txt)、[执行记录](evidence/runs/10-archive-verification.json) |
| Ruff check／format | 7 个文件通过，Ruff 0.17.3，退出 0 | [Lint](evidence/runs/06-ruff.txt)、[格式](evidence/runs/06-format.txt)、[版本](evidence/runs/06-ruff-version.txt) |
| CPU lab | 5 组成本模型与缓冲区数据／指针观察通过断言，退出 0 | [JSON](evidence/runs/02-cpu-lab.json)、[stdout](evidence/runs/02-cpu-lab.stdout.txt) |
| CUDA 可用性 | 返回 cuda_unavailable、cuda_executed=false | [实际输出](evidence/runs/03-cuda-probe.json) |
| 代码哈希及完整命令 | 7 个 .py 的 SHA-256 与实际执行代码一致 | [验证清单](evidence/runs/08-verification.json) |
| 独立 Python 审查 | 初审 6 项定向测试、LOW 修正 4 项测试、重叠来源 1 项测试通过 | [审查范围和实际发现](evidence/python-review.md) |

## 发布结论

未解决的 CRITICAL／HIGH 问题为 0。基础练习、源码证据和两轮自审达到正文所声明范围的发布要求。固定 Python 专项代理调用失败后，由通用代理独立审查代码，执行方式已如实归档。

CUDA 捕获／重放正确性、目标 GPU 吞吐和延迟、条件节点、训练与分布式集成仍待相应设备及专项验证。正文的 CPU 教学验证与范围说明保留这些边界。

正式归档后的交付检查已通过，验证全部本地链接、24 篇连续索引、同日仅 1 篇、图源、数学定界符、源码重建、代码运行哈希和 5 行模型数据。SHA-256 清单覆盖 41 个文件，实际执行 shasum 校验退出 0。校验清单自身、交付检查结果及校验命令输出分别保存，避免互相包含产生循环。[交付检查](evidence/delivery-check.json)、[校验输出](evidence/runs/11-checksums.txt)

## 勘误与更新

暂无。
