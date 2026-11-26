# 审查记录

主题：变长合批、packing 边界与损失归一化。

文章状态：published。技术证据与教学表达两轮均由主执行者自审，代码另有独立通用复核与定向修复复核；未将自审称为独立审查。时间与判据见 [self-review.json](evidence/self-review.json)。

## 第一轮：技术事实与证据

- [x] 主问题连接 padding、Attention 成本、真实样本边界和训练目标。
- [x] 当天实际 fetch PyTorch main，固定完整 SHA，区分稳定 tag、运行包构建和文档缓存日期。
- [x] 追踪 bool mask 转换、math dispatch、QK／mask／softmax 顺序、CE／NLL 分母和 varlen 参数与后端调用。
- [x] 阅读原论文 §3.1、§3.2、§3.3 的相关小节；区分 BERT／因果 LM、论文直方图算法／教学 FFD、CPU／GPU。
- [x] 输入 token、监督位置、token 槽、平方面积与允许位置对分别定义，可复算。
- [x] 两道完整答案已运行，25 项测试普通／优化模式通过；错误对照可见，边界输入已覆盖。
- [x] 成本表是算术模型，CPU 模型是功能验证，没有 GPU、吞吐或收敛数据。
- [x] 固定源码片段、许可、论文／官方笔记、真实输出与审查记录已经归档。

### 实际发现与修正

| 严重程度 | 观察与证据 | 处理 | 复验 |
| --- | --- | --- | --- |
| LOW（note） | 独立审查发现部分张量工厂依赖全局默认设备，meta 默认下违背 CPU 参考合同 | 补回归测试，先复现失败；为 metadata、模型参数、错误对照及 CLI 工厂显式固定 CPU | [失败记录](evidence/runs/meta-device-red.txt)、最终 25 项测试与独立定向复核通过 |
| LOW（note） | 首次 Ruff 有 4 项 I001 import 排序问题 | 只修正本篇文件并格式化 | [首次检查](evidence/runs/initial-ruff-check.txt)、[最终检查](evidence/runs/ruff-check.txt) |
| LOW（note） | 初稿以 NNLS 简称论文算法，容易混同优化问题与具体方法名 | 再读 §3.1，明确 NNLSHP／SPFHP，与本文 first-fit 的对象和选行规则区分 | [原始论文笔记](evidence/paper-notes.md) 与正文一致 |

其他技术实查：固定 dense pack 面积 200 没有被说成比桶化 146 更省；输入 5 个 token 与有效目标 3 个没有混同；mask 不承担标签边界或位置职责；已 shift 的 labels 不再移位；DDP 公式的标准平均假设、累计窗口和 0 计数条件明确；varlen 接口未运行，不以源码当实测。

14 个固定源码片段、445 行与 Git 原字节一致；varlen 公共参数与稳定版一致而完整实现不同。源码、数字、图与运行清单的交叉核验见 [technical-audit.json](evidence/technical-audit.json)。

## 第二轮：教学逻辑与表达

- [x] 从四条长度的手算开始，先区分 token 槽与 Attention 面积，再讲高级接口。
- [x] `[a,b,c]` 和 `[d,e]` 的 mask、位置、标签逐步推演，首次术语有解释和有效前置链接。
- [x] Mermaid 图与图源一致，表明输入关系与参考对照，没有声称真实 trace 或渲染验收。
- [x] 两题有输入、输出、限制、验收、完整实现、推导和真实运行记录。
- [x] FFD 非最优反例与 O(n²) 朴素寻找成本写清，不包装成论文算法。
- [x] 取舍表包含窗口重排、内存、目标权重、rank 对齐、后端与性能验证口径。
- [x] 追问覆盖基础、计算、实现、接口、标签、数值、分布式和排障。
- [x] 预归档与最终本地链接、占位符、文件及发布索引检查均通过，见 delivery-check.json。

除前述算法命名澄清外，教学自审没有发现额外需要修正的问题。重点检查了：样本独立性是本文选择的目标，不能把连续文档训练判成普遍错误；绝对位置实验不能外推其他位置编码；手算损失 3／4 与实测 loss 区分；固定原样本集合的局部对齐不能被说成任意有效 batch 改变后的训练曲线等价。

## 独立 Python 代码复核

专项代理 `/root/nov26_python_review` 在执行前因账户不支持 `gpt-5.4` 模型而失败。随后 `/root/nov26_python_fallback` 使用独立通用 Python／正确性／安全审查角色复核 5 个文件，未写仓库、安装依赖或操作 Git，不冒称专项身份。[结构化审查报告](evidence/python-review.json)

首轮独立报告：1,279 组有界穷举 FFD／成本 oracle、4 组纯 Python metadata／mask 对照、3 组手工 log-sum-exp CE／梯度、3 组 DDP SUM 尺度关系，以及真实 CPU SDPA、错误对照和 CLI 严格 JSON 均通过。发现的唯一 LOW 是默认设备依赖。

主执行者按测试先行修复后，原审查者只复核该 delta：在独立进程实际 `torch.set_default_device("meta")`，确认全部 6 个 metadata Tensor 与全部模型参数仍为 CPU，输出／损失／梯度误差仍小于 `1e-12`；CLI 在同一默认设备下输出与 results.json 一致。审查者精确清理自有临时文件和空目录，没有重复穷举或全套归档测试。该 LOW 已关闭，没有剩余 findings。

本节归档真实审查者报告，不伪造成原始 stdout；其范围是本篇 Python，不是正文／上游的独立审查。首轮与修复后的源码摘要分别保留，最终摘要与执行清单匹配。

## 实际运行

现有环境为 Python 3.12.16、PyTorch 2.15.0、Darwin arm64、Ruff 0.17.3，CUDA 不可用。没有初始化分布式进程组，没有安装或升级依赖。所有张量工厂和模型参数初始化按 CPU 参考合同显式固定设备。

| 验证对象 | 命令 | 实际结果 | 原始记录 |
| --- | --- | --- | --- |
| 基础答案 | `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s code -p test_answers.py -v` | 退出 0，25 项通过 | [普通模式](evidence/runs/unit-tests.txt) |
| 优化模式 | `PYTHONDONTWRITEBYTECODE=1 python -O -m unittest discover -s code -p test_answers.py -v` | 退出 0，25 项通过 | [优化模式](evidence/runs/optimized-tests.txt) |
| 代码检查 | `ruff check code`；`ruff format --check code` | 均退出 0 | [检查](evidence/runs/ruff-check.txt)、[格式](evidence/runs/ruff-format.txt) |
| 两题完整运行 | `PYTHONDONTWRITEBYTECODE=1 python code/run_answers.py --output-dir evidence/runs` | 退出 0，成本算术、CPU 数值对齐和错误对照符合记录 | [stdout](evidence/runs/answers-run.txt)、[结果](evidence/runs/results.json) |

实际命令使用本机 ai-infra 环境；清单用 basename 记录，避免归档个人绝对路径。[verification.json](evidence/runs/verification.json) 保留最终环境、退出码、5 个 Python 摘要及回归历史。初始两个答案 stub 失败见 [red-tests.txt](evidence/runs/red-tests.txt)，默认设备修复前的真实失败单独保留，不当作最终通过。

## 发布结论

技术与教学审查、必要验证及独立代码修复复核通过，无未解决 CRITICAL／HIGH。链接、文件与发布索引最终检查通过，见 [delivery-check.json](evidence/delivery-check.json)。CPU 对齐使用 `1e-12` 绝对误差，不宣传跨后端 bitwise 复现。

未执行 GPU、varlen、分布式通信、真实存储、吞吐、训练收敛、RoPE、dropout 或 MoE 验收。图仅检查源码和语义，没有渲染。Git 提交前基线见 [git-preflight.json](evidence/git-preflight.json)，同步成功以实际推送后 GitHub 查询为准，不提前写入未来提交。

## 勘误与更新

暂无。
