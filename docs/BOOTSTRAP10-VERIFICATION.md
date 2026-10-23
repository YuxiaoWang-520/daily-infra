# 前 10 篇验收记录

2026-10-23（Asia/Shanghai），前 10 篇已按学习顺序归档，并逐篇提交、推送至私密仓库 `YuxiaoWang-520/daily-infra` 的 `main` 分支。[Wiki 索引](../wiki/index.md) 是正式阅读入口。

第一篇从 12 个数字、轴与索引、底层存储逐步引出 stride，再解释转置、连续性和 view。其余文章沿用“小例子 → 自然疑问 → 推导 → 源码 → 工程取舍 → 练习”的顺序；首次出现的基础概念就地说明，进阶实现放在基础模型之后。

## 内容与代码

| 篇目 | 核心内容 | 本次回归的 CPU 测试数 |
| --- | --- | --- |
| 01 | Tensor 的 shape、stride 与连续性 | 26 |
| 02 | 广播、零 stride 与输出物化成本 | 12 |
| 03 | FP32／BF16／FP16 与数值误差 | 19 |
| 04 | 数据搬运、算术强度与 Roofline | 12 |
| 05 | thread／warp／block 与访存映射 | 9 |
| 06 | GPU 异步执行、stream 与计时 | 9 |
| 07 | GEMM 分块与数据复用 | 11 |
| 08 | Tensor Core 的类型、形状与指令条件 | 9 |
| 09 | 算子融合的收益与资源代价 | 9 |
| 10 | Attention、online softmax 与分块 | 15 |
| 合计 | 10 个独立进程运行的测试套件 | 131 |

每篇的 `code/README.md` 说明运行方式；正文提供至少两道练习与详解，完整参考实现和测试保存在 `code/`。图解、资料笔记、固定 commit 的上游源码、获取记录、运行输出、校验和与审查记录均随篇归档。原始论文的精确版本、阅读位置和未保存原文的说明见各篇 `references.md`。

已有 Python 3.10+ 和 PyTorch 时，在仓库根目录运行整套练习测试：

```bash
bash .codex-longrun/init.sh
```

它不安装依赖，每篇使用新的 Python 进程，避免不同文章中同名模块相互影响。单篇实验需要先进入该篇目录，再按其运行说明执行。本次实际使用 `/Users/wangyuxiao/conda3/bin/python3`，环境为 Python 3.10.13、PyTorch 2.9.1、macOS arm64。

## 证据与边界

完整回归退出码为 0，131 项测试全部通过，输出见 [final-tests.txt](../.codex-longrun/final-tests.txt)。[final-audit.json](../.codex-longrun/final-audit.json) 保存逐篇提交、测试数、文件校验和与链接核验统计，并确认所有文章提交均包含在已核验的远端 `main` 中。

终验检查索引 01—10 的发布日期与发布状态、练习与答案入口、图解、源码快照校验和，以及源码永久链接的行号范围。链接检查覆盖本地目标文件是否存在；不把它称为所有外部网站的可用性检查或全部 Markdown 的渲染检查。

技术与教学两轮审查、实际发现、修正和复验见各篇 `review.md`。独立代码审查的范围按篇如实记录；本次整合终验由主代理完成，没有额外宣称一次独立的全套审查。

本机没有 CUDA。通过的是 CPU 参考实现、数值对照和成本模型；第 06 篇的 GPU 计时扩展示例实际返回 SKIP，记录已保留。没有据此宣称实际 CUDA／FlashAttention 执行正确、Tensor Core 指令已命中、HBM 流量下降、显存峰值或 GPU 加速比。
