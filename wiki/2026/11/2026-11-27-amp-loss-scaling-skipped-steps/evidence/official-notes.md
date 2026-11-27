# 官方文档阅读笔记

本次于 2026-11-27 实际检索、提取下列固定版本官方文档。页面内容提取服务均返回 acquired_at=2026-11-26T00:00:00Z；请求日期与服务缓存日期分别保留，不能声称全部内容今天实时重新抓取。

## Automatic Mixed Precision examples（PyTorch 2.15）

[文档](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)

实际读取并用于本文的部分：Typical Mixed Precision Training、Gradient Clipping、Gradient Accumulation、Working with Multiple Models/Losses/Optimizers。

自行归纳：

1. autocast 包围前向和 loss，反向放在该上下文之外；scale(loss).backward 将缩放传给梯度。
2. 如需裁剪，先对该优化器 unscale_，再按原梯度单位裁剪；同一优化器的 step 周期内只能显式 unscale_ 一次。
3. 累计一个有效 batch 时 scale 保持不变，直到所有反向结束才反缩放／step／update。不同分母的 microbatch 仍需先定义正确的目标权重。
4. 多优化器各自 step 决策，所有相关 step 完成后才 update 一次。本文用两个普通 SGD 验证独立决策，没有外推 fused／任意自定义 AMP 优化器。

## Automatic Mixed Precision package（PyTorch 2.15）

[文档](https://docs.pytorch.org/docs/2.15/amp.html)

实际读取 Gradient Scaling 与 autocast／GradScaler API 说明。缩放不能保证所有模型都适合 FP16；scale 可能小于 1；BF16 的范围特点不等于没有任何数值风险。避免通过手工 half() 取代 autocast 的逐运算策略。

## 文档、源码与运行包的分界

- 文档固定到 2.15 URL，不拿 /main/ 文档的结果当稳定行为。
- 源码 main 完整 commit 是 26bbfa344f2f4df9b06fcc980dba95fee81d5a19，稳定 tag v2.15.0 是 31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457。
- main 构造器 device=None 选择当前 accelerator，稳定 tag 与本机签名默认 cuda。本篇显式传 cpu。
- 安装包的 build commit 为 d0f2af8b66b20da4baa50644c0cd3f887a2eae2e，不能称作运行了上述任一源码 SHA。
- 今天 fetch 后以 git show 固定对象读取，保留原有缓存 checkout，不切换其工作树。CPU AmpGradScalerKernels.cpp 在 main 与稳定 tag 全文件字节相同；grad_scaler.py 全文件不同。

本地保存自己的阅读笔记和提取元数据，未复制整页文档。关键状态和内核行为可由本地固定片段与运行结果交叉检查。
