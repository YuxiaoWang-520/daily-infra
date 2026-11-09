# 官方文档阅读笔记

2026-11-09 实际读取指定官方资料。当前默认分支页面没有返回可读正文，当前实现改由当日 fetch 的固定 Git commit 阅读。稳定 API 与实验安装包分别核验。

## FSDP2 稳定 API

阅读 fully_shard 的生命周期、mesh、reshard_after_forward 和参数表示说明。文档中的默认 None 会自动区分根模块和非根模块。True 前向后重分片，False 保留完整参数。正文据此解释容量和反向前收集次数，再与 main 和稳定源码核对。

mesh 确定分片组及设备，参数用分片 DTensor 表示，计算时临时提供完整参数。本文选取一维分片组阐述基础机制，复杂 mesh 的配置需按各维度实际含义解释。[官方 FSDP2 2.15](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)

## Adam 稳定 API

阅读算法框和 betas、eps、weight decay、AMSGrad 参数。正文公式采用无 weight decay、无 AMSGrad 的常规路径，m／v 和偏差修正对应实际实验。

局部状态更新只有在收到同一全局目标的梯度后才能保持等价。练习比较连续多步的参数、梯度与 moments，避免单次参数检查遗漏尺度差异。[官方 Adam 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)

## 来源版本

官方文档以指定 2.15 URL 定位，网页内容可能随维护更新。本篇实现核心以完整 commit 和原始行号归档，稳定 tag 与本地安装包的两个选定入口文件做字节比较，结果在版本与来源清单中。
