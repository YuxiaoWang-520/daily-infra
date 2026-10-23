# 融合资料与源码笔记

核验：2026-10-23。实际固定源码为主，教程网页辅助阅读；没有运行 torch.compile 或 Triton GPU kernel。

## 当前实现链

PyTorch 官方 `main`：`d52f3687e72fac81e20fd7741ab1fd10e0b3899d`。

| 主张 | 固定 scheduler.py 定位 | 限定 |
| --- | --- | --- |
| 可行性包含内部规则、环和相关峰值内存状态检查 | [10414—10463 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/_inductor/scheduler.py#L10414-L10463) | 分支依赖状态，不说所有运行都启用所有检查。 |
| placement 与归约契约会阻止融合 | [10489—10534 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/_inductor/scheduler.py#L10489-L10534) | 不泛化成完整后端支持表。 |
| 共同依赖／buffer 访问是收益评分 | [11341—11365 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/_inductor/scheduler.py#L11341-L11365) | 注释中的 shared 是共同数据访问，不等于 CUDA shared memory。 |
| benchmark 并非每次必跑 | [7401—7472 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/_inductor/scheduler.py#L7401-L7472) | config、模板和后端有多种提前路径。 |
| 适用分支比较单独与融合时间，并处理 spill | [7915—7978 行](https://github.com/pytorch/pytorch/blob/d52f3687e72fac81e20fd7741ab1fd10e0b3899d/torch/_inductor/scheduler.py#L7915-L7978) | 静态代码说明该分支逻辑，不是本机实测。 |

Triton 官方 `main`：`1ec1eee840b8a35b15101e5ab9a4388e905291f3`。[softmax kernel 84—109 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/python/tutorials/02-fused-softmax.py#L84-L109) 与 [资源估计 142—170 行](https://github.com/triton-lang/triton/blob/1ec1eee840b8a35b15101e5ab9a4388e905291f3/python/tutorials/02-fused-softmax.py#L142-L170) 对应“少写回”与“更多资源需求”两面。

实际下载命令、时间、字节数与 SHA256：[source-manifest.json](evidence/source-manifest.json)。为了保留跨多个函数的上下文，归档 [完整 scheduler.py](evidence/upstream/pytorch/pytorch/torch/_inductor/scheduler.py) 与 [PyTorch LICENSE](evidence/upstream/pytorch/pytorch/LICENSE)；另存 [完整 softmax 教程](evidence/upstream/triton-lang/triton/python/tutorials/02-fused-softmax.py) 与 [Triton LICENSE](evidence/upstream/triton-lang/triton/LICENSE)。没有完整仓库 clone／构建。

可读入口：[Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。未引用网页历史性能表或接近四倍的比值作为实际 GPU 收益。

## 自己的推导

三步链 t=x+b、u=t*scale、y=relu(u)，假定完整中间量且每阶段一次读写，bias 独立元素一次计数：`6MN+N` 对 `2MN+N`。实际 HBM 可能因缓存、重读、spill、编译器优化而不同。

`launches*L+Q/B` 是解释用的合成模型，融合 slowdown 只惩罚内存项；不是 Roofline 下界、不拟合硬件、不保证 kernel 数。CPU chunked 程序也没有发生 GPU 融合，不能拿它的耗时评价 CUDA 收益。

本主题使用基础数据流与当前编译器实现，不附会额外论文算法；内存成本背景已在 Roofline 篇及其原论文笔记中记录。

验证：[环境与命令](evidence/runs/summary.json)、[9 项测试](evidence/runs/tests.txt)、[诊断](evidence/runs/lab.json)、[校验和](evidence/SHA256SUMS)、[归档检查](evidence/runs/archive-check.txt)。
