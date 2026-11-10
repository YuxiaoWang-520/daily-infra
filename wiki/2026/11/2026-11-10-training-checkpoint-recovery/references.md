# 参考资料与证据

主题：训练 checkpoint 的一致性恢复。实际检索日期为 2026-11-10（Asia/Shanghai）；检索请求、缓存获取日期、源码 fetch 见 [research-log.json](evidence/research-log.json)。文档服务返回的 acquired_at 为 11 月 6—7 日，本文没有把缓存获取日期说成今日实时页面生成日期。

## 关键论断与来源

| 编号 | 论断 | 一手资料与定位 | 本地归档 | 适用边界 |
| --- | --- | --- | --- | --- |
| S1 | 恢复训练需要 optimizer、epoch 等状态 | [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，general checkpoint | [文档笔记](evidence/docs-notes.md) | 应用还需补 RNG、数据消费和自身组件 |
| S2 | state_dict 是参数与 buffer 的浅引用 | [module.py 固定源码](https://github.com/pytorch/pytorch/blob/a68f117948121cdff578ff7813c5c906ff529b57/torch/nn/modules/module.py#L2203-L2220) | [源码片段](evidence/upstream/torch__nn__modules__module.py.txt) | 保存入口不会自动代表永久冻结 |
| S3 | 参数组按顺序装载；先构造 scheduler 再加载 optimizer | [optimizer.py](https://github.com/pytorch/pytorch/blob/a68f117948121cdff578ff7813c5c906ff529b57/torch/optim/optimizer.py#L934-L985)，state_dict 与 load_state_dict | [源码片段](evidence/upstream/torch__optim__optimizer.py.txt)、[源码阅读笔记](evidence/source-notes.md) | 固定模型与参数组；不验证任意改结构兼容性 |
| S4 | scheduler 有独立进度状态 | [lr_scheduler.py](https://github.com/pytorch/pytorch/blob/a68f117948121cdff578ff7813c5c906ff529b57/torch/optim/lr_scheduler.py#L182-L205)，state_dict／load_state_dict | [片段](evidence/upstream/torch__optim__lr_scheduler.py.txt) | 本例 StepLR 固定 step_size 与 gamma |
| S5 | staging 与整个保存有不同完成条件 | [2.15 DCP](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)、[官方异步示例](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)；AsyncSaveResponse、async_save、AsyncStager | [saver](evidence/upstream/torch__distributed__checkpoint__state_dict_saver.py.txt)、[staging](evidence/upstream/torch__distributed__checkpoint__staging.py.txt)、文档笔记 | 返回类型与 stager 配置有关；源码阅读，未运行 DCP |
| S6 | writer 等待写入结果后 finish 元数据，不能据此承诺整目录覆盖事务 | state_dict_saver._save_state_dict、FileSystemWriter.finish | [filesystem](evidence/upstream/torch__distributed__checkpoint__filesystem.py.txt)、源码笔记 | 默认 use_collectives=True；单 checkpoint 目录、新版本目录策略 |
| S7 | 缓冲文件先 flush，再 fsync；同文件系统成功 replace 原子可见 | [Python 3.12 fsync](https://docs.python.org/3.12/library/os.html#os.fsync)、[replace](https://docs.python.org/3.12/library/os.html#os.replace) | 文档笔记、[失败注入测试](code/test_checkpoint_io.py) | 未同步父目录，未测断电／强杀／网络文件系统 |
| E1 | 完整状态在新进程中接上连续训练；缺项对照可被检出 | 本篇原创合成 CPU 实验 | [lab.json](evidence/runs/lab.json)、[tests.txt](evidence/runs/tests.txt)、[验证说明](evidence/validation.md) | 固定 CPU、dtype、环境、数据与单进程边界 |

## 源码与版本

官方远端为 `https://github.com/pytorch/pytorch.git`，当日执行 fetch，固定默认分支 main 的完整 SHA `a68f117948121cdff578ff7813c5c906ff529b57`。既有缓存保持原 checkout，读取使用 git show，不把整个上游仓库加入归档。

稳定 release 查询为 v2.15.0（2026-10-28 发布），tag commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。实际二进制版本 torch 2.15.0，build Git `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。详见 [versions.json](evidence/versions.json)。

六个源码文件的完整 blob SHA256、选段范围和选段 SHA256 见 [manifest.json](evidence/upstream/manifest.json)。相应 BSD 许可证保存在 [LICENSE](evidence/upstream/LICENSE)。默认分支与稳定 tag 的实际差异保存在 [stable-to-main.diff](evidence/upstream/stable-to-main.diff)，阅读结论见源码笔记。只归档与本题相关的短片段和差异；教程与 Python 文档以自己的笔记归档，保留原始链接以遵守再分发边界。

## 论文记录

本篇研究持久训练状态、随机序列位置和文件发布语义，没有讲解某个论文提出的 checkpoint 优化算法。官方 API 语义、实际运行时源码与反例测试足以支撑主问题，因此不强行引入无关论文。Adam 的一阶矩公式仅用来说明优化器存在历史状态，不推导其收敛或算法性能；更完整优化器算法可单独专题研究。

## 可复现入口

[练习与答案](code/README.md) 提供完整代码、命令和验收条件；[运行摘要](evidence/runs/summary.json) 固定实际环境与已测代码哈希；[校验和](evidence/SHA256SUMS) 保护归档证据。图源 [checkpoint-lifecycle.mmd](assets/checkpoint-lifecycle.mmd) 为本篇原创机制示意，未把它描述为实际 trace。
