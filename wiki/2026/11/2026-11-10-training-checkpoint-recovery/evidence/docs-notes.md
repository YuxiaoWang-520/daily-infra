# 官方文档阅读笔记

访问请求日期：2026-11-10，Asia/Shanghai。实际 URL、provider acquired_at 与查询 ID 记录于 [research-log.json](research-log.json)。以下是本篇自己的概括，未复制页面全文。

## PyTorch：一般训练 checkpoint

[Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，general checkpoint 与 resume training 小节。保存模型权重之外，还需要 optimizer state、epoch 等训练信息。加载后需显式恢复训练模式；本例新构建模型默认 train，保留 Dropout。教程中的 weights_only=True 适用于 Tensor 与基础类型构成的状态字典，应用仍需自己的 schema 和来源检查。

本篇进一步把 RNG 与已消费数据位置纳入实验，不能声称教程会自动为任意 DataLoader 保存所有进度。model.state_dict 的浅引用风险由固定源码直接核对。

## PyTorch 2.15：DCP 与异步保存

[Distributed Checkpoint](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html) 描述多文件分片保存和 in-place load／resharding。它提供存储和状态装载机制；应用需要定义一致的训练状态和保存边界。AsyncSaveResponse 区分 staging_completion 和 upload_completion。DefaultStager 的默认部分选项依赖 CUDA，不能在 CPU 教学环境中假装运行 GPU staging。

[Asynchronous Saving with DCP](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) 的示例把 staging 等待与 upload 等待放在不同位置，说明拷贝完成后训练可以推进，持久保存仍可能尚未完成。后台 CPU／pinned memory 占用与同时未完成的保存数量需要控制。本篇只阅读该路径，没有测其速度或 GPU 内存开销。

## Python 3.12：文件发布

[os.fsync](https://docs.python.org/3.12/library/os.html#os.fsync) 对缓冲文件应先 flush，再 fsync；[os.replace](https://docs.python.org/3.12/library/os.html#os.replace) 在成功时原子替换，跨文件系统可能失败。因此练习的临时文件放在同目录。原子名字替换只覆盖可见性这层语义；本例未同步父目录，没有断电和任意文件系统持久性验收。

## 对正文结论的约束

文件完整、快照隔离、训练逻辑时刻一致，分别需要证据。单文件发布测试验证可捕获异常下旧文件保留；新进程训练对照验证固定环境下状态恢复；DCP 部分只给出源码与文档路径。三种证据不能互相冒充。
