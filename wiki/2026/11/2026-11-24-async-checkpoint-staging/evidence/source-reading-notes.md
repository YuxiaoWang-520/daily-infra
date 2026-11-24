# 固定源码阅读笔记

实际获取：`2026-11-24T11:01:44.438331+08:00`。官方仓库为 `https://github.com/pytorch/pytorch`，默认分支 main：`0d8b36742ee52e2926f5c349cdae9e57c82ed2f0`；稳定 `v2.15.0`：`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。命令、输出和退出码在 [source-update.json](source-update.json)。

每个片段的源路径、完整 SHA、行号、源文件与片段摘要和永久链接在 [source-manifest.json](source-manifest.json)。原样片段保留空白；完整 [LICENSE](upstream/pytorch/LICENSE) 和 [源文件头](upstream/pytorch/source-headers.json) 保留。main 与稳定 tag 的许可证逐字节相同。

## 1. 默认值与实际选择路径

`state_dict_saver.async_save` 先检查已初始化的 process group 是否包含 CPU backend。没有显式 stager 时，writer 若满足 AsyncStager 协议就直接使用；否则创建 `DefaultStager(StagingOptions(False,False,False,False))`。接着 `_stateful_to_state_dict` 提取状态，调用 `stage`。

`StagingOptions()` 本身的默认值都是 True。因此不能把该类的默认构造等同于入口内部调用。`FileSystemWriter` 则继承 `BlockingAsyncStager`，其 `stage` 设定 copy-ahead 并调用父类 `stage`。[选择与提取](upstream/pytorch/save-default-stager-and-state.txt)、[选项默认](upstream/pytorch/staging-options-defaults.txt)、[文件 writer 配置](upstream/pytorch/filesystem-writer-defaults.txt)、[实际 stage](upstream/pytorch/filesystem-blocking-stager.txt)

## 2. CPU 上也要复制 storage

不缓存的 BlockingAsyncStager 先 `_create_cpu_state_dict`，再 `_copy_state_dict`；前者为普通 Tensor 新建对应存储，后者复制值。启用缓存时，会复用既有 CPU state cache，因此不得在上一份保存仍读取它时覆盖。本文显式关闭缓存并只保留一个请求。[暂存代码](upstream/pytorch/blocking-stage-copy-and-cache.txt)、[CPU 新存储](upstream/pytorch/new-cpu-tensor-storage.txt)

DefaultStager 的异步 staging 分支把 `_stage` 提交到线程池。非阻塞设备复制路径还要同步相应 stream／accelerator；“已经提交 copy”不等于“训练现在就能覆盖源数据”。本文只阅读这条代码，不在没有 CUDA 的本机启用这些 GPU 选项。[设备复制与同步](upstream/pytorch/default-stage-and-device-sync.txt)

## 3. 两个 Future 与真正的后台 save

`async_save` 把 `stage` 的结果交给选择的保存执行器。线程执行器先对原 staging Future 调用 `result()`，再调用正常的 `save`。因此，即使调用方拿到的 staging 报告有版本边界，线程仍消费原始暂存任务结果。线程 Future 结束后，执行器安排自身 shutdown；入口对自己创建的默认 stager 也安装清理回调。调用者提供的 stager 则由调用者管理。[提交与默认资源清理](upstream/pytorch/upload-dispatch-and-cleanup.txt)、[等待暂存](upstream/pytorch/worker-waits-stage.txt)、[线程生命周期](upstream/pytorch/thread-future-lifecycle.txt)

`stage` 返回 Future 时，入口构造 `AsyncSaveResponse`；返回字典时则按同步契约返回单个 upload Future。main 与稳定 tag 的 already-done 分支都直接设置面向调用者的 staging 结果，而未在该分支读原异常。本文探针验证安装包的相应表现：staging completion 成功、upload RuntimeError、无 metadata。源码观察与实测各自标记版本，没有把一份构建证明为另一份构建。[main](upstream/pytorch/two-futures-and-completion.txt)、[稳定 tag](upstream/pytorch/stable-completion.txt)、[结果](runs/results.json)

## 4. 数据完成、metadata 与持久性边界

正常 `save` 在调用 writer 的 `write_data` 后等待相应 Torch Future 完成，再进入 `finish`；协调分支汇集结果。`FileSystemWriter` 的数据同步路径 flush 后尝试 fsync，metadata 写入临时文件并进行相应同步，再 rename。其同步操作不受支持时会告警；覆盖已有 metadata 时还有先删除旧文件的动作。[写与完成](upstream/pytorch/write-wait-global-finish.txt)、[数据同步](upstream/pytorch/data-sync.txt)、[metadata](upstream/pytorch/metadata-publication.txt)

因此不把“一个数据文件已存在”“一个 rank 已写完”或“对旧目录进行覆盖”描述成完整应用事务。本文只用新建目录、本地 `sync_files=True` 和保存成功后的 load；未模拟宕机、掉电、对象存储发布或多 rank 失败。

## 5. 稳定 tag 与现有安装包

稳定 tag 的默认选择、Future 分支和 metadata 路径均另外实际读取并归档：[stable 默认](upstream/pytorch/stable-default-selection.txt)、[stable Future](upstream/pytorch/stable-completion.txt)、[stable metadata](upstream/pytorch/stable-metadata.txt)。没有说所有实现逐字相同，尤其文件其他偏移和修改不由片段比较推断。

本机安装包 `2.15.0` 的构建 SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`。未把版本字符串当成 tag／main 的构建身份证，也未构建最新 main。运行命令和警告保存在 [verification.json](runs/verification.json) 及其日志。
