# 固定源码阅读笔记

main：a68f117948121cdff578ff7813c5c906ff529b57。稳定 tag：v2.15.0／31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457。原文选段和 SHA256 见 [manifest](upstream/manifest.json)。

## 从训练对象到冻结快照

Module.state_dict 返回参数与持久 buffer 的引用，文档直接说明是浅拷贝。因而本文把“收集状态字典”和“独立、不可被后续训练改动的快照”分成两件事。CPU 教学答案用 deepcopy；测试实际确认继续一步后快照中的参数与优化器矩不变。

Optimizer.state_dict 保存 state 与 param_groups；load_state_dict 以参数组顺序配对，并提示先初始化 LRScheduler，再加载 optimizer state。模型结构／参数组变化不能仅靠名字猜测。scheduler.state_dict 排除 optimizer 对象本身，load_state_dict 更新其独立状态；StepLR 的 last_epoch 必须与已完成更新数对齐。

审查发现当前 CPU 同 dtype 下 optimizer.load_state_dict 可复用输入快照中的 Tensor。单独冻结 snapshot 并不能保证公开 restore 接口不反向污染快照；答案在加载 optimizer state 时再 deepcopy，新增重复恢复和输入状态不变的回归测试。此为实际本机行为，不将源码中的 copy 相关注释扩张为一律深拷贝的保证。

## 从 staging 到写入完成

AsyncStager 协议的目标是形成后续训练不会改动的 staged state。async_save 先调用 stage，再交给执行器。stage 返回 Future 时包装为 AsyncSaveResponse，包含 staging_completion、upload_completion；返回已完成状态的同步分支则返回 upload Future。调用方应理解实际配置与返回类型。

默认 collective 路径由 planner 形成保存计划，write_data 等待存储写入 Future，之后 finish_checkpoint 调用 writer.finish。关闭 collective 是单独的实验路径，不能套用默认跨 rank 完成语义。

FileSystemWriter.finish 汇总 storage metadata，写临时 metadata，按 sync_files 选项 flush／fsync，再切换名字；存在旧 metadata 时还会删除它。由此不能推导覆盖旧目录是整套 checkpoint 的原子事务。本文建议独立版本目录与应用 latest 指针，且明确这部分未运行多机故障实验。

## 默认分支和稳定版本的实际比较

[完整相关差异](upstream/stable-to-main.diff) 与 manifest 记录六个源文件的原始 blob 哈希。state_dict_saver.py 与 staging.py 完整内容在两 SHA 下相同；module.py 只调整 typing import 注释，lr_scheduler.py 调整 Self import；filesystem.py 增加 SAFETENSORS 路径的 contiguous 处理，finish 与正文相关默认参数未变。

optimizer.py 存在实质改动：main 引入 preserve_state_dtype_on_load 相关策略和 FutureWarning，默认转换、未来计划和 step 特殊处理需分别看版本。本文不跨版本断言全部 dtype 行为相同；练习只使用固定 CPU float64，且实际运行 build SHA 与 tag 不同。调度器初始化警告和参数组顺序约定在比较的两版都存在。

## 未执行边界

未执行 GPU staging、DCP 多 rank 保存、异步 Future 故障或断电恢复。源码笔记定位机制，不能作为吞吐、持久性或跨拓扑数值一致性的实测证据。
