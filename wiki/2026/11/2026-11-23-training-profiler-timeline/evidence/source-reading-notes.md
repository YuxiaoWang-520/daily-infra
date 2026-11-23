# 固定源码阅读笔记

实际获取时间、远端、命令、完整 SHA 和 release 元数据在 [source-update.json](source-update.json)。每个原样片段的行号、源文件摘要、片段摘要和永久链接在 [source-manifest.json](source-manifest.json)。片段没有改写缩进或尾部空白，完整 [LICENSE](upstream/pytorch/LICENSE) 及源文件头一并保留。

## 从采样状态走到记录点

`torch/profiler/profiler.py` 中的 `schedule_fn` 对 `wait+warmup+active` 取模，最后一个 active step 返回 `RECORD_AND_SAVE`，repeat 结束后返回 NONE。`step()` 先关闭当前 step 标记，再计算 `next_step`、更新状态。动作转换表在 NONE→WARMUP 时 prepare、WARMUP→RECORD 时 start，RECORD_AND_SAVE→NONE 时 finalize。

本例编号为 `0:NONE,1:WARMUP,2:RECORD,3:RECORD_AND_SAVE,4:NONE`。回调在通知进入编号 4 时触发，所以不能以 callback 的 `step_num` 倒推“采集了第 4 步”。[schedule 片段](upstream/pytorch/schedule.txt)、[转换表](upstream/pytorch/schedule-transitions.txt)、[编号推进](upstream/pytorch/step-advancement.txt)

`prepare_trace` 为本篇普通路径创建 `torch.autograd.profiler.profile(use_kineto=True)`，底层 `_start_trace` 进入 `_enable_profiler`。C++ 的 TLS callback 注册后，`onFunctionEnterTLS` 从 `recordQueue` 取线程子队列并调用 `begin_op`。后者记录操作字段，并在允许 external correlation 时，为 user scope 或普通 scope 推入对应关联 ID。这是实际入口与记录点，不是凭符号名拼出的假定调用链。[准备](upstream/pytorch/prepare-kineto.txt)、[原生入口](upstream/pytorch/native-start.txt)、[注册](upstream/pytorch/callback-registration.txt)、[进入](upstream/pytorch/record-function-callback.txt)、[字段与关联](upstream/pytorch/correlation-push.txt)

## 为什么 self 不是 CPU 核心忙碌时间？

`_populate_cpu_children` 只将同步 CPU 事件按 `(thread,node_id)` 分组建立嵌套关系。`FunctionEvent.self_cpu_time_total` 对异步／非 CPU 事件返回 0，否则从 total 中减掉直接 CPU children 的 total。它没有从 OS 获取 CPU 核心执行周期，也不会自动扣除所有等待。非 legacy 的 CPU 事件 `device_time_total` 累计关联 kernels 与 children 的设备时长；这不是跨 stream 的时间并集。[父子范围](upstream/pytorch/cpu-parent-tree.txt)、[统计语义](upstream/pytorch/self-and-device-time.txt)、[self device](upstream/pytorch/self-device-time.txt)

## 怎样避免用错关联 ID？

`_parse_kineto_results` 先收集 linked correlation，再把对应设备活动挂到 CPU 事件；未关联 runtime／driver 等记录不能仅因数值 ID 相同被纳入。当前与 stable 源码都核对了 external ID 为 0 时的过滤。本文只解释 PyTorch 这里的路径，不把它扩展为所有采集器、跨进程或跨主机都可直接连接的统一 ID。[main](upstream/pytorch/correlation-link.txt)、[stable](upstream/pytorch/stable-correlation.txt)

## main、稳定 tag 与安装包的关系

main：`8aa44e99457abde4f2c0dd6156a09f5407ac90e0`。稳定 `v2.15.0`：`31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。对稳定版本另外读取了 schedule、self／device time、backend selection 和关联过滤；本篇用到的上述语义在两个固定版本都找到直接证据。没有声称两个完整实现逐字相同，也没有验证所有后端。[stable schedule](upstream/pytorch/stable-schedule.txt)、[stable self time](upstream/pytorch/stable-self-time.txt)、[stable backend](upstream/pytorch/stable-backend.txt)

两版都包含显式配置的 Cuspy 分支。本文选择条件不启用该分支，因此没有逐层研究其设备采集器。[main backend](upstream/pytorch/backend-selection.txt)

安装包 `2.15.0` 的构建 SHA 为 `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`，不能用相同版本字符串证明等于 release tag。运行证据只覆盖现有 CPU 安装包；代码片段按 fetched 固定版本解释。未构建最新 main，也未运行 CUDA、CUPTI、Cuspy 或 NCCL。
