# 源码阅读：从轮次计划到资源与时间记账

读取版本：vLLM 官方 `main`，`0c81a0371f586cc50a0daad6d216fbac21cf1a2d`。2026-10-29 实际 fetch 后固定；[版本记录](versions.json) 和 [逐文件命令与哈希](source-manifest.json) 为获取证据。本文未构建或执行这些上游文件。

## 1. 先从模型调用边界读起

[EngineCore.step，674—704 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/engine/core.py#L674-L704)：调用 `schedule()`，把 `SchedulerOutput` 交给执行器，必要时采样，然后 `update_from_output()`。这支撑“计划按轮发出”。`step_with_batch_queue()` 在 715 行开始是另一条组织路径；不能把普通 `step()` 中等待 future 的位置扩展成所有异步模式的设备同步保证。

本地原文件：[core.py](upstream/vllm-project/vllm/vllm/v1/engine/core.py)。

## 2. 进度、预算与 running 集合

- [699—722 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L699-L722)：注释用 `num_tokens_with_spec`、`num_computed_tokens` 统一表示进度；初始化发放预算与输入预算。不是否认模型存在 Prefill／Decode 的计算差异。
- [839—857 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L839-L857)：running 请求差额还加在途输出占位，受 long-prefill、两项预算、模型长度限制。正文的 `known - computed` 只适用于无投机、无在途、无缓存恢复等普通情况。
- [911—922 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L911-L922) 调 KV 分配；[996—1003 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L996-L1003) 成功后登记工作量并扣预算。
- [1845—1865 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1845-L1865)：调度后的记账会增加 computed 与 in-flight，后续拒绝或异常可能调整。这是不能把任意时刻的 computed 读成 GPU 完成计数的直接原因。

本地原文件：[scheduler.py](upstream/vllm-project/vllm/vllm/v1/core/sched/scheduler.py)。

## 3. waiting 准入不等于把所有等待请求塞入本轮

[1062—1082 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1062-L1082)：优先处理持 KV 的等待队列，再按状态处理普通 waiting；先检查有效 active 上限，并检查依赖。暂停的 streaming 会话也占运行器槽，本篇不模拟。

[1332—1347 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1332-L1347)：禁用 chunking 时，不足以覆盖完整工作会停止；允许时可以裁剪本轮工作量。练习二固定为完整 Prefill，专门辨认容量，不能称为当前 vLLM 默认调度复现。

[1566—1574 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1566-L1574)：成功后记 token、扣预算、切为 `RUNNING`。运行集合与本轮入选集合不同，[1618—1637 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1618-L1637) 有相应断言与解释。

官方优化文档对 Decode 优先作了概括。实际 running 可以含未完 Prefill，还有等待分队列、跳过、优先级等条件；正文采用实际遍历顺序描述，不宣称所有模式具有严格的全局 Decode 优先或 FCFS 保证。

## 4. 两组可选上限，以及默认值的证据边界

[配置 48—76 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/config/scheduler.py#L48-L76) 定义 `max_num_batched_tokens`、`max_num_scheduled_tokens`、`max_num_seqs`、`max_num_active_seqs`。真实创建配置还有额外决策，不能引用类常量断言所有在线部署默认大小。

[调度器 130—143 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L130-L143) 分别选取有效值；[配置 331—344 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/config/scheduler.py#L331-L344) 检查 token 上限至少覆盖序列上限，active 上限不超过运行器上限。`SchedulerOutput` 的 per-request 字典与总数见 [234—247 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/output.py#L234-L247)。

本地原文件：[配置](upstream/vllm-project/vllm/vllm/config/scheduler.py)、[调度输出](upstream/vllm-project/vllm/vllm/v1/core/sched/output.py)。稳定版只核对 tag SHA，没有核对这些字段在稳定版是否存在。

## 5. 分配、抢占、正常结束

waiting 的分配调用在 [1432—1454 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1432-L1454)。传入 `full_sequence_must_fit`，管理器 [517—547 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/kv_cache_manager.py#L517-L547) 可检查完整的当前已知序列，包含必要的准入限制。这里只做容量检查，不能称作立即物理分配所有未来输出页。[569—587 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/kv_cache_manager.py#L569-L587) 再结合 reserved／watermark 核验实际分配。

running 分配失败可触发 [930—980 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L930-L980) 的牺牲选择；[1812—1842 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L1812-L1842) 释放请求所持状态、computed 归零并重新入队。异步旧输出的交付／丢弃还有单独规则，正文不将其简化成无成本暂停。

正常输出由 [2745—2765 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L2745-L2765) 追加并检查停止；完成时在 [2470—2477 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L2470-L2477) 走回收；[3029—3065 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/core/sched/scheduler.py#L3029-L3065) 说明 connector 可延迟释放。不能据此保证“用户已结束就同步释放底层显存”。

本地原文件：[KV 管理器](upstream/vllm-project/vllm/vllm/v1/core/kv_cache_manager.py)、[停止检查辅助代码](upstream/vllm-project/vllm/vllm/v1/core/sched/utils.py)。

## 6. 指标必须绑定观察点

`IterationStats` [525—558 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/metrics/stats.py#L525-L558)：TTFT 从 request arrival 到前端统计观察，ITL 使用连续 engine-core 输出时间戳。两者不是客户端接收的同一只时钟边界。

[594—615 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/vllm/v1/metrics/stats.py#L594-L615) 计算请求级阶段耗时及每请求 TPOT；单输出记录 0。指标文档 [69—81 行](https://github.com/vllm-project/vllm/blob/0c81a0371f586cc50a0daad6d216fbac21cf1a2d/docs/design/metrics.md#L69-L81) 还指出输出事件可能带多 token、平均权重与 benchmark 纳入规则不同。正文数字例子使用同一观察点、每次一个 token，不把服务端统计当成客户端体验实测。

本地原文件：[stats.py](upstream/vllm-project/vllm/vllm/v1/metrics/stats.py)、[指标注册与观察](upstream/vllm-project/vllm/vllm/v1/metrics/loggers.py)、[指标文档](upstream/vllm-project/vllm/docs/design/metrics.md)、[优化指南](upstream/vllm-project/vllm/docs/configuration/optimization.md)。

## 7. 归档完整性

上游文件是 `git show SHA:path` 的原始字节，未改名替换、裁剪或运行格式化器；遵循 [原 Apache 2.0 许可证](upstream/vllm-project/vllm/LICENSE)。配置原文件两处、优化文档一处原有行尾空格，保留以匹配 SHA-256；文章目录的 `.gitattributes` 仅豁免 `evidence/upstream/**` 的空白检查，不豁免本篇正文或练习。源码清单和核心校验和仍核验全部来源字节。
