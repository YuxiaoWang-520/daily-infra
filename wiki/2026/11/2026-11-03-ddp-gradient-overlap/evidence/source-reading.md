# 源码阅读笔记

访问日期：2026-11-03（Asia/Shanghai）。官方 `pytorch/pytorch` 默认分支 main，完整 SHA `6ddf1c556aafe0295f682a2822d1510eda4beab2`；逐文件原文、命令与 hash 在 [manifest](source-manifest.json)。

## 本篇要验证什么

1. 普通固定进程组的 DDP 同步梯度，计算平均不等于自动按有效样本加权。
2. 梯度就绪 hook 怎样变成桶就绪、按序提交的异步归约，以及默认收尾。
3. `no_sync()` 怎样改变同步边界；初始桶、重建桶和旧设计文档有什么区别。
4. 当前代码对编译与手动收尾有哪些明确边界，哪些没有执行。

## 实际获取

新建 partial/shallow、无初始 checkout 的官方源码缓存，再按具体文件 sparse checkout 到固定 main SHA；没有拉权重、初始化子模块、编译或安装。缓存位于 `.cache/upstream/pytorch/`，读取前检查工作区干净。

第一次按旧文档扩展名读取 `docs/source/notes/ddp.rst` 返回 128；用实际 tree 确认当前文件是 `ddp.md`，修正 sparse 路径后全部 9 个来源读取成功。这不是核心证据缺失；没有拿旧路径占位。获取记录见 [versions.json](versions.json)。

稳定发行元数据为 v2.15.0，发布于 `2026-10-28T17:09:49Z`，对应 commit `31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457`。没有单独分析该 tag。当前主分支、稳定版元数据、实际 CPU 环境 PyTorch 2.9.1 是三套边界，不能相互替代。

## 入口与初始化：先保证副本可比较

- [`distributed.py:1074—1096`](upstream/pytorch/pytorch/torch/nn/parallel/distributed.py)：`init_sync=True` 时核对参数形状并同步模块状态，之后初始化 reducer。它不替用户切分输入；docstring 约 471 行明确要求用户负责分片。
- [`_ddp_init_helper:1387—1476`](upstream/pytorch/pytorch/torch/nn/parallel/distributed.py)：构建 bucket assignment、反转初始桶列表、建立 C++ Reducer。`compute_bucket_size_limits:128—162` 对 static graph 或不寻找 unused 的默认配置先用 `sys.maxsize`；所以不能说第一轮必然按指定 cap 拆出多个桶。
- [`reducer.cpp:189—208`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：给梯度累加器注册 hook。包装模型不替代 autograd，也不执行 optimizer 更新。

## 为什么会得到 rank 平均

- [`set_divide_factor:583—600`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：普通固定 world size 取进程组大小；join 的有效进程数分支不在本篇基础模型内。
- [`mark_variable_ready_dense:362—439`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：无自定义 hook 的普通密集路径，复制进桶时乘 `1/div_factor_`，或在已有桶 view 上做除法。batched copy 路径有合并除法，不能以单个分支冒充所有模式。
- [`run_comm_hook:987—1000`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：普通未注册自定义 hook 时进入 `_AllReduceBySumCommHook`。其 [`default_comm_hooks.cpp:46—60`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/default_comm_hooks.cpp) 对桶执行 SUM 并取得 future；它不是第二次自动除 world size。
- 这建立的是按 rank 平均的语义。源码没有从输入 loss 或数据中自动推断有效样本／有效 token 分母；`R*n_r/N` 是本文从目标函数推导出的调用侧修正。

## 从 hook 到按序异步归约

- [`autograd_hook:694—781`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：梯度累加后触发；累积但暂不需要同步时不进入 ready 归约流程；通常记录 ready 顺序并调用 `mark_variable_ready`。
- [`mark_variable_ready:923—984`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：找出桶与偏移，准备密集或稀疏梯度，扣减 pending；桶中所有所需梯度 ready 后进入桶就绪。
- [`mark_bucket_ready:1090—1115`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：`next_bucket_` 是统一的提交次序；编号更后的桶先 ready，不会因此越序提交。当前 skip-unused 分支受配置限制，本篇不启用。
- [`all_reduce_bucket:1002—1040`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：创建 GradBucket，提交通信并保留 future。局部提交不代表其他 rank 已到齐，模型中的 global-ready 也不是每个进程的 enqueue trace。
- [`finalize_backward:1799—1864`](upstream/pytorch/pytorch/torch/csrc/distributed/c10d/reducer.cpp)：默认自动 finalization 等待相关 future／work、解析结果、进行梯度 view／写回的后续处理。CUDA 中要结合 stream 依赖理解，不能把这里的 wait 解释成应用主动调用全设备 synchronize。CPU probe 没有验证 CUDA stream。

## 不能从旧设计图推断当前所有模式

- `docs/source/notes/ddp.md:3—7` 明说设计说明基于 v1.4。原论文基于 v1.5。尽管网页放在当前文档站点，仍不能跳过源码中的更新。
- `reducer.hpp:150—155` 与 `reducer.cpp:2061—2125`：合适配置下按观测到的反向梯度顺序重建桶，再同步桶定义；前端 `distributed.py:1806—1814` 在 forward 前调用重建。应看 warmup 后的实际桶和 trace，不把固定构造顺序当永恒规律。
- `distributed.py:1695—1722`：`no_sync` 暂设 `require_backward_grad_sync=False`，退出恢复原值。forward 必须也在上下文里；只有 loss.backward 在里面并不满足契约。
- `torch/_dynamo/backends/distributed.py:394—442`：DDPOptimizer 的现有路径通过分割图保留 bucket 对应的梯度可见边界。其适用配置与其他编译路径不同，本文未运行 torch.compile，不能宣称所有编译路径都会切桶或都会破坏重叠。
- `distributed.py:2656—2740`：当前 main 有显式手动 backward finalization API，默认仍为 False；需在所有桶 ready 后按其状态要求调用，不支持所有 reducer 配置。基础叙述限定自动收尾，未运行手动模式。

## 验证的边界

代码读取确认机制与分支存在；标准库练习只验证数学与固定输入时序。额外 Gloo 实验执行的是已安装的 2.9.1 默认 DDP，验证本机 CPU 两进程的梯度／参数结果，不是执行此 main、v2.15.0、CUDA/NCCL 或性能测试。实际网络设置与结果见运行记录。
