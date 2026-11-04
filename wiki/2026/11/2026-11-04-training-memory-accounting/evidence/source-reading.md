# 固定源码阅读笔记

检索及字节核验时间：2026-11-04T11:16:30+08:00。官方 PyTorch main：`665ecdd5784fd7c8dfdf7dbb184fceb2b31c387f`，提交时间 `2026-11-04T02:58:20Z`。以下行号均对应当篇归档的原始文件；默认新建 optimizer 的条件不能推广到加载外部 state 或自定义实现。

## 1. Adam 状态从哪里来？

路径：`torch/optim/adam.py` 的 `Adam.step`（L216—L271）→ `_init_group`（L140—L212）→ `adam` 功能更新。

- L152—L161：只收集 `p.grad is not None` 的参数，拒绝稀疏梯度。`requires_grad=False` 本身不是这里的判断条件；冻结时是否已有梯度和 state 必须另查。
- L162—L190：仅在 `len(state)==0` 时创建状态。`step` 是标量；本篇显式 `capturable=False, fused=False` 时放 CPU。两个 moment 由 `torch.zeros_like(p, memory_format=torch.preserve_format)` 创建，没有另指定 dtype，因此普通新状态随参数 dtype。AMSGrad 多一个 `max_exp_avg_sq`。
- L241—L271：初始化／收集之后，把参数、梯度、两个 moment、step 和模式开关传入 `adam`。不是每轮都给每个元素重新分配两份历史数组。
- 没有从这个初始化路径找到“普通 autocast 自动新建额外 FP32 master 参数”的步骤。此结论限定在所读路径，不能推断其他训练框架没有 master copy。

本篇 CPU probe 核对三组实际配置与首次 step 前后；用另一个未参与 loss 的参数核对跳过初始化。源码阅读对应的是今天 main，实际可执行行为来自已安装的 2.9.1；两者没有被混称为同一个版本。

## 2. 论文的两个递推式对应哪几行？

普通 single-tensor 路径，`adam.py` L455—L476：

- `exp_avg.lerp_(grad, 1 - beta1)`：`m + (g-m)(1-β₁) = β₁m+(1-β₁)g`。
- `exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)`：`β₂v+(1-β₂)g²`。
- L529—L547：计算偏差修正项，构造平方根／分母，`param.addcdiv_` 更新参数。分母计算有临时结果，所以持久数组之和不是 step 峰值。
- L779—L800：foreach 批量构造 `exp_avg_sq_sqrt`，再执行后续更新。`optimizer.py` L272—L280 的参数说明明确提示 tensorlist 中间结果可能增加约一份参数大小的峰值。

没有对 foreach 与 fused 进行 CPU／GPU 性能比较，没有假定 foreach 在所有场景增加同一个固定字节数。AMSGrad 只用于状态数组数量的示例，不展开其算法与证明。

## 3. 清梯度到底清了什么？

`torch/optim/optimizer.py` L1108—L1172：`zero_grad` 默认 `set_to_none=True`，其分支执行 `p.grad=None`；另一分支对梯度做 detach／zero。此函数不遍历并删除 Adam 的 moment。

重要边界：设置属性为 None 只是删除该引用；若有其他对象引用旧 storage，它仍可能存活。数值为 0 的梯度与不存在的梯度，在 `step` 中也可能产生不同更新行为。正文不把 `zero_grad(None)` 当作简写，因为那个位置参数实际是 `set_to_none`，直接传 None 会误导调用。

## 4. autocast 的职责在哪里？

`torch/amp/autocast_mode.py` L51—L137：上下文选择 op-specific dtype，建议默认精度创建模型，只在 forward 和 loss 中启用 autocast；CPU BF16 示例保留默认模型创建方式。不能从计算上下文推断参数被原地改为 BF16，也不能说所有输出都使用同一 dtype。

`docs/source/notes/amp_examples.md` L7—L22、L38—L61：autocast 与 GradScaler 是不同组件。本文未实现 FP16 loss scaling，也未用此 CUDA 示例支持 BF16 一定需要或不需要某种 scaler 的广泛结论。

`docs/source/optim.md` L295—L350：当日文档提供 functional AdamW 的显式 BF16 moment 示例，参数和梯度保持 FP32。正文只据创建代码计算 `12P` 这种布局，不把它写成高层 Adam 默认值；未运行 functional 接口／fused CUDA kernel，未验证数值稳定性或 CUDA 支持范围。

## 5. 为什么为 backward 保存的张量不一定是新副本？

`torch/csrc/autograd/saved_variable.cpp` L15—L87：根据变量与输出关系保存原变量或 tensor data，并处理 hook 等条件。L113—L117 提供 `reset_data()`，释放保存的数据引用；这里不足以单独证明任意训练图何时释放所有激活。

`torch/autograd/graph.py` L297—L317：saved tensor 包含操作为了 backward 保存的数据。它可能是输入、参数或中间结果；保存槽位数与新分配 storage 数量不是同一个量。本文没有使用 hook 输出的逻辑大小总和冒充激活净增内存。

CPU 别名 probe 是额外直接证据：base／view 同 storage，clone 为新 storage；逻辑总和 128 bytes，唯一 storage 96 bytes。它没有模拟 CUDA 的异步释放或 allocator。

## 6. 三种内存口径为什么不能相加？

`docs/source/notes/cuda.md` L543—L563：caching allocator 统计的活跃张量内存与 allocator 管理总量不同，未用缓存仍会在驱动工具中显示占用。empty_cache 不释放活跃张量。

`torch/cuda/memory.py`：

- L222—L234：`empty_cache` 的说明与底层入口；部分碎片场景可能受益，但不扩大可用硬件容量。
- L530—L580：allocated／max allocated／reserved 的返回值对应不同统计键。
- L314—L321：requested bytes 与 allocated bytes 可以因分配取整产生差异，进一步说明逻辑张量大小与分配统计不能简单等同。

正文只给 GPU 排查方法，没有在当前无 CUDA 主机上伪造 GPU 数字。驱动占用、外部库分配和 allocator 统计还需要目标环境实测，不能从 CPU 张量载荷反推。

## 许可与覆盖范围

10 个原始文件共 399,859 bytes，与固定 commit 的 Git blob SHA-1 及本篇 SHA-256 清单逐项核对。保留 LICENSE 与版权头；未格式化原始代码。完整文件归档便于复核上下文，不代表整文件每个分支都执行或审计过。

原始资料：[manifest](source-manifest.json)、[版本和获取过程](versions.json)、[PyTorch LICENSE](upstream/pytorch/pytorch/LICENSE)。
