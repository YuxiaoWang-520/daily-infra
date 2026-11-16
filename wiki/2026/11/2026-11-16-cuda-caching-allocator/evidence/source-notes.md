# 源码读取与推导

本次读取固定 main 与稳定版源码，记录见 [版本](versions.json) 和 [清单](upstream/manifest.json)。

## 入口与候选选择

main 的 `NativeCachingAllocator::allocate` 获取设备和当前 stream，把正大小请求交给 `malloc`。后者进入设备分配器，舍入后选择池、计算新段大小，再按有序空闲块集合查找。`get_free_block` 的 lower_bound 后还有 stream、expandable 和过大块条件。单一进程的空闲总量无法包含这些匹配条件。

模型把所有请求固定在一个普通池与复用域，使用线性枚举取最小候选。它的大小条件是代码可见的教学规则，8 MiB 最小段与任意正余量分裂均由正文明确限定。

## 普通段回收与失败恢复

`malloc` 的普通失败路径包含回调、分配、可用缓存回收和全部可回收缓存重试。分配拒绝策略、graph capture 和池规则会改变分支。正文使用的模型保留“整段释放后重试”这一机制，失败后可回收缓存可能已归还，存活分配继续有效。

`release_blocks` 在普通段路径检查 prev／next，整段满足释放条件时才能归还。被活跃块钉住的空闲部分仍可能用于较小的合适请求。模型的 trapped_bytes 表示这种布局约束，并与 releasable_bytes 一起划分模型 inactive 总量。

## 事件与计数

`free_locked` 先减少 allocated 统计，跨 stream 使用记录会进入事件处理。`process_events` 查询事件，完成后调用 `free_block`。后者合并邻块，并减少 active 统计。默认池 `release_cached_blocks` 的非捕获路径会同步事件，再回收缓存，所以 empty_cache 可能有等待成本。

`free_block` 更新 inactive_split 时明确排除 expandable 段。读取 async backend 的统计还须遵守 memory.py 的适用性说明。将某个字段为零解释为所有布局问题都消失，会越过这些统计范围。

## 稳定版对照

`should_split` 和 `get_allocation_size` 的指定完整函数文本两版一致，比较结果与哈希单独归档。main 的释放入口增加了捕获追踪动作，正文限定了普通非捕获路径。这个局部比较只证明选定函数的关系。[具体范围](source-comparison.json)
