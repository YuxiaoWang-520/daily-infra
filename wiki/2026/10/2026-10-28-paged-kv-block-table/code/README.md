# 运行分页 KV 练习

先阅读正文第 7 节的题目，再根据下面的契约自行实现，最后对照 [完整参考答案](answers.py)。

## 接口与约束

```python
slot_for_token(block_table, position, *, block_size, length, num_blocks) -> int
page_usage(lengths, *, block_size, bytes_per_token) -> dict[str, int]
PagedKVCache(*, num_blocks, block_size, head_dim)
```

- 所有尺寸与位置都是 Python 整数，拒绝 bool。页长、池大小及每 token 字节数为正；长度允许零。有效位置必须小于有效长度。
- `slot_for_token` 返回 token 槽号，非元素／字节地址；只检查被访问的表项。缺少项或非法块号报 `ValueError`，超过有效历史报 `IndexError`。教学池允许块号 `0`，拒绝负号索引。
- `page_usage` 返回 `tokens`、`blocks`、`unused_slots`、`content_bytes`、`allocated_bytes`。每请求独占页、按需分配；不计元数据、池预留、共享或 lookahead。
- `PagedKVCache.append(request_id, key, value)` 接受非空 CPU FP64 `[T,D]`、有限且同形状的 K／V，复制新行，不保留梯度，返回新增行的槽号。预检失败或容量不足不能改变有效前缀及所有权；不是 GPU 异步事务模型。
- `read` 返回按逻辑顺序收集的 K、V 副本；`length` 返回有效长度；`block_table` 返回 tuple 副本；`free_block_count` 查看空闲页数。非空字符串 ID 在第一次追加时建立，未知 ID 报错。
- `release` 归还独占页，未知／重复释放报 `KeyError`。FIFO 页队列是题设，当前 vLLM 还有 null block、引用、缓存淘汰及不同复用顺序。本例没有并发、共享或安全隔离。

## 运行

在本篇目录中，使用已有 Python 3.10+ 与 PyTorch 环境：

```bash
python3 -m unittest discover -s code -p 'test_*.py' -v
python3 code/run_lab.py
```

本次实际解释器为 `/Users/wangyuxiao/conda3/bin/python3`，Python 3.10.13、PyTorch 2.9.1、CPU，无 CUDA，未安装依赖。[环境、命令与退出码](../evidence/runs/summary.json)。

应看到 `23 tests` 通过。诊断包括：地址例子 `[12,13,14,15,20,21,22]`；`[5,3]` 的页账本为内容 `256`、已分配页容量 `384 byte`；A 的块表从 `[0,3]` 复用后变为 `[0,3,1]`。实际输出见 [lab.json](../evidence/runs/lab.json)。

测试包含页边界、无效尾槽、OOM 状态保持、跨请求互不干扰、返回副本、非连续输入、非有限值、默认设备，以及 100 步固定种子的追加／释放与独立连续 K／V 账本对照。SDPA 对照使用最后位置的一行 Query，全部给定 KV 都可见，`dropout_p=0.0`。

`read` 会 gather 数据，CPU 池初始化用 NaN 帮助暴露误读；释放后不清空旧存储。这是便于检查的语义模型，不是 GPU paged kernel、性能优化实现或多租户隔离组件。没有执行上游 vLLM。
