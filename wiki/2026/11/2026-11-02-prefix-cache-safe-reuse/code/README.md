# 运行、题目与验收

两道练习都只依赖 Python 标准库，要求 Python 3.10+。不需要安装 PyTorch、vLLM 或 GPU 环境。本次实际解释器为 Python 3.10.13，macOS arm64；Ruff 0.15.9 是已有的可选静态检查工具，不是运行依赖。

| 文件 | 用途 |
| --- | --- |
| [models.py](models.py) | 冻结的 context、块与池快照，以及输入检查辅助函数 |
| [exercises.py](exercises.py) | 学生接口，默认抛出 `NotImplementedError` |
| [answers.py](answers.py) | 两道题的完整参考实现 |
| [test_answers.py](test_answers.py) | 29 个测试，含身份反例、计数守恒与容量失败回归 |
| [run_lab.py](run_lab.py) | 输出五组命中结果与完整元数据状态轨迹 |

## 执行

所有命令在本篇目录执行，即 `code/` 的上一层。

```bash
python -m unittest discover -s code -p test_answers.py -v
python code/run_lab.py
```

在 `exercises.py` 中完成自己的版本后，运行同一套验收：

```bash
EXERCISE_IMPL=exercises python -m unittest discover -s code -p test_answers.py -v
```

也可以在 `code/` 中新建 `my_solution.py`，改成 `EXERCISE_IMPL=my_solution`。`run_lab.py` 固定使用参考答案。两题任一未实现，相应测试报错是预期结果。

在已有 Ruff 的环境中，可额外运行：

```bash
python -m ruff check --select E,F,I,UP --ignore E501 code
python -m ruff format --check code
```

## 练习一：构造完整前缀的指纹

实现 `block_hashes(tokens, block_size, context) -> tuple[str,...]`：

1. `tokens` 是非空 list 或 tuple；每个 ID 是非负 Python `int`，拒绝 `bool`。块大小 B 是正 Python `int`，也拒绝 `bool`。
2. `context` 是 `PrefixContext(namespace, adapter, salt)`；三个字段均为非空字符串。本题把模型版本、位置与缓存表示等条件交给可信调用方放入 namespace，并没有自动解析这些条件。
3. 将 `["context-v1", B, namespace, adapter, salt]` 以 `json.dumps(value, ensure_ascii=True, separators=(",", ":"))` 编码，再用 UTF-8 字节求 SHA-256，得到十六进制根指纹。
4. 每个完整块求 `["block-v1", parent_hex, 当前块 token list]` 的相同指纹，并用结果更新 parent；不足一块的尾部不单独产键。
5. 按块顺序返回完整 64 字符键的 tuple，不修改输入。

实现 `reusable_tokens(tokens, B, context, available) -> int`：available 是非空字符串键的 set／frozenset，允许为空集合。按前面的完整键顺序查找，首个 miss 后停止；最多复用 `(L-1)//B` 块，以模拟基础普通生成的 logits 与整块对齐边界。非法输入抛出 `ValueError`。

验收至少覆盖：8 个共同位置命中 8，7 个共同位置命中 4，前文变化后局部重复不命中，不同模型域／适配器／salt 不命中，相同的 8-token 输入只跳过 4 个，以及缓存中有后缀键但缺前面的键。测试还改变 `PYTHONHASHSEED` 启动两个独立 Python 进程，以排除误用进程内 `hash()`。

这是一份自定义教学编码协议，不是 vLLM Pickle／CBOR 的字节级复现。它不计算 KV、不检测调用方传错的模型身份，也不提供权限认证。

## 练习二：共享完整块的生命周期

`PrefixPool(capacity)` 只处理完整块元数据。它不接收原始 token 长度或 logits 要求，因此练习一的 `L-1` 上限由上层查找策略负责；两题不是完整引擎的可直接拼接接口。以下 keys 表示本题已选定管理的有序完整块。

- 构造参数 capacity 为正 Python int，拒绝 bool；初始物理 ID 为 `0..capacity-1`，空闲队列也按这个顺序。
- `acquire(request_id, keys) -> tuple[int,...]`：request_id 为未占用的非空字符串；keys 为非空、有序、互异的非空字符串 tuple。先选连续 ready 前缀，遇到 miss 就为整个剩余尾部新分配。请求内的 key 唯一性反映本题的有序前缀标签约定，池本身不验证真实哈希链。
- 同 key 有多个 ready 物理副本时，优先选引用数大于 0 的副本，次序相同再按小 ID 选择。这样可以继续共享活跃块，保留空闲块供 miss 分配；这是本题策略，不是对上游副本选择策略的声明。
- 容量预检应扣除即将 touch 的零引用命中块。若可用块不足，抛出 `RuntimeError`，整份快照必须与调用前一致。
- 成功时先 touch 全部命中：零引用命中块移出空闲队列，引用数加 1。然后从空闲队首分配 miss；覆盖 ready 块前，只移除该物理 ID 的旧索引，不能删除同 key 的其他副本。
- 新分配块 `ready=False`，暂不提供缓存命中；`complete(request_id)` 把本请求的完整块声明完成并加入索引，可以重复调用。
- `release(request_id)` 按请求块逆序减引用。只有减到 0 的块入队：ready 块追加队尾，未完成块按收集到的逆序放队首；有其他请求引用的块继续保留且不入队。
- `snapshot() -> PoolView`：按物理 ID 返回 BlockView tuple、从队首到队尾的 free_order、按键排序且副本 ID 排序的 cache_entries，以及按请求 ID 排序的 requests。所有嵌套结构均不可变。
- 重复请求 ID、未知请求、无效参数抛出 `ValueError`，不改变状态。容量不足不能驱逐活跃块。完成后不修改已缓存内容；本题没有追加写入接口。

手算轨迹详见 [正文第 8 节](../README.md)。测试会从全部活跃请求重新统计每个物理块的引用数，并验证：空闲队列恰好是零引用块集合，缓存索引恰好对应 ready 块，失败操作保持原快照。

模型不涉及真实 KV、未满物理尾块、写时复制、异步 GPU 依赖、并发、跨实例传输或认证。真实引擎可提前登记已调度的块并结合执行依赖；本题采用更保守的显式 complete。Python deque 的中间移除是线性操作，不是对引擎数据结构性能的模拟。

## 真实运行记录

命令、工作目录、解释器、退出码见 [summary.json](../evidence/runs/summary.json)；[tests.txt](../evidence/runs/tests.txt) 包含 29 个通过测试，[lab.json](../evidence/runs/lab.json) 保存真实 hash／元数据输出。记录工具顺便列出了宿主机已有 PyTorch 版本，但本篇代码没有导入它。

参考实现及独立审查见 [answers.py](answers.py)、[Python 审查记录](../evidence/python-review.md)。所有数据均为公开机制下的合成标签，没有内部项目标识或真实用户内容。
