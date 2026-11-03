# CPU Gloo 验证的环境边界

2026-11-03，已有 PyTorch 2.9.1、CPU float64、macOS arm64，进程数为 2。未安装或升级依赖。

首轮执行 `python code/ddp_probe.py --output evidence/runs/ddp.json` 时，没有显式覆盖 Gloo 网卡，进程组超时设为 30 s，也没有单独的诊断 AllReduce。在 DDP 构造的 `_verify_param_shape_across_processes` 处，rank 1 报 `gloo/transport/uv/unbound_buffer.cc` 等待 send 超时；spawn 随后终止另一进程。这里是根据真实工具输出保存的摘要，首轮完整 stderr 没有另外写成文件，不称为完整原始日志。

后续为 probe 增加了初始 SUM AllReduce 断言、阶段日志，将通信超时设为 60 s，并在本机显式设置 `GLOO_SOCKET_IFNAME=lo0`。复验成功，初始归约得到 3，五组梯度与参数检查均通过。独立代码审查者也在同一接口设置下实际运行成功。

这些变化没有逐个做单变量对照，因此不能断言首次失败只由接口选择或某个固定原因造成。实际最终命令、环境覆盖、退出码和完整捕获输出分别见 [ddp-command.json](runs/ddp-command.json)、[ddp.txt](runs/ddp.txt)、[结果](runs/ddp.json)。最终归档路径会再次运行。

`lo0` 是本机 macOS 的回环接口名，不是所有系统通用配置。这只验证本机两个 CPU 进程的通信与 DDP 数值行为，不代表跨机网络、GPU 后端或重叠性能。
