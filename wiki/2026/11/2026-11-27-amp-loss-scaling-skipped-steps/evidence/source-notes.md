# 从公式到源码的阅读路线

所有路径相对 pytorch/pytorch。固定 main：26bbfa344f2f4df9b06fcc980dba95fee81d5a19。逐片段的原文件 SHA256、行号、原字节摘要、永久链接在 [source-manifest.json](source-manifest.json)，不靠教学转述代替源码。

1. torch/amp/grad_scaler.py L210—222：scale 在输出所在设备延迟建立 scale 张量，将 loss 乘以 scale。对应链式法则，不是把已丢失的梯度从零恢复。
2. L270—301：遍历本优化器参数的梯度，跳过 None，按设备和 dtype 分组；普通不允许 FP16 梯度的分支抛 ValueError，调用 foreach 检查／反缩放内核。
3. L342—378：状态验证、求逆 scale、记录 found_inf_per_device；_maybe_opt_step 只在缓存标记合计为零时调用 optimizer.step。
4. L43—55、L414—427、L477—487：READY／UNSCALED／STEPPED。普通路径 READY 才自动 unscale，UNSCALED 使用缓存；custom AMP-aware optimizer 的分支另外处理，本文没有运行。
5. aten/src/ATen/native/cpu/AmpGradScalerKernels.cpp L97—121：相关标量路径检查输入有限性、设置标记，再乘 inverse scale。不能推成所有前向张量或所有后续梯度修改都会被检查。
6. Python L516—564 和 CPU L168—190：汇集本周期标记，非有限时 scale 回退、连续次数归零；有限周期累计到增长间隔后尝试增长，并防止 FP32 增长产生非有限 scale，最后重置每优化器状态。
7. Python L632—684：state_dict 包含五个 scaler 字段，load_state_dict 恢复它们；文档说明在 update 后保存。本文直接使用公共 state_dict 中公开输出的 _growth_tracker 键观察计数，没有读写对象的私有 tracker 张量。
8. torch/nn/utils/clip_grad.py L170—188：裁剪系数为 max_norm/(total_norm+1e-6)，再钳到不超过 1；数学教学版省略 epsilon 并标清。

源码支持的当前行为与本机安装包的 CPU 验证相符，但没有把二进制 build SHA 与 main SHA 混同。普通 SGD 的返回 None 无法判断跳步；计数来自本例覆写的普通 step 调用。多个优化器的局部决策不能由全局 scale 变化反推。

## 本地片段与许可

归档 13 个必要片段和官方完整 LICENSE。片段保持固定 Git 对象原始字节，具体 provenance 在清单中，未复制整个仓库。图源是作者自行编写的 Mermaid，数字是本篇合成输入，没有第三方图像、内部日志、私有实现或真实训练数据。
