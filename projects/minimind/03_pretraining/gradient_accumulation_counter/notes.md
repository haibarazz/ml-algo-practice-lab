# 梯度累积：用小 batch 模拟大 batch笔记

## 关键公式与数据流

- 等效 batch size：$B_{eff}=B_{micro}\times accumulation\_steps\times world\_size$。
- 每个 micro-batch 使用 $L/N$ 反传，累积 N 次后得到平均梯度。

## 易错点

- 忘记除以 accumulation_steps，会让梯度尺度变大。
- 忘记处理尾部剩余 batch，会丢掉最后几步梯度。
- zero_grad 放错位置会清空尚未累积完的梯度。

## 面试追问

::: details 参考回答：梯度累积和直接增大 batch size 完全等价吗？

在没有 BatchNorm、dropout 随机性和分布式同步差异时，梯度均值接近等价。但优化器状态更新频率、随机性和日志 step 语义仍可能不同。

:::

::: details 参考回答：为什么梯度裁剪要放在 unscale 之后？

混合精度下梯度可能被 scaler 放大，直接裁剪会裁到错误尺度。先 unscale 再 clip，裁剪阈值才对应真实梯度范数。

:::
