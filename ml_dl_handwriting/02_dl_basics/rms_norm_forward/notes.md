# RMSNorm Forward Notes

## 核心公式

- $RMS(x)=\sqrt{\frac{1}{d}\sum_j x_j^2+\epsilon}$。
- $y=\gamma\frac{x}{RMS(x)}$；RMSNorm 去掉了 LayerNorm 的减均值步骤。

## 易错点

- 写成 LayerNorm，额外减了均值。
- 沿 batch 维求 RMS，而不是沿最后的 feature 维。
- 忘记 gamma 或 gamma shape 不能广播。
- RMSNorm 只保留 re-scaling，不提供 re-centering invariance。

## 面试追问

::: details 参考回答：RMSNorm 相比 LayerNorm 省掉了什么计算？

RMSNorm 省掉了减均值，只计算平方均值的根并做缩放。相比 LayerNorm，它少了一步均值计算和 re-centering，计算和实现都更轻。

:::

::: details 参考回答：为什么很多 LLM 使用 RMSNorm？

很多 LLM 使用 RMSNorm，是因为它在效果接近 LayerNorm 的同时更简单、更快，并且适合大规模训练的数值稳定需求。它保留了对整体尺度的归一化，通常足够稳定残差流。

:::

::: details 参考回答：RMSNorm 保留了什么不变性，又失去了什么不变性？

RMSNorm 保留 re-scaling invariance：输入整体乘一个常数后，归一化方向基本不变。它失去了 re-centering invariance，因为不会减均值，输入整体平移会改变输出。

:::

::: details 参考回答：RMSNorm 的 eps 应该放在哪里？

eps 通常放在平方均值内部再开方，即 `sqrt(mean(x^2) + eps)`，避免 RMS 极小时除零。也有实现放在分母外，但要保持训练和推理一致，并注意数值尺度差异。

:::
