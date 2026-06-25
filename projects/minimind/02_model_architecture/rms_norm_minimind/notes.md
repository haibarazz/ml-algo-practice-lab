# RMSNorm：只按均方根缩放的归一化笔记

## 关键公式与数据流

- $RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}$。
- $RMSNorm(x)=w\odot \frac{x}{RMS(x)}$。
- 与 LayerNorm 相比，RMSNorm 去掉 $x-\mu$，只保留尺度归一化。

## 易错点

- 把 RMSNorm 写成 LayerNorm，额外减均值会改变模型行为。
- eps 放错位置会影响数值尺度。
- weight shape 应匹配 hidden size 或 head dim，广播轴不能错。

## 面试追问

::: details 参考回答：为什么很多 LLM 使用 RMSNorm 而不是 BatchNorm？

BatchNorm 依赖 batch 统计，训练和自回归推理时分布不一致；RMSNorm 对每个 token 的 hidden 维归一化，不依赖 batch，适合变长序列和小 batch。

:::

::: details 参考回答：RMSNorm 相比 LayerNorm 少了什么不变性？

它保留尺度不变性，但不减均值，所以不具备平移不变性。输入整体加一个常数会改变 RMSNorm 输出，而 LayerNorm 会先去中心化。

:::
