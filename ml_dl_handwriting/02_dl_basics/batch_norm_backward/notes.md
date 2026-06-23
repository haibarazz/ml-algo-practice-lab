# BatchNorm Backward Notes

## 核心公式

- 紧凑写法：$dx=\frac{\gamma}{m\sqrt{\sigma^2+\epsilon}}\left(m\,dy-\sum dy-\hat x\sum(dy\hat x)\right)$。
- $d\gamma=\sum_i dy_i\hat x_i$，$d\beta=\sum_i dy_i$。

## 易错点

- 求和 axis 写错，batch 维和 feature 维混淆。
- 忘记乘 gamma，导致传回上一层的梯度尺度错误。
- backward 必须使用 forward 缓存的同一个 `var + eps`。
- 手推链路较长，推荐先写数值梯度校验。

## 面试追问

::: details 参考回答：手推 BN backward 的关键链路是什么？

关键链路是从 `y = gamma * x_hat + beta` 反传到 `x_hat`，再穿过标准化中的减均值、除标准差和方差计算。实现时最容易错的是 axis 和归一化分母，所以通常用紧凑公式或数值梯度校验。

:::

::: details 参考回答：为什么 `d_beta` 是上游梯度求和，`d_gamma` 要乘 normalized input？

`beta` 是加到每个样本上的平移项，所以它的梯度就是上游梯度沿 batch 维求和。`gamma` 乘在 normalized input 上，因此梯度要累加 `dy * x_hat`。

:::

::: details 参考回答：BN backward 中哪些量必须从 forward 缓存？

需要缓存 batch 均值、方差或 `inv_std`、normalized input、gamma，以及归一化时使用的 axis/eps。使用 forward 中同一批统计量很重要，否则 backward 和实际计算图不一致。

:::

::: details 参考回答：BN 对 batch size 敏感会怎样影响训练稳定性？

batch size 小会让统计量估计噪声大，backward 中梯度也会受这批样本的均值方差强耦合影响。结果可能是 loss 抖动、收敛变慢，甚至 train/eval 分布不一致。

:::
