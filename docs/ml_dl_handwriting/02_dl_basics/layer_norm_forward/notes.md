# LayerNorm Forward Notes

## 核心公式

- $\mu=\frac{1}{d}\sum_j x_j$，$\sigma^2=\frac{1}{d}\sum_j(x_j-\mu)^2$。
- $y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$，统计量通常在每个样本的特征维上计算。

## 易错点

- 和 BatchNorm 混淆 axis；LN 不依赖 batch 维统计。
- 忘记 `keepdims`，gamma/beta 广播容易错。
- normalized_shape 和最后若干维不匹配。
- Pre-LN/Post-LN 的位置改变会影响深层 Transformer 的梯度流。

## 面试追问

- LayerNorm 和 BatchNorm 的统计维度有什么区别？
- 为什么 NLP/Transformer 中 LayerNorm 比 BatchNorm 常见？
- Pre-LN 和 Post-LN Transformer 的训练稳定性有什么差异？
- LayerNorm 的 gamma/beta 是否必须？去掉会怎样？
