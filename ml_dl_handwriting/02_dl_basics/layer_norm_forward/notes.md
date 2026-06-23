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

::: details 参考回答：LayerNorm 和 BatchNorm 的统计维度有什么区别？

BatchNorm 通常对 batch 维统计每个特征的均值方差，依赖同 batch 样本。LayerNorm 对每个样本内部的特征维统计，不依赖 batch 中其他样本。

:::

::: details 参考回答：为什么 NLP/Transformer 中 LayerNorm 比 BatchNorm 常见？

NLP/Transformer 中 batch size、序列长度和 padding 经常变化，batch 统计不稳定。LayerNorm 按 token 或样本自身归一化，训练和推理行为一致，更适合自回归和小 batch 场景。

:::

::: details 参考回答：Pre-LN 和 Post-LN Transformer 的训练稳定性有什么差异？

Post-LN 把 LayerNorm 放在残差之后，深层时梯度路径更容易不稳定。Pre-LN 把 LayerNorm 放在子层之前，让残差分支提供更直接的梯度通道，因此深层 Transformer 更容易训练。

:::

::: details 参考回答：LayerNorm 的 gamma/beta 是否必须？去掉会怎样？

gamma/beta 不是数学上必须的，但它们提供可学习的尺度和平移，避免归一化过度限制表示。去掉后模型仍可运行，但表达灵活性和最终效果可能下降。

:::
