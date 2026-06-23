# Dropout Notes

## 核心公式

- inverted dropout 训练态：$y=\frac{m\odot x}{1-p}$，$m\sim Bernoulli(1-p)$。
- 推理态通常直接 $y=x$。

## 易错点

- 推理时还随机丢弃，导致输出不稳定。
- 训练时没有除以 keep probability，导致期望变小。
- `p` 和 `keep_prob` 混淆。
- mask 需要和输入 shape 可广播，并且通常只在训练态采样。

## 面试追问

::: details 参考回答：Dropout 为什么能缓解过拟合？

Dropout 训练时随机屏蔽部分激活，迫使模型不能过度依赖某些神经元组合。它相当于训练许多子网络的近似集成，因此能降低共适应和过拟合。

:::

::: details 参考回答：什么是 inverted dropout？为什么推理时不用再缩放？

inverted dropout 在训练时除以 keep probability，让激活的期望保持不变。这样推理时可以直接使用完整网络，不需要再额外乘缩放系数，部署更简单。

:::

::: details 参考回答：Dropout 和 BatchNorm 同时使用有什么注意点？

Dropout 会改变激活分布，而 BatchNorm 又依赖 batch 统计，两者顺序和强度不当可能让统计量变噪。实践中要谨慎调 dropout rate，现代架构里有时减少 BN 附近的 dropout。

:::

::: details 参考回答：Transformer 中 attention dropout 和 FFN dropout 分别作用在哪里？

attention dropout 通常作用在 attention weights 上，随机弱化 token 间依赖。FFN dropout 作用在前馈层激活或输出上，主要正则化逐 token 的非线性变换。

:::
