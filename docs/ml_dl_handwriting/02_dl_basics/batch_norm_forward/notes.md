# BatchNorm Forward Notes

## 核心公式

- $\mu_B=\frac{1}{m}\sum_i x_i$，$\sigma_B^2=\frac{1}{m}\sum_i(x_i-\mu_B)^2$。
- $\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$，$y_i=\gamma\hat x_i+\beta$。

## 易错点

- 把 LayerNorm 的 axis 用到 BatchNorm；BN 通常按 batch 维统计每个特征。
- 忘记 eps，方差很小时会除零或放大噪声。
- 训练态用 batch stats，推理态用 running stats，本模块只做训练态。
- batch size 很小时统计不稳定，效果可能变差。

## 面试追问

::: details 参考回答：BatchNorm 训练和推理有什么不同？

训练时 BN 使用当前 mini-batch 的均值和方差，并更新 running mean/var。推理时不能依赖单个 batch 的统计量，而是使用训练期间累积的 running stats 保证输出稳定。

:::

::: details 参考回答：BN 为什么对 batch size 敏感？

BN 的统计量来自 batch 维，batch size 太小时均值和方差估计噪声很大。这样会让归一化本身引入随机扰动，导致训练不稳定或推理统计量不可靠。

:::

::: details 参考回答：BN 的 gamma 和 beta 有什么作用？

gamma 和 beta 让模型在归一化后仍能学习合适的尺度和平移。否则归一化会强行限制表示分布，可能削弱某些层需要保留的幅度信息。

:::

::: details 参考回答：为什么 Transformer/LLM 更常用 LayerNorm 或 RMSNorm？

Transformer/LLM 的序列长度、batch 组成和自回归推理形态让 batch 统计不稳定，也不方便跨 token 使用。LayerNorm/RMSNorm 按样本内部维度归一化，不依赖 batch，更适合变长序列和小 batch 训练。

:::
