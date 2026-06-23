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

- BatchNorm 训练和推理有什么不同？
- BN 为什么对 batch size 敏感？
- BN 的 gamma 和 beta 有什么作用？
- 为什么 Transformer/LLM 更常用 LayerNorm 或 RMSNorm？
