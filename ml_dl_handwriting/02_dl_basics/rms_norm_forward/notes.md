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

- RMSNorm 相比 LayerNorm 省掉了什么计算？
- 为什么很多 LLM 使用 RMSNorm？
- RMSNorm 保留了什么不变性，又失去了什么不变性？
- RMSNorm 的 eps 应该放在哪里？
