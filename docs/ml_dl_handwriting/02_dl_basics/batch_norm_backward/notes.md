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

- 手推 BN backward 的关键链路是什么？
- 为什么 `d_beta` 是上游梯度求和，`d_gamma` 要乘 normalized input？
- BN backward 中哪些量必须从 forward 缓存？
- BN 对 batch size 敏感会怎样影响训练稳定性？
