# Adam Optimizer Notes

## 核心公式

- $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$，$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$。
- $\hat m_t=m_t/(1-\beta_1^t)$，$\hat v_t=v_t/(1-\beta_2^t)$，$\theta_t=\theta_{t-1}-\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)$。

## 易错点

- 忘记 bias correction，训练早期步长会偏小。
- `t` 从 0 开始带入校正项会除零。
- `v` 应累计 `grad ** 2`，不是累计 `grad`。
- Adam 的 weight decay 和 L2 regularization 在自适应优化器中并不完全等价，AdamW 会解耦。

## 面试追问

::: details 参考回答：Adam 和 SGD momentum 的区别是什么？

SGD momentum 维护梯度的一阶动量，用历史方向平滑更新。Adam 同时维护一阶动量和二阶平方梯度，用二阶估计为每个参数自适应调整步长。

:::

::: details 参考回答：bias correction 为什么必要？

训练初期 `m` 和 `v` 从 0 初始化，会系统性偏小。bias correction 用 `1 - beta^t` 修正这个偏差，否则前几步的有效更新尺度会不符合预期。

:::

::: details 参考回答：AdamW 为什么把 weight decay 解耦？

传统 L2 正则把权重项加到梯度里，会被 Adam 的自适应二阶缩放影响，不再等价于简单权重衰减。AdamW 直接在参数更新中衰减权重，使正则强度更可控。

:::

::: details 参考回答：Adam 在稀疏梯度、噪声梯度场景下有什么优势和风险？

Adam 对稀疏梯度和噪声梯度友好，因为每个参数有自适应学习率，少更新参数也能获得较大有效步长。风险是有时泛化不如 SGD，学习率、weight decay 和 beta 设置不当也会导致训练后期不稳。

:::
