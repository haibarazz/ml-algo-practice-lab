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

- Adam 和 SGD momentum 的区别是什么？
- bias correction 为什么必要？
- AdamW 为什么把 weight decay 解耦？
- Adam 在稀疏梯度、噪声梯度场景下有什么优势和风险？
