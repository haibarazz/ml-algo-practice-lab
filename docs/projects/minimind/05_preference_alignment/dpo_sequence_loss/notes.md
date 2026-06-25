# DPO 序列级损失：让 chosen 相对 rejected 更可能笔记

## 关键公式与数据流

- $\Delta_\pi=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$。
- $\Delta_{ref}=\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x)$。
- $L=-\log\sigma(\beta(\Delta_\pi-\Delta_{ref}))$。

## 易错点

- chosen/rejected batch 顺序写反会把偏好信号反过来。
- 忘记 reference margin 会退化成简单偏好分类，约束变弱。
- logprob 应只统计回答 token，不能把 prompt 也算进去。

## 面试追问

::: details 参考回答：DPO 里的 reference model 起什么作用？

reference model 提供原模型的偏好基线，DPO 优化的是 policy 相对 reference 的偏好提升。它相当于隐式约束模型不要偏离原能力太远。

:::

::: details 参考回答：beta 变大或变小有什么影响？

beta 越大，同样的 margin 差产生更强梯度，训练更激进；beta 越小，更新更温和。过大可能过拟合偏好数据，过小可能学不动。

:::
