# DPO Loss Notes

## 核心公式

- $L_{DPO}=-\log\sigma\left(\beta[(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))-(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))]\right)$。
- 其中 $y_w$ 是 chosen，$y_l$ 是 rejected，$\beta$ 控制相对 reference 的约束强度。

## 易错点

- `chosen`/`rejected` 顺序写反，loss 会鼓励坏答案。
- 忘记减 reference margin，退化成简单偏好分类。
- sigmoid/log 分开算容易数值不稳定，应使用 log-sigmoid。
- logprob 应只统计回答 token，prompt token 不应影响偏好差。

## 面试追问

::: details 参考回答：DPO 和 PPO/RLHF 的主要区别是什么？

PPO/RLHF 通常先训练 reward model，再用强化学习优化策略，并需要采样和 KL 约束。DPO 直接用偏好对构造分类式目标优化语言模型，流程更简单，不显式训练 reward model。

:::

::: details 参考回答：DPO 里 reference model 起什么作用？

reference model 提供一个基准偏好 margin，约束当前模型不要为了迎合偏好数据而偏离原模型太多。它相当于隐式 KL 约束的一部分，帮助保留原有语言能力和稳定训练。

:::

::: details 参考回答：beta 变大或变小会怎样影响训练？

beta 控制偏好 margin 的放大程度，也影响相对 reference 的约束强度。beta 太大时更新更激进，可能过拟合偏好；太小时信号变弱，模型变化不明显。

:::

::: details 参考回答：DPO 为什么可以看作不显式训练 reward model 的偏好优化？

DPO 从 RLHF 的最优策略形式出发，把 reward 差转化为策略和 reference 的 logprob 差。这样可以直接用 chosen/rejected 偏好优化模型，而不需要单独拟合一个显式 reward 函数。

:::
